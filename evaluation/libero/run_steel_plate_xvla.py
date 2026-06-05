#!/usr/bin/env python3
"""
run_steel_plate_xvla.py
───────────────────────
把自定义"钢板放入框"吸盘任务（原本用 VLA-Adapter 训练）直接喂给 X-VLA 推理服务跑，
零样本（zero-shot）测试 X-VLA-Libero 权重在该自定义任务上的表现。

环境侧（仿真）：复用 min_bundle 的 libero —— SuctionPanda + SuctionStickyWrapper + 自定义钢板 BDDL。
策略侧（推理）：HTTP 请求 X-VLA deploy.py server（默认 127.0.0.1:8090 /act）。

动作对接：X-VLA 输出绝对位姿 (T,10)=[pos3, rot6d, grip1]，因此：
  - settle 后将 controller.use_delta=False（绝对模式）
  - rot6d -> axisangle，组成 env 需要的 [pos3, aa3, grip1]
  - grip 离散到 {-1,+1}；SuctionPanda: +1=吸盘启动(抓), -1=释放

用法（在 min_bundle 的 VLA-Adapter-main 目录下，vla-adapter 环境）：
  python run_steel_plate_xvla.py \
      --bddl-file /data/.../custom/pick_up_the_steel_plate_and_place_it_in_the_basket.bddl \
      --server_ip 127.0.0.1 --server_port 8090 \
      --num_episodes 3 --save-subdir steel_plate_xvla_zeroshot
"""
import argparse
import collections
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional

import imageio
import json_numpy
import numpy as np
import requests

# ── 让 min_bundle 自带的 libero（含 steel_plate 自定义物体）优先于已安装的 editable 包 ──
_VLA_ROOT = os.path.dirname(os.path.abspath(__file__))
# 脚本被放在 min_bundle/VLA-Adapter-main/ 下；libero 根在同级的 ../libero
_LIBERO_ROOT = os.path.join(os.path.dirname(_VLA_ROOT), "libero")
for _p in (_LIBERO_ROOT, _VLA_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# ── libero / robosuite ──────────────────────────────────────────────────────
import libero.libero.envs.robots  # noqa: F401  注册 SuctionPanda
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING
from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

EPS = 1e-6


# ── 6D 旋转 <-> 轴角（与 X-VLA libero_client 保持一致）──────────────────────
def rotate6d_to_axisangle(r6d: np.ndarray) -> np.ndarray:
    single = r6d.ndim == 1
    if single:
        r6d = r6d[None, :]
    a1 = r6d[:, 0:3]
    a2 = r6d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + EPS)
    b3 = np.cross(b1, b2, axis=-1)
    R = np.stack([b1, b2, b3], axis=-1)  # (N,3,3)
    out = []
    for i in range(R.shape[0]):
        quat = T.mat2quat(R[i])
        out.append(T.quat2axisangle(quat))
    out = np.stack(out, axis=0)
    return out[0] if single else out


def mat_to_rotate6d(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)


def flip_agentview(img: np.ndarray) -> np.ndarray:
    """标准 LIBERO agentview 朝向矫正（垂直+水平翻转），与 X-VLA 训练朝向一致。"""
    return np.flip(np.flip(img, 0), 1)


# ── 构建钢板吸盘环境（复用 run_suction_eval 的 collect 风格）─────────────────
def make_suction_env(bddl_file: str, resolution: int = 256):
    controller_config = load_controller_config(default_controller="OSC_POSE")
    problem_info = BDDLUtils.get_problem_info(bddl_file)
    problem_name = problem_info["problem_name"]
    env_kwargs = dict(
        bddl_file_name=bddl_file,
        robots=["SuctionPanda"],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        camera_names=["robot0_eye_in_hand", "agentview"],
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env = TASK_MAPPING[problem_name](**env_kwargs)
    env = SuctionStickyWrapper(env)
    env.seed(0)
    return env


def find_robots(env):
    """向内层 .env 逐层查找具有 .robots 的 robosuite env。"""
    cur = env
    for _ in range(6):
        if hasattr(cur, "robots") and getattr(cur, "robots"):
            return cur.robots
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    raise RuntimeError("找不到 robots（无法定位 robosuite env）")


# ── X-VLA HTTP 策略客户端 ────────────────────────────────────────────────────
class XVLAClient:
    def __init__(self, host: str, port: int, steps: int = 10, domain_id: int = 3):
        self.url = f"http://{host}:{port}/act"
        self.steps = steps
        self.domain_id = domain_id
        self.reset()

    def reset(self) -> None:
        self.proprio: Optional[np.ndarray] = None
        self.action_plan: Deque[List[float]] = collections.deque()

    def _format_query(self, agentview, wrist, robo_pos, robo_ori6d, goal) -> Dict:
        main_view = flip_agentview(agentview)
        wrist_view = wrist
        closed = np.concatenate([robo_pos, robo_ori6d, np.array([0.0])], axis=-1)
        closed = np.concatenate([closed, np.zeros_like(closed)], axis=-1)  # 20 维
        if self.proprio is None:
            self.proprio = closed
        return {
            "proprio": json_numpy.dumps(self.proprio),
            "language_instruction": goal,
            "image0": json_numpy.dumps(main_view),
            "image1": json_numpy.dumps(wrist_view),
            "domain_id": self.domain_id,
            "steps": self.steps,
        }

    def _post(self, payload: Dict) -> np.ndarray:
        resp = requests.post(self.url, json=payload, timeout=60)
        resp.raise_for_status()
        action = np.array(resp.json()["action"])  # (T,10)=[pos3,rot6d,grip1]
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"server 返回动作形状异常: {action.shape}")
        return action

    def step(self, agentview, wrist, robo_pos, robo_ori6d, goal) -> np.ndarray:
        if not self.action_plan:
            payload = self._format_query(agentview, wrist, robo_pos, robo_ori6d, goal)
            action = self._post(payload)
            self.proprio[:9] = action[-1, :9].copy()
            target_eef = action[:, :3]
            target_axis = rotate6d_to_axisangle(action[:, 3:9])
            target_grip = action[:, 9:10]
            final = np.concatenate([target_eef, target_axis, target_grip], axis=-1)
            for row in final.tolist():
                self.action_plan.append(row)
        a = np.array(self.action_plan.popleft(), dtype=np.float32)
        a[-1] = 1.0 if a[-1] > 0.5 else -1.0  # 吸盘: +1 抓 / -1 放
        return a


# ── 单 episode rollout ───────────────────────────────────────────────────────
def run_episode(env, robots, policy: XVLAClient, goal: str,
                max_steps: int, num_steps_wait: int = 10):
    policy.reset()
    obs = env.reset()
    frames: List[np.ndarray] = []
    success = False

    # 1) delta 模式下 settle（此时 use_delta 仍为默认 True，零动作不移动）
    for r in robots:
        r.controller.use_delta = True
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    # 2) 切绝对位姿模式（匹配 X-VLA 输出）
    for r in robots:
        r.controller.use_delta = False

    for t in range(max_steps):
        av = obs["agentview_image"]
        wr = obs["robot0_eye_in_hand_image"]
        frames.append(np.hstack([flip_agentview(av), flip_agentview(wr)]))

        ctrl = robots[0].controller
        robo_pos = np.asarray(ctrl.ee_pos, dtype=np.float32)
        robo_ori6d = mat_to_rotate6d(np.asarray(ctrl.ee_ori_mat, dtype=np.float32))

        action = policy.step(av, wr, robo_pos, robo_ori6d, goal)
        obs, reward, done, info = env.step(action.tolist())
        if done:
            success = True
            break
    return success, frames


def save_video(frames, save_dir: Path, ep: int, success: bool, slug: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y_%m_%d-%H_%M_%S")
    path = save_dir / f"{ts}--ep{ep:02d}--success={success}--{slug}.mp4"
    imageio.mimsave(path.as_posix(), frames, fps=30, output_params=["-pix_fmt", "yuv420p"])
    print(f"  [视频] {path}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bddl-file", required=True)
    p.add_argument("--server_ip", default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=8090)
    p.add_argument("--num_episodes", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--steps", type=int, default=10, help="X-VLA 每次请求返回的动作步数")
    p.add_argument("--domain_id", type=int, default=3)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--save-subdir", default="steel_plate_xvla_zeroshot")
    args = p.parse_args()

    _pi = BDDLUtils.get_problem_info(args.bddl_file)
    goal = (_pi.get("language_instruction") or _pi.get("language") or
            "pick the steel plate and place it in the basket").strip()
    slug = goal.lower().replace(" ", "_")[:50]
    print(f"[info] 指令: {goal}")
    print(f"[info] server: http://{args.server_ip}:{args.server_port}/act")

    env = make_suction_env(args.bddl_file, args.resolution)
    robots = find_robots(env)
    policy = XVLAClient(args.server_ip, args.server_port, steps=args.steps, domain_id=args.domain_id)

    save_dir = Path("rollouts") / args.save_subdir / time.strftime("%Y_%m_%d")
    n_success = 0
    for ep in range(args.num_episodes):
        print(f"\n[info] ===== episode {ep+1}/{args.num_episodes} =====")
        t0 = time.time()
        success, frames = run_episode(env, robots, policy, goal, args.max_steps)
        save_video(frames, save_dir, ep, success, slug)
        n_success += int(success)
        print(f"  success={success}  steps={len(frames)}  用时 {time.time()-t0:.1f}s")

    print(f"\n[summary] X-VLA 零样本钢板任务: {n_success}/{args.num_episodes} 成功")
    print(f"[summary] 视频目录: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
