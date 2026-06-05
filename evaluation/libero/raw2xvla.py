#!/usr/bin/env python3
"""
raw2xvla.py
───────────
把吸盘"多零件按形状分拣"采集的 raw_hdf5（states+actions，无图像）离线 replay，
打包成 **X-VLA LiberoHandler** 直接可读的训练 hdf5（每条一个文件）+ 汇总 meta.json。

为什么从 raw 而不是 converted 打包：
  abs_action_6d 必须取 OSC controller 的目标位姿 (goal_pos / goal_ori)，
  converted_hdf5 没存这个，所以必须重新 replay 一次。replay 时同步渲染两路图像，
  一次产出 action + image，最省事也最准。

输出 schema（对齐 X-VLA datasets/domain_handler/simulations.py:LiberoHandler）:
  /abs_action_6d                              (T,10) f32 = pos3 + rot6d6 + grip1
  /observations/agentview_image               (T,H,W,3) u8  （已按 X-VLA 朝向 flip）
  /observations/robot0_eye_in_hand_image      (T,H,W,3) u8  （腕部不 flip）
  /language_instruction                       scalar bytes

依赖：在装了 libero/robosuite/mujoco 的环境跑（WSL conda env `vla-adapter`）。
X-VLA 仓库本身只在 Windows，故用 --libero-root 指向 WSL 的 libero 根，加进 sys.path。

用法（WSL）：
  /home/x/miniforge3/envs/vla-adapter/bin/python3 raw2xvla.py \
    --raw-root  /home/x/vla/libero/data/suction_dataset_multi_part_sorting/raw_hdf5 \
    --out-dir   /home/x/vla/libero/data/suction_dataset_multi_part_sorting/xvla_hdf5 \
    --meta-out  /home/x/vla/libero/data/suction_dataset_multi_part_sorting/xvla_meta.json \
    --libero-root /home/x/vla/libero
  # 调试只跑前 1 条： 追加 --limit 1
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np
import h5py


# ── 工具 ─────────────────────────────────────────────────────────────────────
def _decode(v):
    """h5py attr 可能是 bytes / np.bytes_ / str，统一成 str。"""
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "O"):
        return _decode(v.tobytes() if v.dtype.kind == "S" else v.item())
    return str(v)


def find_robots(env):
    """逐层向内查找具有 .robots 的 robosuite env（穿过 SuctionStickyWrapper 等）。"""
    cur = env
    for _ in range(8):
        if hasattr(cur, "robots") and getattr(cur, "robots"):
            return cur.robots
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    raise RuntimeError("找不到 robots（无法定位 robosuite env）")


def flip_agentview(img):
    """LIBERO agentview 朝向矫正（上下+左右翻转），与 X-VLA 训练/推理朝向一致。"""
    return np.flip(np.flip(img, 0), 1)


def mat_to_rotate6d(R):
    """3x3 旋转矩阵 → 6D（前两列），与 X-VLA libero_client / run_steel_plate_xvla 一致。"""
    R = np.asarray(R, dtype=np.float32)
    return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)


# 三任务规范指令：co-train 需统一指令；同时清洗早期数据 language 里的
# "shared stable v1" 等噪声后缀。按形状关键词映射（换位数据指令不变，同样命中）。
CANON_LANG = {
    "rectangular": "pick the rectangular steel plate and place it gently in the red bin",
    "round": "pick the round steel plate and place it gently in the blue bin",
    "triangular": "pick the triangular steel plate and place it gently in the yellow bin",
}


def normalize_language(raw_lang):
    low = raw_lang.lower()
    for key, canon in CANON_LANG.items():
        if key in low:
            return canon
    return raw_lang


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True,
                    help="raw_hdf5 根目录（递归收集 *.hdf5），或单个子目录")
    ap.add_argument("--out-dir", required=True, help="X-VLA 训练 hdf5 输出目录")
    ap.add_argument("--meta-out", required=True, help="汇总 meta.json 输出路径")
    ap.add_argument("--libero-root", default="/home/x/vla/libero",
                    help="WSL libero 根（加入 sys.path 以 import libero）")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--cap-index", type=int, default=5,
                    help="跳过开头 N 步（force sensor 不稳定的 settle 帧），与 create_dataset.py 一致")
    ap.add_argument("--suction-normal-max-angle-deg", type=float, default=10.0)
    ap.add_argument("--suction-effective-radius-ratio", type=float, default=0.7)
    ap.add_argument("--suction-detach-grace-steps", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 条（调试）")
    ap.add_argument("--keep-raw-lang", action="store_true",
                    help="保留 raw 原始 language（默认规范化为三条标准指令，清洗噪声）")
    ap.add_argument("--append-meta", action="store_true",
                    help="若 meta.json 已存在，追加 datalist 而非覆盖")
    args = ap.parse_args()

    # libero 上 sys.path
    if os.path.isdir(args.libero_root) and args.libero_root not in sys.path:
        sys.path.insert(0, args.libero_root)

    import libero.libero.envs.robots  # noqa: F401  注册 SuctionPanda
    import libero.libero.utils.utils as libero_utils
    from libero.libero.envs import TASK_MAPPING
    from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper
    from robosuite import load_controller_config

    raw_files = sorted(glob.glob(os.path.join(args.raw_root, "**", "*.hdf5"), recursive=True))
    if args.limit > 0:
        raw_files = raw_files[: args.limit]
    if not raw_files:
        print(f"[raw2xvla] 未找到 raw hdf5：{args.raw_root}")
        sys.exit(1)
    print(f"[raw2xvla] 待处理 {len(raw_files)} 条 raw。输出 → {args.out_dir}")

    suction_kwargs = dict(
        normal_max_angle_deg=args.suction_normal_max_angle_deg,
        effective_radius_ratio=args.suction_effective_radius_ratio,
        detach_grace_steps=args.suction_detach_grace_steps,
    )

    env = None             # 单例复用：所有 demo 同一个 problem_name
    env_problem = None
    datalist = []

    for k, raw_path in enumerate(raw_files):
        with h5py.File(raw_path, "r") as f:
            if "trajectory" in f:
                src = f["trajectory"]
            elif "data" in f:                      # 兼容 robomimic 单 demo
                src = f["data"][list(f["data"].keys())[0]]
            else:
                print(f"  [skip] 无 trajectory/data：{raw_path}")
                continue
            states = src["states"][()]
            actions = np.asarray(src["actions"][()])
            problem = json.loads(_decode(f.attrs["problem_info"]))
            problem_name = problem["problem_name"]
            language = problem["language_instruction"]
            if not args.keep_raw_lang:
                language = normalize_language(language)
            bddl_file = _decode(f.attrs["bddl_file_name"])
            bddl_content = _decode(f.attrs["bddl_file_content"])

        # 建 env（仅一次；problem_name 全相同）
        if env is None:
            # raw 里存的 bddl_file_name 可能已失效（早期数据的 bddl 被重命名/删除）；
            # 改用 raw 自带的 bddl_file_content 写临时 bddl 建 env，物体初始布局
            # 随后由 reset_from_xml_string(model_xml) 覆盖，故用哪份 bddl 不影响结果。
            tmp_bddl = os.path.join(tempfile.gettempdir(), "_raw2xvla_env.bddl")
            with open(tmp_bddl, "w") as _bf:
                _bf.write(bddl_content)
            controller_config = load_controller_config(default_controller="OSC_POSE")
            env_kwargs = dict(
                bddl_file_name=tmp_bddl,
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
                camera_heights=args.resolution,
                camera_widths=args.resolution,
            )
            env = SuctionStickyWrapper(TASK_MAPPING[problem_name](**env_kwargs), **suction_kwargs)
            env_problem = problem_name
        assert problem_name == env_problem, "problem_name 不一致，无法复用 env"

        # 重置到该 demo 的初始 mujoco state。不用 reset_from_xml_string：
        # 采集时 model_xml 含 center_marker 等额外 site，与 bddl 重建的 env 不一致会报错。
        # 直接覆盖 qpos/qvel 即可——物体集合与自由度顺序同采集 bddl，state(134) 维度匹配。
        for _ in range(5):
            try:
                env.reset()
                break
            except Exception:
                continue
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        ctrl = find_robots(env)[0].controller

        abs6d, av_imgs, eih_imgs = [], [], []
        for j, a in enumerate(actions):
            if j < args.cap_index:               # 跳过 settle 帧（不 step、不记录，对齐 create_dataset.py）
                continue
            obs, _, _, _ = env.step(a.tolist())
            gp = np.asarray(ctrl.goal_pos, dtype=np.float32)        # (3,)
            rot6d = mat_to_rotate6d(ctrl.goal_ori)                  # (6,)  goal_ori 是 3x3
            grip = np.float32(a[-1])                                # 吸盘 -1放/+1吸
            abs6d.append(np.concatenate([gp, rot6d, [grip]]))       # (10,)
            av_imgs.append(flip_agentview(obs["agentview_image"]))
            eih_imgs.append(obs["robot0_eye_in_hand_image"])

        if not abs6d:
            print(f"  [skip] 轨迹过短：{raw_path}")
            continue

        abs6d = np.stack(abs6d).astype(np.float32)                  # (T,10)
        av_imgs = np.ascontiguousarray(np.stack(av_imgs)).astype(np.uint8)
        eih_imgs = np.ascontiguousarray(np.stack(eih_imgs)).astype(np.uint8)

        # 输出路径：保留 raw 的子目录结构以避免换位/默认同名相撞
        rel = os.path.relpath(raw_path, args.raw_root)
        out_path = os.path.join(args.out_dir, os.path.splitext(rel)[0] + ".h5")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with h5py.File(out_path, "w") as o:
            o.create_dataset("abs_action_6d", data=abs6d)
            g = o.create_group("observations")
            g.create_dataset("agentview_image", data=av_imgs, compression="gzip")
            g.create_dataset("robot0_eye_in_hand_image", data=eih_imgs, compression="gzip")
            o.create_dataset("language_instruction", data=np.bytes_(language))
            o.attrs["source_raw"] = raw_path
            o.attrs["bddl_file"] = bddl_file
        datalist.append(os.path.abspath(out_path))
        print(f"  [{k+1}/{len(raw_files)}] T={abs6d.shape[0]:4d}  {os.path.basename(out_path)}")

    if env is not None:
        try:
            env.env.close()
        except Exception:
            pass

    # 汇总 meta.json（X-VLA datasets/dataset.py 读这个）
    if args.append_meta and os.path.exists(args.meta_out):
        with open(args.meta_out) as f:
            meta = json.load(f)
        meta["datalist"] = sorted(set(meta.get("datalist", []) + datalist))
    else:
        meta = {
            "dataset_name": "libero",
            "datalist": sorted(datalist),
            "observation_key": [
                "observations/agentview_image",
                "observations/robot0_eye_in_hand_image",
            ],
            "language_instruction_key": "language_instruction",
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.meta_out)), exist_ok=True)
    with open(args.meta_out, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[raw2xvla] 完成。打包 {len(datalist)} 条 → meta: {args.meta_out}")


if __name__ == "__main__":
    main()
