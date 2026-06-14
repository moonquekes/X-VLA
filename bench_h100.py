#!/usr/bin/env python3
"""
bench_h100.py — 在没有真实数据的情况下，测试 X-VLA LoRA 微调在 H100 上的最优 batch size 和内存占用。

用法（H100 上）：
  cd ~/data/X-VLA
  conda activate lingbot-vla
  python bench_h100.py --model_path ~/data/X-VLA-Libero
"""

import argparse, time, sys
import torch
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/root/data/X-VLA")

def make_batch(bs, seq_len, num_views, num_actions, dim_action, dim_proprio, device, dtype):
    """生成对应 Libero LiberoHandler 输出格式的随机 batch。"""
    vocab_size = 32000
    return {
        "input_ids":   torch.randint(0, vocab_size, (bs, seq_len), device=device),
        "image_input": torch.randn(bs, num_views, 3, 224, 224, device=device, dtype=dtype),
        "image_mask":  torch.ones(bs, num_views, dtype=torch.bool, device=device),
        "domain_id":   torch.full((bs,), 3, dtype=torch.long, device=device),  # libero=3
        "proprio":     torch.randn(bs, dim_proprio, device=device, dtype=dtype),
        "action":      torch.randn(bs, num_actions, dim_action, device=device, dtype=dtype),
    }


def bench_one(model, bs, seq_len, num_views, num_actions, dim_action, dim_proprio,
              device, dtype, n_warmup=3, n_bench=10):
    torch.cuda.reset_peak_memory_stats()
    # 用 autocast 做混精度，避免 modules_to_save fp32/bf16 混用问题
    amp_ctx = torch.autocast(device_type="cuda", dtype=dtype) if dtype != torch.float32 else torch.no_grad().__class__()
    try:
        for _ in range(n_warmup):
            batch = make_batch(bs, seq_len, num_views, num_actions, dim_action, dim_proprio, device, torch.float32)
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss_dict = model(**batch)
            loss = sum(loss_dict.values())
            loss.backward()
            model.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_bench):
            batch = make_batch(bs, seq_len, num_views, num_actions, dim_action, dim_proprio, device, torch.float32)
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss_dict = model(**batch)
            loss = sum(loss_dict.values())
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_bench

        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        return dt, peak_mem
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="X-VLA 权重路径（如 ~/data/X-VLA-Libero）")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    from models.modeling_xvla import XVLA
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")

    print(f"加载模型: {args.model_path} ...")
    # float32 加载，autocast 在 forward 时做混精度（和 accelerate 训练一致）
    model = XVLA.from_pretrained(args.model_path).to(device=device)

    lora_config = LoraConfig(
        lora_alpha=16, r=8, bias="none",
        target_modules="all-linear",
        modules_to_save=["transformer.soft_prompt_hub",
                         "transformer.action_encoder",
                         "transformer.action_decoder"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    # 从模型读取维度
    base = model.base_model.model if hasattr(model, "base_model") else model
    num_actions = base.num_actions
    dim_action  = base.action_space.dim_action
    dim_proprio = dim_action  # EE6DActionSpace 无独立 dim_proprio，默认等于 dim_action
    seq_len     = 50          # encode_language 典型长度
    num_views   = 2           # libero: agentview + wrist

    print(f"\n== 模型维度 ==")
    print(f"  num_actions={num_actions}, dim_action={dim_action}, "
          f"dim_proprio={dim_proprio}, seq_len={seq_len}, num_views={num_views}")
    print(f"  dtype={args.dtype}, GPU={torch.cuda.get_device_name(0)}\n")

    print(f"{'batch':>6} | {'s/iter':>8} | {'peak_mem':>10} | {'samples/s':>10} | 状态")
    print("-" * 55)

    best_bs, best_throughput = 1, 0.0
    for bs in args.batch_sizes:
        dt, mem = bench_one(model, bs, seq_len, num_views, num_actions, dim_action, dim_proprio, device, dtype)
        if dt is None:
            print(f"{bs:>6} | {'OOM':>8} | {'OOM':>10} | {'OOM':>10} | X")
        else:
            throughput = bs / dt
            flag = "<-- 最优" if throughput > best_throughput else ""
            print(f"{bs:>6} | {dt:>8.3f} | {mem:>9.2f}G | {throughput:>10.1f} | {flag}")
            if throughput > best_throughput:
                best_throughput, best_bs = throughput, bs

    # 推荐训练命令
    print(f"\n== 推荐配置（90 条 Libero 数据，LoRA, H100）==")
    # 约等于 90 条 × ~50 step/条 / bs ≈ 每 epoch step 数，跑 500 epoch
    steps_per_epoch = max(1, 90 // best_bs)
    iters = steps_per_epoch * 500
    save_interval = max(100, iters // 10)
    print(f"""
python peft_train.py \\
  --models ~/data/X-VLA-Libero \\
  --train_metas_path <YOUR_META_JSON> \\
  --output_dir ~/data/X-VLA-sorting-ckpt \\
  --batch_size {best_bs} \\
  --iters {iters} \\
  --freeze_steps 200 \\
  --warmup_steps 300 \\
  --learning_rate 2e-4 \\
  --save_interval {save_interval} \\
  --log_interval 10 \\
  --use_cosine_decay
""")


if __name__ == "__main__":
    main()
