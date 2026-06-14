#!/bin/bash
# 在 H100 上用 LoRA 微调 X-VLA（90条分拣数据）
# 用法：bash run_train_h100.sh <meta_json_path>
# 示例：bash run_train_h100.sh ~/data/sorting_xvla_meta.json
#
# H100 benchmark 实测（2026-06-06）：
#   batch=16 → 24.2GB 显存，58.7 samples/s  ← 最优
#   batch=32 → OOM（80GB H100 + LoRA r=8 + bf16 autocast）
#
# 参数说明（90条数据）：
#   iters=3000  ≈ 90/16 * ~530 epochs
#   freeze_steps=200：前200步只训练 action heads，再解冻 VLM
#   warmup_steps=300：LR 线性预热
#   use_cosine_decay：推荐小数据防过拟合

set -e

META="${1:?请传入 meta json 路径，如: bash run_train_h100.sh ~/data/sorting_meta.json}"
MODEL_PATH="$HOME/data/X-VLA-Libero"
OUTPUT_DIR="$HOME/data/X-VLA-sorting-ckpt"
XVLA_DIR="$HOME/data/X-VLA"

cd "$XVLA_DIR"

PYTHONPATH="$XVLA_DIR:$PYTHONPATH" \
~/miniconda3/envs/lingbot-vla/bin/python peft_train.py \
  --models           "$MODEL_PATH" \
  --train_metas_path "$META" \
  --output_dir       "$OUTPUT_DIR" \
  --batch_size       16 \
  --iters            3000 \
  --freeze_steps     200 \
  --warmup_steps     300 \
  --learning_rate    2e-4 \
  --save_interval    500 \
  --log_interval     10 \
  --use_cosine_decay
