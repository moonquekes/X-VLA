#!/bin/bash
# r14: 从 r12 热启动定向修补——swap 加权 meta(cf 仅 D) + 低 lr 短训。
# 目标: 保住 r12 的 round-A(8/8) 同时把 B 从 16.7% 救回来。短步+低lr+多存防过拟合/遗忘。
set -e
META="$HOME/data/sorting/sorting_meta_r14.json"
MODEL_PATH="$HOME/data/X-VLA-Libero"
OUTPUT_DIR="$HOME/data/X-VLA-sorting-ckpt-r14"
XVLA_DIR="$HOME/data/X-VLA"
# 热启动: 从 r12 已攻破 round-A 的 adapter 继续训
export XVLA_RESUME_LORA="$HOME/data/X-VLA-sorting-ckpt-r12cf/ckpt-7500"
cd "$XVLA_DIR"

PYTHONPATH="$XVLA_DIR:$PYTHONPATH" \
~/miniconda3/envs/lingbot-vla/bin/accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  peft_train.py \
  --models           "$MODEL_PATH" \
  --train_metas_path "$META" \
  --output_dir       "$OUTPUT_DIR" \
  --batch_size       16 \
  --iters            1500 \
  --freeze_steps     20 \
  --warmup_steps     50 \
  --learning_rate    3e-5 \
  --save_interval    250 \
  --log_interval     10 \
  --use_cosine_decay
