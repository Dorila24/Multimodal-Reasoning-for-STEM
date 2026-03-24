#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="Qwen/Qwen2-VL-2B-Instruct"

# Setup 3: SFT on linxy
python -m src.train \
  --setup linxy \
  --base-model "$BASE_MODEL" \
  --linxy-config human_handwrite \
  --output-dir checkpoints \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --device-map none \
  --dataloader-num-workers 2

# Setup 4: SFT on linxy + MathWriting-human subset
python -m src.train \
  --setup linxy_mathwriting \
  --base-model "$BASE_MODEL" \
  --linxy-config human_handwrite \
  --mathwriting-max-samples 12000 \
  --output-dir checkpoints \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --max-seq-length 1024 \
  --warmup-steps 50 \
  --gradient-checkpointing \
  --device-map none \
  --dataloader-num-workers 4 \
  --cpu-threads 8

# Final evaluation (4 setups)
python -m src.evaluate \
  --base-model "$BASE_MODEL" \
  --linxy-config human_handwrite \
  --linxy-adapter checkpoints/sft_linxy/adapter \
  --combined-adapter checkpoints/sft_linxy_mathwriting/adapter \
  --max-test-samples 70 \
  --one-shot-pool-size 64 \
  --one-shot-strategy fixed \
  --device-map none \
  --output-dir outputs/eval_final

# Optional one-shot retrieval experiment
python -m src.evaluate \
  --base-model "$BASE_MODEL" \
  --linxy-config human_handwrite \
  --linxy-adapter checkpoints/sft_linxy/adapter \
  --combined-adapter checkpoints/sft_linxy_mathwriting/adapter \
  --max-test-samples 70 \
  --one-shot-pool-size 64 \
  --one-shot-strategy nearest_visual \
  --device-map none \
  --output-dir outputs/eval_nearest_oneshot

# Streamlit demo
streamlit run app.py
