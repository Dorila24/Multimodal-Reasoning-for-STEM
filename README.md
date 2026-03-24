# Track-10: Handwritten Formula OCR (Qwen2-VL-2B)

Repository for the technical task **Multimodal Reasoning for STEM**.

## Task Coverage

- Fine-tuned VLM for `image -> LaTeX` handwritten formula OCR.
- Trained on:
  - [linxy/LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR)
  - [deepcopy/MathWriting-human](https://huggingface.co/datasets/deepcopy/MathWriting-human) (subset)
- Evaluated on `linxy/LaTeX_OCR:test` (70 samples) for 4 required setups:
  1. zero-shot
  2. one-shot
  3. SFT (`linxy`)
  4. SFT (`linxy + MathWriting-human`)
- Streamlit app that uploads handwritten image and returns LaTeX + rendered formula.

## Final Metrics (saved run)

| Setup | Exact Match | NES | CER |
|---|---:|---:|---:|
| zero_shot | 0.6143 | 0.8905 | 0.1396 |
| one_shot | 0.3143 | 0.8069 | 0.2014 |
| sft_linxy | 0.9857 | 0.9974 | 0.0026 |
| sft_linxy_mathwriting | **1.0000** | **1.0000** | **0.0000** |

Saved metrics files:
- `reports/metrics_summary_final.csv`
- `reports/metrics_summary_final.json`

## Code Structure

- `src/train.py` - supervised fine-tuning (LoRA)
- `src/evaluate.py` - evaluation for all required setups
- `src/modeling.py` - model loading and generation
- `src/data.py` - dataset loading and normalization
- `src/prompts.py` - prompt templates (zero-shot / one-shot / train)
- `src/utils.py` - normalization, post-processing and metrics
- `app.py` - Streamlit application
- `reports/task1_report.md` - Task 1 report with screenshots

## Environment Setup

Recommended: Python 3.10 in a clean conda environment.

```bash
conda create -n hw10_qwen python=3.10 -y
conda activate hw10_qwen

python -m pip install --upgrade pip
pip install 'numpy<2'
pip install -r requirements.txt

export USE_TF=0
export TRANSFORMERS_NO_TF=1
```

## Training

### Setup 3: SFT on `linxy/LaTeX_OCR`

```bash
python -m src.train \
  --setup linxy \
  --base-model Qwen/Qwen2-VL-2B-Instruct \
  --linxy-config human_handwrite \
  --output-dir checkpoints \
  --num-train-epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --device-map none \
  --dataloader-num-workers 2
```

### Setup 4: SFT on `linxy + MathWriting-human`

```bash
python -m src.train \
  --setup linxy_mathwriting \
  --base-model Qwen/Qwen2-VL-2B-Instruct \
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
```

## Evaluation (all 4 setups)

```bash
python -m src.evaluate \
  --base-model Qwen/Qwen2-VL-2B-Instruct \
  --linxy-config human_handwrite \
  --linxy-adapter checkpoints/sft_linxy/adapter \
  --combined-adapter checkpoints/sft_linxy_mathwriting/adapter \
  --max-test-samples 70 \
  --one-shot-pool-size 64 \
  --one-shot-strategy fixed \
  --device-map none \
  --output-dir outputs/eval_final
```

Optional stronger one-shot experiment:

```bash
python -m src.evaluate \
  --base-model Qwen/Qwen2-VL-2B-Instruct \
  --linxy-config human_handwrite \
  --linxy-adapter checkpoints/sft_linxy/adapter \
  --combined-adapter checkpoints/sft_linxy_mathwriting/adapter \
  --max-test-samples 70 \
  --one-shot-pool-size 64 \
  --one-shot-strategy nearest_visual \
  --device-map none \
  --output-dir outputs/eval_nearest_oneshot
```

## Streamlit App

```bash
streamlit run app.py
```

In the app:
- `Base model`: `Qwen/Qwen2-VL-2B-Instruct`
- `LoRA adapter path`: `checkpoints/sft_linxy_mathwriting/adapter`

## Trained Checkpoints

Both trained adapters are included directly in this repository:
- `checkpoints/sft_linxy/adapter`
- `checkpoints/sft_linxy_mathwriting/adapter`

So the project can be run from a single GitHub location without external model-hosting dependencies.

## Report and Screenshots

- [Report](reports/task1_report.md)
- Screenshots:
  - Input photo:
    
    ![input photo](reports/screenshots/01_input_photo.png)
  - LATEX output:
 
    ![latex output](reports/screenshots/02_latex_output.png)
  - Rendered formula:
    
    ![rendered formula](reports/screenshots/03_rendered_formula.png)
