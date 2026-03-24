"""Evaluation script for required 4 setups on linxy/LaTeX_OCR:test."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_track_datasets
from .modeling import generate_from_messages, load_base_with_adapter, load_processor, load_vlm
from .prompts import build_chat_messages, build_one_shot_messages
from .utils import compute_metrics, safe_latex_from_generation, save_json, save_jsonl, set_seed

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 4 setups for handwritten formula OCR")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--linxy-config", type=str, default="human_handwrite")
    parser.add_argument("--cache-dir", type=str, default=None)

    parser.add_argument("--linxy-adapter", type=str, default="checkpoints/sft_linxy/adapter")
    parser.add_argument(
        "--combined-adapter",
        type=str,
        default="checkpoints/sft_linxy_mathwriting/adapter",
    )

    parser.add_argument("--max-test-samples", type=int, default=70)
    parser.add_argument("--one-shot-pool-size", type=int, default=64)
    parser.add_argument(
        "--one-shot-strategy",
        type=str,
        choices=["fixed", "nearest_visual"],
        default="fixed",
        help="How to select the in-context example for one-shot mode.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true")

    parser.add_argument(
        "--device-map",
        type=str,
        choices=["auto", "none"],
        default="auto",
        help="Use 'auto' only on CUDA. On MPS/CPU use 'none'.",
    )

    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_device_map(device_map_arg: str):
    if device_map_arg == "none":
        return None
    if device_map_arg == "auto":
        if torch.cuda.is_available():
            return "auto"
        return None
    return None


def choose_fixed_one_shot_example(train_ds):
    """Pick a representative example (not too short/noisy) for one-shot ICL."""
    rows = [train_ds[i] for i in range(len(train_ds))]

    def score(sample):
        latex = str(sample.get("latex", ""))
        n = len(latex)
        has_math_token = "\\" in latex
        # Prefer medium-length formulas with LaTeX commands.
        length_penalty = abs(n - 45)
        cmd_bonus = -20 if has_math_token else 0
        return length_penalty + cmd_bonus

    rows = sorted(rows, key=score)
    return rows[0]


def _image_signature(image) -> np.ndarray:
    """Compute a cheap visual signature for nearest-example retrieval in one-shot mode."""
    gray = image.convert("L").resize((96, 96))
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    ink = 1.0 - arr

    h, w = ink.shape
    mass = float(ink.sum() + 1e-6)
    ys, xs = np.indices((h, w), dtype=np.float32)

    cx = float((xs * ink).sum() / mass / max(w, 1))
    cy = float((ys * ink).sum() / mass / max(h, 1))

    h_var = float(ink.mean(axis=1).var())
    v_var = float(ink.mean(axis=0).var())
    ink_ratio = float(ink.mean())
    aspect = float(image.width / max(image.height, 1))

    return np.array([aspect, ink_ratio, cx, cy, h_var, v_var], dtype=np.float32)


def build_one_shot_pool(train_ds) -> tuple[list[dict], np.ndarray]:
    rows = [train_ds[i] for i in range(len(train_ds))]
    signatures = np.stack([_image_signature(row["image"]) for row in rows], axis=0)
    return rows, signatures


def pick_nearest_one_shot(query_image, pool_rows: list[dict], pool_signatures: np.ndarray) -> tuple[int, dict]:
    query_sig = _image_signature(query_image)
    dists = np.linalg.norm(pool_signatures - query_sig[None, :], axis=1)
    best_idx = int(np.argmin(dists))
    return best_idx, pool_rows[best_idx]


def run_inference_setup(
    *,
    setup_name: str,
    mode: str,
    base_model: str,
    eval_ds,
    output_dir: Path,
    max_new_tokens: int,
    temperature: float,
    bf16: bool,
    device_map,
    one_shot_strategy: str,
    one_shot_fixed_example: dict | None,
    one_shot_pool_rows: list[dict] | None,
    one_shot_pool_signatures: np.ndarray | None,
    adapter_path: str | None = None,
) -> dict:
    processor = load_processor(base_model)
    if adapter_path:
        model = load_base_with_adapter(
            base_model_name=base_model,
            adapter_path=adapter_path,
            prefer_bf16=bf16,
            device_map=device_map,
        )
    else:
        model = load_vlm(
            base_model,
            prefer_bf16=bf16,
            device_map=device_map,
        )

    refs: list[str] = []
    preds: list[str] = []
    rows: list[dict] = []

    for idx, sample in enumerate(eval_ds):
        image = sample["image"]
        reference = sample["latex"]

        selected_demo_idx = None
        selected_demo_latex = None

        if mode == "zero_shot":
            messages = build_chat_messages(image, target_latex=None)
        elif mode == "one_shot":
            if one_shot_strategy == "fixed":
                demo = one_shot_fixed_example
            else:
                selected_demo_idx, demo = pick_nearest_one_shot(
                    image,
                    one_shot_pool_rows,
                    one_shot_pool_signatures,
                )

            selected_demo_latex = demo["latex"]
            messages = build_one_shot_messages(
                demo["image"],
                demo["latex"],
                image,
            )
        else:
            raise ValueError(f"Unknown mode={mode}")

        raw_pred = generate_from_messages(
            model,
            processor,
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        pred = safe_latex_from_generation(raw_pred)

        refs.append(reference)
        preds.append(pred)
        rows.append(
            {
                "id": idx,
                "setup": setup_name,
                "prediction": pred,
                "reference": reference,
                "one_shot_strategy": one_shot_strategy if mode == "one_shot" else None,
                "one_shot_demo_idx": selected_demo_idx,
                "one_shot_demo_latex": selected_demo_latex,
            }
        )

        if (idx + 1) % 10 == 0:
            LOGGER.info("[%s] processed %s/%s", setup_name, idx + 1, len(eval_ds))

    metrics = compute_metrics(preds, refs)
    metrics["setup"] = setup_name

    pred_path = output_dir / f"predictions_{setup_name}.jsonl"
    save_jsonl(rows, pred_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main() -> None:
    configure_logging()
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_track_datasets(
        linxy_config=args.linxy_config,
        linxy_train_max_samples=args.one_shot_pool_size,
        linxy_test_max_samples=args.max_test_samples,
        mathwriting_max_samples=1,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )

    if len(bundle.linxy_train) == 0:
        raise RuntimeError("linxy train split is empty, cannot build one-shot prompt")

    one_shot_fixed_example = choose_fixed_one_shot_example(bundle.linxy_train)
    LOGGER.info("Fixed one-shot demo latex: %s", one_shot_fixed_example["latex"])

    one_shot_pool_rows, one_shot_pool_signatures = build_one_shot_pool(bundle.linxy_train)
    LOGGER.info(
        "One-shot pool prepared: size=%s, strategy=%s",
        len(one_shot_pool_rows),
        args.one_shot_strategy,
    )

    eval_ds = bundle.linxy_test
    device_map = resolve_device_map(args.device_map)

    all_metrics = []

    # 1) Zero-shot
    all_metrics.append(
        run_inference_setup(
            setup_name="zero_shot",
            mode="zero_shot",
            base_model=args.base_model,
            adapter_path=None,
            eval_ds=eval_ds,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            bf16=args.bf16,
            device_map=device_map,
            one_shot_strategy=args.one_shot_strategy,
            one_shot_fixed_example=one_shot_fixed_example,
            one_shot_pool_rows=one_shot_pool_rows,
            one_shot_pool_signatures=one_shot_pool_signatures,
        )
    )

    # 2) One-shot
    all_metrics.append(
        run_inference_setup(
            setup_name="one_shot",
            mode="one_shot",
            base_model=args.base_model,
            adapter_path=None,
            eval_ds=eval_ds,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            bf16=args.bf16,
            device_map=device_map,
            one_shot_strategy=args.one_shot_strategy,
            one_shot_fixed_example=one_shot_fixed_example,
            one_shot_pool_rows=one_shot_pool_rows,
            one_shot_pool_signatures=one_shot_pool_signatures,
        )
    )

    # 3) SFT on linxy
    linxy_adapter = Path(args.linxy_adapter)
    if linxy_adapter.exists():
        all_metrics.append(
            run_inference_setup(
                setup_name="sft_linxy",
                mode="zero_shot",
                base_model=args.base_model,
                adapter_path=str(linxy_adapter),
                eval_ds=eval_ds,
                output_dir=output_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                bf16=args.bf16,
                device_map=device_map,
                one_shot_strategy=args.one_shot_strategy,
                one_shot_fixed_example=one_shot_fixed_example,
                one_shot_pool_rows=one_shot_pool_rows,
                one_shot_pool_signatures=one_shot_pool_signatures,
            )
        )
    else:
        LOGGER.warning("Skip sft_linxy: adapter not found at %s", linxy_adapter)

    # 4) SFT on linxy + mathwriting
    combined_adapter = Path(args.combined_adapter)
    if combined_adapter.exists():
        all_metrics.append(
            run_inference_setup(
                setup_name="sft_linxy_mathwriting",
                mode="zero_shot",
                base_model=args.base_model,
                adapter_path=str(combined_adapter),
                eval_ds=eval_ds,
                output_dir=output_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                bf16=args.bf16,
                device_map=device_map,
                one_shot_strategy=args.one_shot_strategy,
                one_shot_fixed_example=one_shot_fixed_example,
                one_shot_pool_rows=one_shot_pool_rows,
                one_shot_pool_signatures=one_shot_pool_signatures,
            )
        )
    else:
        LOGGER.warning("Skip sft_linxy_mathwriting: adapter not found at %s", combined_adapter)

    summary_path = output_dir / "metrics_summary.json"
    save_json(
        {
            "results": all_metrics,
            "config": {
                "one_shot_strategy": args.one_shot_strategy,
                "one_shot_pool_size": args.one_shot_pool_size,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
        },
        summary_path,
    )

    df = pd.DataFrame(all_metrics)
    preferred_cols = [
        "setup",
        "num_samples",
        "exact_match",
        "norm_edit_similarity",
        "char_error_rate",
        "non_empty_rate",
        "latex_like_rate",
    ]
    cols = [c for c in preferred_cols if c in df.columns]
    df = df[cols]
    csv_path = output_dir / "metrics_summary.csv"
    df.to_csv(csv_path, index=False)

    LOGGER.info("Evaluation summary:\n%s", df.to_string(index=False))
    LOGGER.info("Saved JSON: %s", summary_path)
    LOGGER.info("Saved CSV: %s", csv_path)


if __name__ == "__main__":
    main()
