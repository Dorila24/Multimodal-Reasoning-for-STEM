"""SFT training script for Qwen2-VL on handwritten formula OCR."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from .data import build_sft_train_dataset, load_track_datasets
from .modeling import VLSFTCollator, attach_lora, load_processor, load_vlm
from .utils import save_json, set_seed

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2-VL-2B for handwritten formula OCR")
    parser.add_argument("--setup", type=str, choices=["linxy", "linxy_mathwriting"], required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--linxy-config", type=str, default="human_handwrite")

    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--cache-dir", type=str, default=None)

    parser.add_argument("--linxy-train-max-samples", type=int, default=None)
    parser.add_argument("--linxy-test-max-samples", type=int, default=70)
    parser.add_argument("--mathwriting-max-samples", type=int, default=30000)

    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=2048)

    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument(
        "--device-map",
        type=str,
        choices=["auto", "none"],
        default="auto",
        help="Use 'auto' only on CUDA. On MPS/CPU use 'none'.",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=0, help="0 -> keep PyTorch defaults")

    return parser.parse_args()


def is_mps_available() -> bool:
    return bool(torch.backends.mps.is_available() and torch.backends.mps.is_built())


def resolve_device_map(device_map_arg: str):
    if device_map_arg == "none":
        return None
    if device_map_arg == "auto":
        if torch.cuda.is_available():
            return "auto"
        return None
    return None


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    use_mps = is_mps_available() and not use_cuda

    if use_mps:
        # Improves robustness when some kernels are missing on MPS.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # Helps avoid aggressive memory reservation that may look like freezing.
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
        torch.set_float32_matmul_precision("high")

    if args.cpu_threads and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, args.cpu_threads // 2))
        except RuntimeError:
            pass

    output_dir = Path(args.output_dir) / f"sft_{args.setup}"
    adapter_dir = output_dir / "adapter"
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_bundle = load_track_datasets(
        linxy_config=args.linxy_config,
        linxy_train_max_samples=args.linxy_train_max_samples,
        linxy_test_max_samples=args.linxy_test_max_samples,
        mathwriting_max_samples=args.mathwriting_max_samples,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    train_dataset = build_sft_train_dataset(datasets_bundle, args.setup)

    processor = load_processor(args.base_model)
    device_map = resolve_device_map(args.device_map)
    model = load_vlm(
        args.base_model,
        prefer_bf16=args.bf16,
        device_map=device_map,
    )
    model = attach_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    if use_mps and device_map is None:
        model = model.to("mps")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    collator = VLSFTCollator(processor=processor, max_length=args.max_seq_length)

    use_bf16 = bool(args.bf16 and use_cuda)
    use_fp16 = bool((not use_bf16) and use_cuda)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=use_cuda,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        optim="adamw_torch",
    )

    LOGGER.info(
        "Runtime: cuda=%s, mps=%s, device_map=%s, workers=%s, seq_len=%s",
        use_cuda,
        use_mps,
        device_map,
        args.dataloader_num_workers,
        args.max_seq_length,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_state()

    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    metadata = {
        "setup": args.setup,
        "base_model": args.base_model,
        "train_size": len(train_dataset),
        "linxy_train_size": len(datasets_bundle.linxy_train),
        "mathwriting_train_size": len(datasets_bundle.mathwriting_train),
        "eval_size": len(datasets_bundle.linxy_test),
        "adapter_dir": str(adapter_dir),
        "runtime": {"cuda": use_cuda, "mps": use_mps, "device_map": str(device_map)},
        "hparams": {
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "max_seq_length": args.max_seq_length,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
    }
    save_json(metadata, output_dir / "train_metadata.json")

    LOGGER.info("Training finished. Adapter saved to: %s", adapter_dir)


if __name__ == "__main__":
    main()
