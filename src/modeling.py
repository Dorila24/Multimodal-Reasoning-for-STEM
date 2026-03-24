"""Model loading, generation, and data collator for Qwen2-VL."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModel
except ImportError:  # pragma: no cover
    from transformers import AutoModelForVision2Seq as AutoVLMModel

from .prompts import build_chat_messages


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def pick_torch_dtype(prefer_bf16: bool = True):
    if torch.cuda.is_available():
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    # On Apple Silicon MPS, float16 usually gives much better speed/memory than float32.
    if torch.backends.mps.is_available():
        return torch.float16

    return torch.float32


def load_processor(model_name_or_path: str):
    return AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)


def load_vlm(
    model_name_or_path: str,
    *,
    prefer_bf16: bool = True,
    device_map: str | None = "auto",
):
    dtype = pick_torch_dtype(prefer_bf16=prefer_bf16)
    return AutoVLMModel.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )


def attach_lora(
    model,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    target_modules = target_modules or DEFAULT_LORA_TARGET_MODULES

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, config)
    return model


def load_base_with_adapter(
    *,
    base_model_name: str,
    adapter_path: str,
    prefer_bf16: bool = True,
    device_map: str | None = "auto",
):
    base = load_vlm(
        base_model_name,
        prefer_bf16=prefer_bf16,
        device_map=device_map,
    )
    return PeftModel.from_pretrained(base, adapter_path)


def _collect_images(messages: list[dict]):
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                images.append(part["image"])
    return images


@torch.no_grad()
def generate_from_messages(
    model,
    processor,
    messages: list[dict],
    *,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> str:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images = _collect_images(messages)
    if not images:
        raise ValueError("No image found in messages")

    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    if temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        gen_kwargs.update({"do_sample": False})

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_token_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    generated = processor.batch_decode(
        new_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return generated.strip()


@dataclass
class VLSFTCollator:
    processor: object
    max_length: int = 2048

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        images = [ex["image"] for ex in features]

        prompt_texts = []
        full_texts = []
        for ex in features:
            prompt_messages = build_chat_messages(ex["image"], target_latex=None)
            full_messages = build_chat_messages(ex["image"], target_latex=ex["latex"])

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for idx, plen in enumerate(prompt_lengths.tolist()):
            labels[idx, : int(plen)] = -100

        batch["labels"] = labels
        return batch
