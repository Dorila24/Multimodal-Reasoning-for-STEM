"""Streamlit app for handwritten formula OCR using trained Qwen2-VL model."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import streamlit as st
from PIL import Image

from src.modeling import generate_from_messages, load_base_with_adapter, load_processor, load_vlm
from src.prompts import build_chat_messages
from src.utils import safe_latex_from_generation

st.set_page_config(page_title="Handwritten Formula to LaTeX", page_icon="MATH", layout="centered")
st.title("Handwritten Formula to LaTeX (Qwen2-VL-2B)")


with st.sidebar:
    st.header("Model Settings")
    base_model = st.text_input("Base model", value="Qwen/Qwen2-VL-2B-Instruct")
    adapter_path = st.text_input("LoRA adapter path (optional)", value="checkpoints/sft_linxy_mathwriting/adapter")
    max_new_tokens = st.slider("Max new tokens", min_value=32, max_value=256, value=128, step=8)
    if st.button("Reload model cache"):
        st.cache_resource.clear()
        st.success("Model cache cleared. Next inference will reload weights.")


def _resolve_device_map():
    if torch.cuda.is_available():
        return "auto"
    return None


def _runtime_label() -> str:
    if torch.cuda.is_available():
        return "CUDA"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "MPS"
    return "CPU"


@st.cache_resource(show_spinner=True)
def _load_runtime(base_model_name: str, adapter: str):
    processor = load_processor(base_model_name)
    device_map = _resolve_device_map()

    if adapter and adapter.strip():
        model = load_base_with_adapter(
            base_model_name=base_model_name,
            adapter_path=adapter,
            prefer_bf16=True,
            device_map=device_map,
        )
    else:
        model = load_vlm(
            base_model_name,
            prefer_bf16=True,
            device_map=device_map,
        )
    model.eval()
    return model, processor


uploaded_file = st.file_uploader(
    "Upload a handwritten formula image",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    if adapter_path.strip() and not Path(adapter_path).exists():
        st.warning(f"Adapter path does not exist: {adapter_path}. Base model will be used if loading fails.")

    if st.button("Recognize LaTeX"):
        try:
            with st.spinner("Loading model and generating LaTeX..."):
                t0 = time.perf_counter()
                model, processor = _load_runtime(base_model, adapter_path)
                messages = build_chat_messages(image, target_latex=None)
                generated = generate_from_messages(
                    model,
                    processor,
                    messages,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                )
                latex = safe_latex_from_generation(generated)
                elapsed = time.perf_counter() - t0
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
        else:
            st.caption(f"Runtime: {_runtime_label()} | Inference time: {elapsed:.2f}s")
            st.subheader("Predicted LaTeX")
            st.code(latex, language="latex")

            st.subheader("Rendered")
            try:
                st.latex(latex.replace("$", ""))
            except Exception as exc:
                st.warning(f"Could not render LaTeX directly: {exc}")
