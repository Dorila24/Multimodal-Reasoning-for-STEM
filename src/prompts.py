"""Prompt templates for handwritten formula -> LaTeX transcription."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a precise OCR assistant for handwritten mathematical formulas. "
    "Return only valid LaTeX for the formula in the image. "
    "Do not explain your answer."
)

USER_PROMPT = (
    "Transcribe the handwritten mathematical formula from the image into LaTeX. "
    "Output LaTeX only."
)


def build_chat_messages(image, target_latex: str | None = None):
    """Build a basic image+text chat for Qwen2-VL style models."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]

    if target_latex is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_latex}],
            }
        )

    return messages


def _strip_math_delimiters(latex: str) -> str:
    latex = (latex or "").strip()
    latex = latex.replace("$$", "").replace("$", "").strip()
    return latex


def build_one_shot_messages(example_image, example_latex: str, query_image):
    """Build one-shot as a true multi-turn demonstration for chat VLMs."""
    example_latex = _strip_math_delimiters(example_latex)

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Example task. Read the image and return only LaTeX, "
                        "without markdown fences and without explanations."
                    ),
                },
                {"type": "image", "image": example_image},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example_latex}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Now solve the same task for the next image. "
                        "Return only LaTeX."
                    ),
                },
                {"type": "image", "image": query_image},
            ],
        },
    ]
