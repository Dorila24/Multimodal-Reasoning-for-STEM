"""Utility helpers: seeding, LaTeX normalization and metrics."""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def normalize_latex(text: str) -> str:
    """A lightweight normalization to reduce formatting-only mismatches."""
    if text is None:
        return ""
    text = text.strip()
    text = text.replace("$$", "").replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def postprocess_latex(text: str) -> str:
    """Apply lightweight LaTeX fixes for common OCR generation artifacts."""
    text = (text or "").strip()

    # Fix common malformed tokenization for lim.
    text = text.replace(r"\operatorname* { l i m }", r"\lim")

    # Fix `\sqrt 4` / `\sqrt4` / `\sqrt x` -> `\sqrt { 4 }` / etc.
    pattern = re.compile(r"\\sqrt\s*(?!\{)(\\?[A-Za-z]+|\d+)")
    blocked = {"\\left", "\\right", "\\big", "\\Big", "\\Bigl", "\\Bigr"}

    def _repl(match: re.Match) -> str:
        token = match.group(1)
        if token in blocked:
            return match.group(0)
        return f"\\sqrt {{ {token} }}"

    prev = None
    while text != prev:
        prev = text
        text = pattern.sub(_repl, text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            sub_cost = prev[j - 1] + (ca != cb)
            curr.append(min(insert_cost, delete_cost, sub_cost))
        prev = curr
    return prev[-1]


def safe_latex_from_generation(text: str) -> str:
    """Keep first non-empty line and remove markdown fences if model emits them."""
    text = (text or "").strip()
    text = text.replace("```latex", "").replace("```", "").strip()
    if "\n" in text:
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        return postprocess_latex(first)
    return postprocess_latex(text)


def _has_balanced_braces(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _has_balanced_envs(text: str) -> bool:
    begins = re.findall(r"\\begin\s*\{([^}]+)\}", text)
    ends = re.findall(r"\\end\s*\{([^}]+)\}", text)
    return sorted(begins) == sorted(ends)


def _is_latex_like(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return 0.0
    if not _has_balanced_braces(text):
        return 0.0
    if not _has_balanced_envs(text):
        return 0.0
    return 1.0


def compute_metrics(predictions: list[str], references: list[str]) -> dict:
    assert len(predictions) == len(references), "Predictions and references must have same length"

    normalized_preds = [normalize_latex(safe_latex_from_generation(p)) for p in predictions]
    normalized_refs = [normalize_latex(r) for r in references]

    exact = []
    similarities = []
    cer_values = []
    non_empty = []
    latex_like = []

    for pred, ref in zip(normalized_preds, normalized_refs):
        exact.append(float(pred == ref))
        dist = levenshtein_distance(pred, ref)
        denom = max(len(ref), len(pred), 1)
        similarities.append(1.0 - (dist / denom))
        cer_values.append(dist / max(len(ref), 1))
        non_empty.append(float(bool(pred.strip())))
        latex_like.append(_is_latex_like(pred))

    return {
        "num_samples": len(references),
        "exact_match": float(np.mean(exact)) if exact else 0.0,
        "norm_edit_similarity": float(np.mean(similarities)) if similarities else 0.0,
        "char_error_rate": float(np.mean(cer_values)) if cer_values else 0.0,
        "non_empty_rate": float(np.mean(non_empty)) if non_empty else 0.0,
        "latex_like_rate": float(np.mean(latex_like)) if latex_like else 0.0,
    }


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
