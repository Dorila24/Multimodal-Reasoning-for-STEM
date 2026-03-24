"""Dataset loading and normalization for Track-10."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datasets import Dataset, DatasetDict, Image, concatenate_datasets, load_dataset

IMAGE_COLUMN_CANDIDATES = [
    "image",
    "img",
    "formula_image",
    "pixel_values",
    "rendered_image",
]

TEXT_COLUMN_CANDIDATES = [
    "latex",
    "formula",
    "equation",
    "text",
    "label",
    "ground_truth",
    "gt",
    "target",
]


@dataclass
class TrackDatasets:
    linxy_train: Dataset
    linxy_test: Dataset
    mathwriting_train: Dataset


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
        return " ".join(str(item) for item in value)
    return str(value)


def _find_image_column(ds: Dataset) -> str:
    columns = ds.column_names

    for name in IMAGE_COLUMN_CANDIDATES:
        if name in columns:
            return name

    for col, feature in ds.features.items():
        if isinstance(feature, Image):
            return col

    for col in columns:
        low = col.lower()
        if "image" in low or "img" in low:
            return col

    raise ValueError(f"Could not identify image column in dataset columns={columns}")


def _find_text_column(ds: Dataset, image_col: str) -> str:
    columns = ds.column_names

    for name in TEXT_COLUMN_CANDIDATES:
        if name in columns and name != image_col:
            return name

    for col in columns:
        if col == image_col:
            continue
        feat = ds.features[col]
        if getattr(feat, "dtype", None) == "string":
            return col

    for col in columns:
        low = col.lower()
        if col != image_col and any(token in low for token in ["latex", "formula", "equ", "label", "text"]):
            return col

    raise ValueError(f"Could not identify text/latex column in dataset columns={columns}")


def _is_non_empty_latex(example: dict) -> bool:
    latex = str(example.get("latex", "")).strip()
    return bool(latex)


def standardize_formula_dataset(ds: Dataset, source_name: str) -> Dataset:
    image_col = _find_image_column(ds)
    text_col = _find_text_column(ds, image_col)

    def _map_fn(ex: dict) -> dict:
        return {
            "image": ex[image_col],
            "latex": _coerce_text(ex[text_col]).strip(),
            "source": source_name,
        }

    mapped = ds.map(_map_fn, remove_columns=ds.column_names)
    # Drop invalid/empty targets to avoid noisy supervision and broken evaluation rows.
    mapped = mapped.filter(_is_non_empty_latex)
    return mapped


def _load_dataset_with_fallback(path: str, configs: Iterable[str | None], cache_dir: str | None = None) -> DatasetDict:
    errors: list[str] = []
    for cfg in configs:
        try:
            if cfg is None:
                return load_dataset(path, cache_dir=cache_dir)
            return load_dataset(path, cfg, cache_dir=cache_dir)
        except Exception as exc:  # pragma: no cover
            errors.append(f"config={cfg}: {exc}")
    message = "\n".join(errors)
    raise RuntimeError(f"Failed to load dataset {path}. Tried configs:\n{message}")


def _take(ds: Dataset, max_samples: int | None, seed: int = 42) -> Dataset:
    if max_samples is None or max_samples <= 0 or len(ds) <= max_samples:
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


def load_track_datasets(
    *,
    linxy_config: str = "human_handwrite",
    linxy_train_max_samples: int | None = None,
    linxy_test_max_samples: int | None = 70,
    mathwriting_max_samples: int | None = None,
    cache_dir: str | None = None,
    seed: int = 42,
) -> TrackDatasets:
    """Load and standardize datasets required by the technical task."""

    linxy_ds = _load_dataset_with_fallback(
        "linxy/LaTeX_OCR",
        configs=[linxy_config, None],
        cache_dir=cache_dir,
    )

    if "train" not in linxy_ds:
        raise KeyError("linxy/LaTeX_OCR does not contain 'train' split")
    if "test" not in linxy_ds:
        raise KeyError("linxy/LaTeX_OCR does not contain 'test' split")

    linxy_train = standardize_formula_dataset(linxy_ds["train"], "linxy")
    linxy_test = standardize_formula_dataset(linxy_ds["test"], "linxy")

    linxy_train = _take(linxy_train, linxy_train_max_samples, seed=seed)
    linxy_test = _take(linxy_test, linxy_test_max_samples, seed=seed)

    math_ds = _load_dataset_with_fallback(
        "deepcopy/MathWriting-human",
        configs=[None],
        cache_dir=cache_dir,
    )

    if "train" in math_ds:
        math_train_raw = math_ds["train"]
    else:
        first_split = next(iter(math_ds.keys()))
        math_train_raw = math_ds[first_split]

    math_train = standardize_formula_dataset(math_train_raw, "mathwriting")
    math_train = _take(math_train, mathwriting_max_samples, seed=seed)

    return TrackDatasets(
        linxy_train=linxy_train,
        linxy_test=linxy_test,
        mathwriting_train=math_train,
    )


def build_sft_train_dataset(
    datasets_bundle: TrackDatasets,
    setup: str,
) -> Dataset:
    if setup == "linxy":
        return datasets_bundle.linxy_train
    if setup == "linxy_mathwriting":
        return concatenate_datasets([datasets_bundle.linxy_train, datasets_bundle.mathwriting_train])
    raise ValueError("setup must be one of: ['linxy', 'linxy_mathwriting']")
