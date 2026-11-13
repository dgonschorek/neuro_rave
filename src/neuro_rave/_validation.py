"""Input validation and categorical encoding at the public API boundary."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

_REAL_KINDS = "iuf"


def as_windows(X: ArrayLike) -> NDArray[np.float32]:
    """Return ``X`` as a float32 array shaped (samples, time) or (samples, time, channels)."""
    try:
        arr = np.asarray(X)
    except ValueError as err:
        raise ValueError(f"X must be a rectangular numeric array: {err}") from err
    if arr.dtype.kind not in _REAL_KINDS:
        raise TypeError(
            f"X must contain real numbers (integer or float dtype); got dtype {arr.dtype}."
        )
    if arr.ndim not in (2, 3):
        raise ValueError(
            "X must have shape (samples, time) or (samples, time, channels); "
            f"got shape {arr.shape} with {arr.ndim} dimension(s)."
        )
    if 0 in arr.shape:
        raise ValueError(f"X must not have an empty dimension; got shape {arr.shape}.")
    with np.errstate(over="ignore", invalid="ignore"):
        windows = arr.astype(np.float32)
    n_bad = int(np.size(windows) - np.count_nonzero(np.isfinite(windows)))
    if n_bad:
        raise ValueError(
            f"X contains {n_bad} non-finite value(s) after conversion to float32 "
            "(NaN, Inf, or magnitudes beyond the float32 range)."
        )
    return windows


def check_layout(windows: NDArray[np.float32], input_shape: tuple[int, ...]) -> None:
    """Raise unless ``windows`` has the per-row shape the model was fitted on."""
    if windows.shape[1:] != input_shape:
        expected = "(samples, " + ", ".join(str(d) for d in input_shape) + ")"
        raise ValueError(
            f"X has shape {windows.shape}, but the model was fitted on samples shaped "
            f"{expected}. Inference requires the same time length and channel layout."
        )


@dataclass(frozen=True)
class EncodedCategories:
    """Integer codes into ``categories``; ``-1`` marks a missing entry (``known`` is False)."""

    codes: NDArray[np.int64]
    known: NDArray[np.bool_]
    categories: NDArray[Any]


def _is_nan(value: object) -> bool:
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def encode_categories(
    values: ArrayLike,
    *,
    name: str,
    n_rows: int,
    allow_missing: bool,
    min_categories: int = 2,
) -> EncodedCategories:
    """Map arbitrary hashable categories to codes ``0..K-1`` in sorted category order."""
    as_objects = np.asarray(values, dtype=object)
    if as_objects.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional with one entry per row of X; "
            f"got shape {as_objects.shape}."
        )
    entries: list[Any] = as_objects.tolist()
    if len(entries) != n_rows:
        raise ValueError(f"{name} has {len(entries)} entries, but X has {n_rows} rows.")

    known_mask = np.array([v is not None for v in entries], dtype=bool)
    if not allow_missing and not known_mask.all():
        raise ValueError(f"{name} contains None at {int((~known_mask).sum())} row(s).")
    observed = [v for v in entries if v is not None]
    if any(_is_nan(v) for v in observed):
        hint = " Use None to mark a missing label." if allow_missing else ""
        raise ValueError(f"{name} contains NaN.{hint}")
    if not all(isinstance(v, Hashable) for v in observed):
        raise TypeError(f"{name} entries must be hashable categories (e.g. str or int).")
    try:
        ordered = sorted(set(observed))
    except TypeError as err:
        kinds = sorted({type(v).__name__ for v in observed})
        raise TypeError(
            f"{name} mixes category types that cannot be ordered ({', '.join(kinds)}); "
            "use a single type such as all str or all int."
        ) from err
    if len(ordered) < min_categories:
        found = ", ".join(repr(v) for v in ordered) or "none"
        raise ValueError(
            f"{name} must contain at least {min_categories} distinct observed values; "
            f"found {len(ordered)} ({found})."
        )

    lookup = {category: code for code, category in enumerate(ordered)}
    codes = np.array([lookup[v] if v is not None else -1 for v in entries], dtype=np.int64)
    return EncodedCategories(codes=codes, known=known_mask, categories=_category_array(ordered))


def _category_array(ordered: list[Any]) -> NDArray[Any]:
    arr = np.asarray(ordered)
    if arr.ndim == 1 and arr.dtype.kind != "O":
        return arr
    as_objects = np.empty(len(ordered), dtype=object)
    as_objects[:] = ordered
    return as_objects
