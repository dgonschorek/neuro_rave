"""Train/test splits for the batch diagnostic: generation, validation and design checks.

Every split must contain every domain in both its training and its test part, so that chance
level and pooled summaries mean the same thing in every split. With groups (cells, recordings,
animals, ...), no group may appear on both sides. Requested designs that cannot meet these rules
raise ``ValueError``; there is no silent fallback to a different design.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Split = tuple[NDArray[np.int64], NDArray[np.int64]]
MAX_ATTEMPTS = 50


def _name(names: Sequence[Any], code: int) -> str:
    return repr(names[code])


def check_split(
    train: NDArray[np.int64],
    test: NDArray[np.int64],
    domain_codes: NDArray[np.int64],
    group_codes: NDArray[np.int64] | None,
    domain_names: Sequence[Any],
    label: str,
) -> None:
    """Raise ``ValueError`` unless every domain is on both sides and no group crosses the split."""
    k = len(domain_names)
    for part, idx in (("training", train), ("test", test)):
        present = np.bincount(domain_codes[idx], minlength=k) > 0
        if not present.all():
            missing = ", ".join(_name(domain_names, int(c)) for c in np.flatnonzero(~present))
            raise ValueError(
                f"{label}: domain(s) {missing} have no samples in the {part} part. Every "
                "domain must appear in both parts so that the probe can predict it and chance "
                "level is the same in every split."
            )
    if group_codes is not None:
        shared = np.intersect1d(group_codes[train], group_codes[test])
        if shared.size:
            raise ValueError(
                f"{label}: {shared.size} group(s) appear in both the training and the test part."
            )


def validate_explicit(
    splits: Sequence[tuple[Any, Any]],
    n: int,
    domain_codes: NDArray[np.int64],
    group_codes: NDArray[np.int64] | None,
    domain_names: Sequence[Any],
) -> list[Split]:
    """Check user-supplied ``(train_indices, test_indices)`` pairs and return them as int arrays."""
    if isinstance(splits, (str, bytes)) or len(splits) == 0:
        raise ValueError("splits must be a non-empty sequence of (train_indices, test_indices).")
    out: list[Split] = []
    for i, pair in enumerate(splits):
        label = f"split {i}"
        if len(pair) != 2:
            raise ValueError(f"{label}: expected (train_indices, test_indices).")
        parts = []
        for part, raw in zip(("train", "test"), pair, strict=True):
            arr = np.asarray(raw)
            if arr.ndim != 1 or arr.size == 0:
                raise ValueError(f"{label}: {part} indices must be a non-empty 1-D sequence.")
            if arr.dtype.kind not in "iu":
                raise ValueError(
                    f"{label}: {part} indices must be integers (positions), not {arr.dtype}."
                )
            if np.min(arr) < 0 or np.max(arr) >= n:
                raise ValueError(f"{label}: {part} indices must lie in [0, {n - 1}].")
            if np.unique(arr).size != arr.size:
                raise ValueError(f"{label}: {part} indices contain duplicates.")
            parts.append(arr.astype(np.int64))
        train, test = parts
        if np.intersect1d(train, test).size:
            raise ValueError(f"{label}: the same sample is in both the training and the test part.")
        check_split(train, test, domain_codes, group_codes, domain_names, label)
        out.append((train, test))
    return out


def unit_coverage(
    domain_codes: NDArray[np.int64], group_codes: NDArray[np.int64] | None, k: int
) -> NDArray[np.int64]:
    """Number of independent units per domain: distinct groups, or samples without groups."""
    if group_codes is None:
        return np.bincount(domain_codes, minlength=k).astype(np.int64)
    pairs = np.unique(np.stack([domain_codes, group_codes], axis=1), axis=0)
    return np.bincount(pairs[:, 0], minlength=k).astype(np.int64)


def generate(
    domain_codes: NDArray[np.int64],
    group_codes: NDArray[np.int64] | None,
    domain_names: Sequence[Any],
    *,
    n_splits: int,
    n_repeats: int,
    seed: int,
    model_selection: Any,
) -> tuple[list[Split], str, list[int], int]:
    """Stratified (group) K-fold splits, repeated, verified after generation.

    Returns the splits, the strategy name, the ``random_state`` used for each repeat and the number
    of distinct repeats (repeats can coincide when there are few groups).
    """
    k = len(domain_names)
    units = unit_coverage(domain_codes, group_codes, k)
    unit = "group" if group_codes is not None else "sample"
    if np.min(units) < n_splits:
        worst = int(np.argmin(units))
        lead = (
            f"domain {_name(domain_names, worst)} has {int(units[worst])} {unit}(s), but "
            f"n_splits={n_splits} needs at least {n_splits} per domain so that every domain is in "
            "every test part (and, with the remaining ones, in every training part)."
        )
        if group_codes is not None and np.min(units) == 1:
            lead += (
                " A domain whose samples all belong to one group cannot be both trained on and "
                "tested: holding that group out removes the domain from probe training. This is "
                "why groups=domains is not a valid design for predicting domain identity; use "
                "groups such as recordings or cells nested within domains."
            )
        raise ValueError(lead + " Reduce n_splits (minimum 2) or provide explicit splits.")
    idx = np.arange(domain_codes.size)
    splits: list[Split] = []
    states: list[int] = []
    signatures: set[tuple[tuple[int, ...], ...]] = set()
    for r in range(n_repeats):
        for attempt in range(MAX_ATTEMPTS):
            state = seed + r * MAX_ATTEMPTS + attempt
            if group_codes is None:
                cv = model_selection.StratifiedKFold(n_splits, shuffle=True, random_state=state)
                folds = list(cv.split(idx, domain_codes))
            else:
                cv = model_selection.StratifiedGroupKFold(
                    n_splits, shuffle=True, random_state=state
                )
                folds = list(cv.split(idx, domain_codes, group_codes))
            try:
                checked = []
                for f, (tr, te) in enumerate(folds):
                    tr, te = np.sort(tr).astype(np.int64), np.sort(te).astype(np.int64)
                    check_split(tr, te, domain_codes, group_codes, domain_names, f"fold {f}")
                    checked.append((tr, te))
            except ValueError:
                continue
            splits.extend(checked)
            states.append(state)
            signatures.add(tuple(sorted(tuple(te.tolist()) for _, te in checked)))
            break
        else:
            raise ValueError(
                f"No {n_splits}-fold {'grouped ' if group_codes is not None else ''}split with "
                f"every domain in every training and test part was found in {MAX_ATTEMPTS} "
                f"attempts (repeat {r}). Reduce n_splits or provide explicit splits; no row-wise "
                "fallback is used."
            )
    strategy = "stratified_group_kfold" if group_codes is not None else "stratified_kfold"
    return splits, strategy, states, len(signatures)
