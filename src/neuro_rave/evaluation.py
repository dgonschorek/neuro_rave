"""Evaluation functions for comparing models that reduce domain (e.g. session) effects.

All functions take plain arrays and return dictionaries of floats, NumPy arrays and lists. They work
with any method's outputs, not only this package's models.

Rows are samples. A sample is one time series (``(time,)``) or one multichannel time series
(``(time, channels)``); per-sample arrays are flattened to features where a function needs a vector.
Group, domain and class IDs keep their original values (strings or integers) in every output; they
are never reinterpreted as array positions.

``reconstruction_scores``, ``classification_scores``, ``response_correlation``,
``centroid_similarity``, ``best_matches`` and ``matching_agreement`` need only NumPy.
``domain_probe`` and ``assess_batch_effects`` fit scikit-learn classifiers and need the
``evaluation`` extra::

    python -m pip install "neuro-rave[evaluation]"
"""

from __future__ import annotations

import contextlib
import importlib
import math
from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neuro_rave import _splits

__all__ = [
    "assess_batch_effects",
    "best_matches",
    "centroid_similarity",
    "classification_scores",
    "domain_probe",
    "matching_agreement",
    "reconstruction_scores",
    "response_correlation",
]

_EPS = 1e-15


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _sklearn(module: str) -> Any:
    try:
        return importlib.import_module(f"sklearn.{module}")
    except ImportError as err:
        raise ImportError(
            "This function needs scikit-learn. Install the evaluation extra: "
            'python -m pip install "neuro-rave[evaluation]"'
        ) from err


def _samples(X: ArrayLike, name: str) -> NDArray[np.float64]:
    """``X`` as float64 (n_samples, n_features); requires finite values and at least one sample."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[0] == 0:
        raise ValueError(
            f"{name} must have shape (samples, ...) with at least one sample; got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr.reshape(arr.shape[0], -1)


def _labels(values: ArrayLike, name: str, n: int, *, allow_missing: bool) -> NDArray[Any]:
    """One label per sample as an object array; ``None`` marks a missing label."""
    arr = np.empty(n, dtype=object)
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or raw.shape[0] != n:
        raise ValueError(f"{name} must be one-dimensional with {n} entries; got shape {raw.shape}.")
    arr[:] = list(raw)
    for v in arr:
        if v is None:
            if not allow_missing:
                raise ValueError(f"{name} contains None; every sample needs a value here.")
        elif isinstance(v, (float, np.floating)) and math.isnan(v):
            raise ValueError(f"{name} contains NaN; use None to mark a missing label.")
        elif not isinstance(v, Hashable):
            raise TypeError(f"{name} entries must be hashable (e.g. str or int).")
    return arr


def _ordered(values: Sequence[Any] | NDArray[Any], name: str) -> list[Any]:
    distinct = {v for v in values if v is not None}
    try:
        return sorted(distinct)
    except TypeError as err:
        raise TypeError(f"{name} mixes types that cannot be ordered; use a single type.") from err


def _to_list(values: Sequence[Any] | NDArray[Any]) -> list[Any]:
    return [v.item() if isinstance(v, np.generic) else v for v in values]


# --------------------------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------------------------


def _recon_block(err2: NDArray[np.float64], base2: NDArray[np.float64] | None) -> dict[str, Any]:
    n, f = err2.shape
    sse = float(err2.sum())
    out: dict[str, Any] = {"n_samples": int(n), "mse": sse / (n * f)}
    if base2 is not None:
        sse_base = float(base2.sum())
        out["baseline_mse"] = sse_base / (n * f)
        out["reconstruction_skill"] = 1.0 - sse / sse_base if sse_base > 0 else float("nan")
    return out


def reconstruction_scores(
    X: ArrayLike,
    X_hat: ArrayLike,
    *,
    reference_mean: ArrayLike | None = None,
    groups: ArrayLike | None = None,
) -> dict[str, Any]:
    """Reconstruction error of ``X_hat`` against ``X``, overall and per group.

    Args:
        X: Recorded samples, shape (samples, time) or (samples, time, channels).
        X_hat: Reconstructions with the same shape.
        reference_mean: Optional reference prediction, usually the feature-wise mean of the
            *training* samples (``X_train.mean(axis=0)``), with the shape of one sample. It defines
            the baseline every sample is compared with.
        groups: Optional group ID per sample (e.g. domain or response type); no ``None``.

    Returns:
        ``mse``: mean squared error over all samples and features (each sample weighs equally,
        because all samples have the same number of features). ``per_sample_mse``: array (samples,).
        With ``reference_mean``: ``baseline_mse`` (the same for predicting the reference) and
        ``reconstruction_skill = 1 - SSE_model / SSE_baseline``. Skill is 1 for a perfect
        reconstruction, 0 for doing as well as the reference, negative for worse. Because the
        reference comes from training data, this is not the standard R^2 / explained variance of
        the test data. It is NaN when ``SSE_baseline`` is 0 (the samples equal the reference).
        ``per_group``: the same scores for each group, keyed by the original group IDs.
        ``n_samples``, ``n_features``.
    """
    x = _samples(X, "X")
    x_hat = _samples(X_hat, "X_hat")
    if np.shape(X) != np.shape(X_hat):
        raise ValueError(f"X has shape {np.shape(X)} but X_hat has shape {np.shape(X_hat)}.")
    err2 = (x - x_hat) ** 2
    base2: NDArray[np.float64] | None = None
    if reference_mean is not None:
        ref = np.asarray(reference_mean, dtype=np.float64)
        if ref.size != x.shape[1]:
            raise ValueError(
                "reference_mean must have the shape of one sample "
                f"({np.shape(X)[1:]}); got {ref.shape}."
            )
        if not np.all(np.isfinite(ref)):
            raise ValueError("reference_mean contains NaN or Inf.")
        base2 = (x - ref.reshape(1, -1)) ** 2
    out = _recon_block(err2, base2)
    out["n_features"] = int(x.shape[1])
    out["per_sample_mse"] = err2.mean(axis=1)
    if groups is not None:
        g = _labels(groups, "groups", x.shape[0], allow_missing=False)
        out["per_group"] = {
            _to_list([k])[0]: _recon_block(err2[g == k], None if base2 is None else base2[g == k])
            for k in _ordered(g, "groups")
        }
    return out


# --------------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------------


def _class_block(t: NDArray[Any], p: NDArray[Any], classes: list[Any]) -> dict[str, Any]:
    index = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for a, b in zip(t, p, strict=True):
        cm[index[a], index[b]] += 1
    support = cm.sum(axis=1)
    present = support > 0
    recall = np.full(len(classes), np.nan)
    recall[present] = np.diag(cm)[present] / support[present]
    n = int(len(t))
    return {
        "n_scored": n,
        "accuracy": float(np.trace(cm) / n) if n else float("nan"),
        "balanced_accuracy": float(np.mean(recall[present])) if present.any() else float("nan"),
        "chance_balanced_accuracy": 1.0 / int(present.sum()) if present.any() else float("nan"),
        "majority_accuracy": float(np.max(support) / n) if n else float("nan"),
        "recall_per_class": {c: float(r) for c, r in zip(classes, recall, strict=True)},
        "support_per_class": {c: int(s) for c, s in zip(classes, support, strict=True)},
        "confusion_matrix": cm,
    }


def classification_scores(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    classes: Sequence[Any] | None = None,
    groups: ArrayLike | None = None,
    y_proba: ArrayLike | None = None,
    proba_classes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Accuracy, balanced accuracy and confusion matrix of predicted against true labels.

    Args:
        y_true: True label per sample; ``None`` where the truth is unknown. Those samples are
            excluded from every score and counted in ``n_missing_true``. ``None`` is never a class.
        y_pred: Predicted label per sample (no ``None``).
        classes: Class order for the confusion matrix and per-class results. Default: sorted
            union of the true labels (of scored samples) and the predicted labels. Labels outside an
            explicit ``classes`` raise an error.
        groups: Optional group ID per sample (e.g. domain); scores are repeated per group.
        y_proba: Optional class probabilities, shape (samples, len(proba_classes)), used for
            ``log_loss``.
        proba_classes: Required with ``y_proba``: the class of each probability column
            (for this package's models, ``model.classes_``).

    Returns:
        ``accuracy``; ``balanced_accuracy`` (mean recall over classes present among the scored true
        labels); ``chance_balanced_accuracy`` (1 / number of those classes); ``majority_accuracy``
        (accuracy of always predicting the most frequent true class); ``confusion_matrix`` (rows
        true, columns predicted, both in ``classes`` order); ``recall_per_class``,
        ``support_per_class``; ``classes``; counts ``n_total``, ``n_scored``, ``n_missing_true``;
        optionally ``log_loss`` (mean negative log probability of the true class, probabilities
        clipped at 1e-15) and ``per_group``.
    """
    n = len(np.asarray(y_true, dtype=object))
    t_all = _labels(y_true, "y_true", n, allow_missing=True)
    p_all = _labels(y_pred, "y_pred", n, allow_missing=False)
    scored = np.array([v is not None for v in t_all], dtype=bool)
    t, p = t_all[scored], p_all[scored]
    if classes is None:
        order = _ordered(np.concatenate([t, p]), "labels")
    else:
        order = list(classes)
        if len(set(order)) != len(order):
            raise ValueError("classes contains duplicates.")
        unknown = ({*t} | {*p}) - set(order)
        if unknown:
            raise ValueError(f"labels not in classes: {sorted(map(repr, unknown))}.")
    order = _to_list(order)
    out = _class_block(t, p, order)
    out.update(classes=order, n_total=int(n), n_missing_true=int((~scored).sum()))
    if y_proba is not None:
        if proba_classes is None:
            raise ValueError("proba_classes is required with y_proba (the class of each column).")
        cols = _to_list(list(proba_classes))
        proba = np.asarray(y_proba, dtype=np.float64)
        if proba.shape != (n, len(cols)):
            raise ValueError(f"y_proba must have shape ({n}, {len(cols)}); got {proba.shape}.")
        col = {c: i for i, c in enumerate(cols)}
        missing = {*t} - set(col)
        if missing:
            raise ValueError(
                f"true labels without a probability column: {sorted(map(repr, missing))}."
            )
        p_true = proba[scored][np.arange(int(scored.sum())), [col[c] for c in t]]
        out["log_loss"] = (
            float(-np.log(np.clip(p_true, _EPS, 1.0)).mean()) if len(t) else float("nan")
        )
    if groups is not None:
        g = _labels(groups, "groups", n, allow_missing=False)
        out["per_group"] = {}
        for k in _ordered(g, "groups"):
            sel = g[scored] == k
            block = _class_block(t[sel], p[sel], order)
            block["n_missing_true"] = int(((g == k) & ~scored).sum())
            out["per_group"][_to_list([k])[0]] = block
    return out


# --------------------------------------------------------------------------------------------
# domain probe
# --------------------------------------------------------------------------------------------


def domain_probe(
    Z_train: ArrayLike,
    domains_train: ArrayLike,
    Z_test: ArrayLike,
    domains_test: ArrayLike,
    *,
    kind: Literal["knn", "linear"] = "knn",
    n_neighbors: int = 15,
    C: float = 1.0,  # noqa: N803  (scikit-learn's name)
    balance: bool = True,
    max_train_per_domain: int | None = None,
    standardize: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """How well a separately trained classifier predicts the domain from held-out embeddings.

    The probe is trained on ``Z_train`` and scored only on ``Z_test``; use samples (ideally whole
    recordings) that the probe never saw. Standardization is fitted on the probe's training samples
    only. Needs scikit-learn (``neuro-rave[evaluation]``).

    Args:
        Z_train, domains_train: Probe-training embeddings (samples, features) and their domain IDs.
        Z_test, domains_test: Held-out embeddings and domain IDs; every test domain must occur in
            training.
        kind: ``"knn"`` (k-nearest neighbours, Euclidean) or ``"linear"`` (multinomial logistic
            regression, L2 penalty ``C``).
        n_neighbors: k for ``"knn"``.
        balance: If True, randomly subsample the training samples to the same number per domain
            (the smallest domain's count, or ``max_train_per_domain`` if smaller), so the probe
            has a
            uniform domain prior. If False, all (or ``max_train_per_domain``) samples per domain.
        max_train_per_domain: Optional cap on training samples per domain (memory and run time).
        standardize: Scale features with training mean and SD before fitting.
        seed: Seed for subsampling (and the solver).

    Returns:
        ``balanced_accuracy`` (mean recall over test domains), ``accuracy``, ``log_loss`` (mean over
        test samples) and ``balanced_log_loss`` (mean over domains of the per-domain mean);
        baselines ``chance_balanced_accuracy`` (1/K), ``majority_accuracy`` (share of the most
        frequent test domain: the accuracy of always predicting it), ``uniform_log_loss`` (ln K) and
        ``prior_log_loss`` (log loss of predicting the probe's training domain proportions);
        ``confusion_matrix`` (rows true, columns predicted); ``classes`` (domain order);
        ``n_train_used``, ``train_counts``, ``test_counts``; ``settings``.
        k-NN probabilities are vote fractions with add-one smoothing,
        ``(votes + 1) / (k + K)``, so the log loss stays finite.
    """
    if kind not in ("knn", "linear"):
        raise ValueError(f"kind must be 'knn' or 'linear'; got {kind!r}.")
    a = _samples(Z_train, "Z_train")
    b = _samples(Z_test, "Z_test")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Z_train has {a.shape[1]} features but Z_test has {b.shape[1]}.")
    da = _labels(domains_train, "domains_train", a.shape[0], allow_missing=False)
    db = _labels(domains_test, "domains_test", b.shape[0], allow_missing=False)
    classes = _ordered(da, "domains_train")
    if len(classes) < 2:
        raise ValueError("domains_train must contain at least two domains.")
    unseen = set(db) - set(classes)
    if unseen:
        raise ValueError(f"test domains absent from training: {sorted(map(repr, unseen))}.")
    return _probe_core(
        a,
        da,
        b,
        db,
        classes,
        kind=kind,
        n_neighbors=n_neighbors,
        C=C,
        balance=balance,
        max_train_per_domain=max_train_per_domain,
        standardize=standardize,
        seed=seed,
    ).result


@dataclass(frozen=True)
class _ProbeRun:
    """A probe result plus the per-test-sample arrays that pooled summaries need."""

    result: dict[str, Any]
    test_codes: NDArray[np.int64]
    nll: NDArray[np.float64]
    prior_nll: NDArray[np.float64]


def _probe_core(
    a: NDArray[np.float64],
    da: NDArray[Any],
    b: NDArray[np.float64],
    db: NDArray[Any],
    classes: list[Any],
    *,
    kind: str,
    n_neighbors: int,
    C: float,  # noqa: N803  (scikit-learn's name)
    balance: bool,
    max_train_per_domain: int | None,
    standardize: bool,
    seed: int,
) -> _ProbeRun:
    """Fit the probe on (a, da) and score it on (b, db); inputs are already validated."""
    rng = np.random.default_rng(seed)
    counts = {c: int((da == c).sum()) for c in classes}
    per = min(counts.values()) if balance else None
    if max_train_per_domain is not None:
        per = max_train_per_domain if per is None else min(per, max_train_per_domain)
    idx = []
    for c in classes:
        rows = np.flatnonzero(da == c)
        take = len(rows) if per is None else min(per, len(rows))
        idx.append(np.sort(rng.choice(rows, take, replace=False)) if take < len(rows) else rows)
    sel = np.concatenate(idx)
    a, da_used = a[sel], da[sel]

    if standardize:
        mu, sd = a.mean(axis=0), a.std(axis=0)
        sd[sd == 0] = 1.0
        a, b = (a - mu) / sd, (b - mu) / sd
    code = {c: i for i, c in enumerate(classes)}
    ya = np.array([code[v] for v in da_used])
    yb = np.array([code[v] for v in db])
    k_cls = len(classes)
    if kind == "knn":
        neighbors = _sklearn("neighbors")
        if n_neighbors > len(ya):
            raise ValueError(
                f"n_neighbors={n_neighbors} exceeds the {len(ya)} probe-training samples."
            )
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors).fit(a, ya)
        votes = np.zeros((len(yb), k_cls))
        votes[:, clf.classes_] = clf.predict_proba(b) * n_neighbors
        proba = (votes + 1.0) / (n_neighbors + k_cls)
        settings: dict[str, Any] = {
            "n_neighbors": n_neighbors,
            "probability": "(votes + 1) / (k + K)",
        }
    else:
        linear = _sklearn("linear_model")
        clf = linear.LogisticRegression(C=C, max_iter=1000, random_state=seed).fit(a, ya)
        proba = np.zeros((len(yb), k_cls))
        proba[:, clf.classes_] = clf.predict_proba(b)
        settings = {"C": C, "penalty": "l2", "max_iter": 1000}
    settings.update(
        kind=kind,
        balance=balance,
        max_train_per_domain=max_train_per_domain,
        standardize=standardize,
        seed=seed,
    )

    pred = proba.argmax(axis=1)
    cm = np.zeros((k_cls, k_cls), dtype=np.int64)
    np.add.at(cm, (yb, pred), 1)
    present = cm.sum(axis=1) > 0
    recall = np.diag(cm)[present] / cm.sum(axis=1)[present]
    nll = -np.log(np.clip(proba[np.arange(len(yb)), yb], _EPS, 1.0))
    prior = np.bincount(ya, minlength=k_cls) / len(ya)
    prior_nll = -np.log(np.clip(prior[yb], _EPS, 1.0))
    test_counts = np.bincount(yb, minlength=k_cls)
    result = {
        "kind": kind,
        "classes": _to_list(classes),
        "balanced_accuracy": float(recall.mean()),
        "accuracy": float(np.trace(cm) / len(yb)),
        "log_loss": float(nll.mean()),
        "balanced_log_loss": float(np.mean([nll[yb == i].mean() for i in np.flatnonzero(present)])),
        "chance_balanced_accuracy": 1.0 / k_cls,
        "majority_accuracy": float(np.max(test_counts) / len(yb)),
        "uniform_log_loss": float(np.log(k_cls)),
        "prior_log_loss": float(prior_nll.mean()),
        "confusion_matrix": cm,
        "n_train_used": int(len(ya)),
        "train_counts": {
            c: int(n)
            for c, n in zip(_to_list(classes), np.bincount(ya, minlength=k_cls), strict=True)
        },
        "test_counts": {c: int(n) for c, n in zip(_to_list(classes), test_counts, strict=True)},
        "settings": settings,
    }
    return _ProbeRun(result, yb.astype(np.int64), nll, prior_nll)


# --------------------------------------------------------------------------------------------
# batch diagnostic
# --------------------------------------------------------------------------------------------

_PROBE_KINDS = ("linear", "knn")
_QUESTION = (
    "How well can domain identity be predicted from held-out samples, under the supplied sampling "
    "and split design?"
)


def _codes(values: NDArray[Any], order: list[Any]) -> NDArray[np.int64]:
    index = {v: i for i, v in enumerate(order)}
    return np.array([index[v] for v in values], dtype=np.int64)


def _n_train_used(train_codes: NDArray[np.int64], k: int, balance: bool, cap: int | None) -> int:
    counts = np.bincount(train_codes, minlength=k).astype(np.int64)
    per: int | None = int(np.min(counts)) if balance else None
    if cap is not None:
        per = cap if per is None else min(per, cap)
    return int(np.sum(counts) if per is None else np.sum(np.minimum(counts, per)))


def _probe_over_splits(
    a: NDArray[np.float64],
    d: NDArray[Any],
    splits: list[_splits.Split],
    classes: list[Any],
    kind: str,
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run one probe on every split (seed + split index) and pool the test predictions."""
    runs = [
        _probe_core(a[tr], d[tr], a[te], d[te], classes, kind=kind, seed=seed + i, **settings)
        for i, (tr, te) in enumerate(splits)
    ]
    k = len(classes)
    cm = np.sum([r.result["confusion_matrix"] for r in runs], axis=0)
    support = cm.sum(axis=1)
    codes = np.concatenate([r.test_codes for r in runs])
    nll = np.concatenate([r.nll for r in runs])
    prior_nll = np.concatenate([r.prior_nll for r in runs])
    chance = 1.0 / k
    ba = [float(r.result["balanced_accuracy"]) for r in runs]
    ll = [float(r.result["log_loss"]) for r in runs]
    return {
        "pooled": {
            "balanced_accuracy": float(np.mean(np.diag(cm) / support)),
            "accuracy": float(np.trace(cm) / support.sum()),
            "log_loss": float(nll.mean()),
            "balanced_log_loss": float(np.mean([nll[codes == i].mean() for i in range(k)])),
            "chance_balanced_accuracy": chance,
            "majority_accuracy": float(np.max(support) / np.sum(support)),
            "uniform_log_loss": float(np.log(k)),
            "prior_log_loss": float(prior_nll.mean()),
            "confusion_matrix": cm,
            "classes": _to_list(classes),
            "n_scored": int(support.sum()),
        },
        "split_summary": {
            "balanced_accuracy": {
                "values": ba,
                "mean": float(np.mean(ba)),
                "min": float(np.min(ba)),
                "max": float(np.max(ba)),
            },
            "log_loss": {"values": ll, "mean": float(np.mean(ll))},
            "n_splits_above_chance": int(sum(v > chance for v in ba)),
            "n_splits": len(runs),
        },
        "per_split": [r.result for r in runs],
    }


def _probe_message(kind: str, res: dict[str, Any]) -> list[str]:
    pooled, summ = res["pooled"], res["split_summary"]
    above, total = summ["n_splits_above_chance"], summ["n_splits"]
    lines = [
        f"{kind} probe: pooled balanced accuracy {pooled['balanced_accuracy']:.3f} "
        f"(chance {pooled['chance_balanced_accuracy']:.3f}; splits "
        f"{summ['balanced_accuracy']['min']:.3f}-{summ['balanced_accuracy']['max']:.3f}, "
        f"above chance in {above} of {total})."
    ]
    if above == total:
        lines.append(
            f"Batch-associated differences are detectable with the {kind} probe under this split "
            "design."
        )
    else:
        lines.append(
            f"The {kind} probe did not exceed chance in every split; this does not show that the "
            "domains are equivalent."
        )
    return lines


def _label_analysis(
    a: NDArray[np.float64],
    d: NDArray[Any],
    dcode: NDArray[np.int64],
    domain_ids: list[Any],
    lab: NDArray[Any],
    g: NDArray[Any] | None,
    explicit: list[_splits.Split] | None,
    *,
    kinds: list[str],
    settings: dict[str, Any],
    n_splits: int,
    n_repeats: int,
    max_classes: int,
    seed: int,
) -> tuple[dict[str, Any], list[str]]:
    """Class composition per domain and domain predictability within shared classes."""
    k = len(domain_ids)
    known = np.array([v is not None for v in lab], dtype=bool)
    class_ids = _ordered(lab[known], "y")
    counts = np.zeros((k, len(class_ids)), dtype=np.int64)
    for i, c in enumerate(class_ids):
        is_c = np.array([v == c for v in lab], dtype=bool) & known
        counts[:, i] = np.bincount(dcode[is_c], minlength=k)
    dom_list, cls_list = _to_list(domain_ids), _to_list(class_ids)
    present = counts > 0
    n_domains_with = present.sum(axis=0)
    labelled = counts.sum(axis=1)
    shares = counts / np.maximum(labelled[:, None], 1)
    has_labels = labelled > 0
    max_share_diff = (
        float(np.max(shares[has_labels].max(axis=0) - shares[has_labels].min(axis=0)))
        if has_labels.sum() >= 2 and class_ids
        else float("nan")
    )
    confounded = [c for c, m in zip(cls_list, n_domains_with, strict=True) if m == 1]
    out: dict[str, Any] = {
        "classes": cls_list,
        "counts": {
            dom: {c: int(counts[i, j]) for j, c in enumerate(cls_list)}
            for i, dom in enumerate(dom_list)
        },
        "unknown_per_domain": {
            dom: int((~known & (dcode == i)).sum()) for i, dom in enumerate(dom_list)
        },
        "n_unknown": int((~known).sum()),
        "domains_per_class": {
            c: [dom_list[i] for i in np.flatnonzero(present[:, j])] for j, c in enumerate(cls_list)
        },
        "confounded_classes": confounded,
        "max_class_share_difference": max_share_diff,
        "within_class": {},
        "max_classes": max_classes,
    }
    msgs: list[str] = []
    if not class_ids:
        msgs.append("y contains no known labels; label analyses were skipped.")
        return out, msgs
    if np.isfinite(max_share_diff):
        msgs.append(
            "Labelled class proportions differ between domains by up to "
            f"{max_share_diff:.2f} (absolute difference in share). Composition differences alone "
            "can make domains predictable."
        )
    for c in confounded:
        dom = out["domains_per_class"][c][0]
        msgs.append(
            f"Class {c!r} is labelled in only one domain ({dom!r}); class and domain cannot be "
            "disentangled for this class."
        )

    order = sorted(
        (j for j in range(len(class_ids)) if n_domains_with[j] >= 2),
        key=lambda j: -int(counts[:, j].sum()),
    )
    for rank, j in enumerate(order):
        c = cls_list[j]
        entry: dict[str, Any] = {"n_samples": int(counts[:, j].sum())}
        out["within_class"][c] = entry
        if rank >= max_classes:
            entry.update(status="skipped", reason=f"outside the max_classes={max_classes} budget")
            continue
        rows = np.flatnonzero(known & np.array([v == class_ids[j] for v in lab], dtype=bool))
        sub_d = dcode[rows]
        sub_g = None if g is None else _codes(g[rows], _ordered(g[rows], "groups"))
        units = _splits.unit_coverage(sub_d, sub_g, k)
        entry["units_per_domain"] = {
            dom: int(units[i]) for i, dom in enumerate(dom_list) if counts[i, j] > 0
        }
        excluded: dict[Any, str] = {}
        if explicit is None:
            keep = [i for i in range(k) if units[i] >= n_splits]
            for i in range(k):
                if counts[i, j] > 0 and units[i] < n_splits:
                    excluded[dom_list[i]] = (
                        f"{int(units[i])} {'group' if g is not None else 'sample'}(s) < "
                        f"n_splits={n_splits}"
                    )
        else:
            pos = np.full(len(lab), -1)
            pos[rows] = np.arange(len(rows))
            keep = []
            for i in range(k):
                if counts[i, j] == 0:
                    continue
                ok = all(
                    (sub_d[pos[tr][pos[tr] >= 0]] == i).any()
                    and (sub_d[pos[te][pos[te] >= 0]] == i).any()
                    for tr, te in explicit
                )
                if ok:
                    keep.append(i)
                else:
                    excluded[dom_list[i]] = "not in both parts of every supplied split"
        entry["excluded_domains"] = excluded
        if len(keep) < 2:
            entry.update(status="skipped", reason="fewer than two domains with enough data")
            continue
        sel = np.isin(sub_d, keep)
        rows, sub_d = rows[sel], sub_d[sel]
        names = [domain_ids[i] for i in keep]
        sub_codes = _codes(np.array([domain_ids[i] for i in sub_d], dtype=object), names)
        sub_g = None if g is None else _codes(g[rows], _ordered(g[rows], "groups"))
        try:
            if explicit is None:
                class_splits, _, _, _ = _splits.generate(
                    sub_codes,
                    sub_g,
                    names,
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    seed=seed,
                    model_selection=_sklearn("model_selection"),
                )
            else:
                pos = np.full(len(lab), -1)
                pos[rows] = np.arange(len(rows))
                class_splits = []
                for s, (tr, te) in enumerate(explicit):
                    tr_c, te_c = pos[tr][pos[tr] >= 0], pos[te][pos[te] >= 0]
                    _splits.check_split(tr_c, te_c, sub_codes, sub_g, names, f"split {s}")
                    class_splits.append((tr_c.astype(np.int64), te_c.astype(np.int64)))
        except ValueError as err:
            entry.update(status="skipped", reason=str(err))
            continue
        entry.update(status="assessed", domains=_to_list(names), probes={}, skipped_probes={})
        sub_a, sub_lab_d = a[rows], d[rows]
        for kind in kinds:
            if kind == "knn":
                smallest = min(
                    _n_train_used(
                        sub_codes[tr],
                        len(names),
                        settings["balance"],
                        settings["max_train_per_domain"],
                    )
                    for tr, _ in class_splits
                )
                if smallest < settings["n_neighbors"]:
                    entry["skipped_probes"][kind] = (
                        f"n_neighbors={settings['n_neighbors']} exceeds the {smallest} "
                        "probe-training samples in at least one split"
                    )
                    continue
            res = _probe_over_splits(sub_a, sub_lab_d, class_splits, names, kind, settings, seed)
            entry["probes"][kind] = {"pooled": res["pooled"], "split_summary": res["split_summary"]}
            pooled = res["pooled"]
            msgs.append(
                f"Within class {c!r}, {kind} probe: pooled balanced accuracy "
                f"{pooled['balanced_accuracy']:.3f} (chance "
                f"{pooled['chance_balanced_accuracy']:.3f}, {len(names)} domains)."
            )
    return out, msgs


def assess_batch_effects(
    X: ArrayLike,
    domains: ArrayLike,
    *,
    groups: ArrayLike | None = None,
    y: ArrayLike | None = None,
    probes: Sequence[Literal["linear", "knn"]] = ("linear", "knn"),
    n_splits: int | None = None,
    n_repeats: int | None = None,
    splits: Sequence[tuple[ArrayLike, ArrayLike]] | None = None,
    n_neighbors: int = 15,
    C: float = 1.0,  # noqa: N803  (scikit-learn's name)
    balance: bool = True,
    max_train_per_domain: int | None = None,
    standardize: bool = True,
    max_classes: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """How well domain identity can be predicted from held-out samples, before any correction.

    Cross-validates :func:`domain_probe` (k-NN and/or logistic regression) on the samples
    themselves. It describes batch-associated differences under the supplied sampling and split
    design. It does not decide whether they are technical or biological, whether correction is
    needed, or whether a lower score means a better representation. No model is trained other
    than the probes, and no significance test is performed. Needs scikit-learn
    (``neuro-rave[evaluation]``).

    Splits: every split has every domain in both its training and its test part. With ``groups``,
    whole groups are held out (stratified group K-fold) and no group appears on both sides; each
    domain needs at least ``n_splits`` groups, so ``groups=domains`` is rejected: a domain that is
    one indivisible group cannot be both trained on and tested. Without ``groups``, samples are
    split individually, which assumes they are independent. Explicit ``splits`` are validated the
    same way.
    Designs that cannot meet these rules raise ``ValueError``; there is no fallback to another
    design. Scaling (``standardize``) and balancing are fitted on each training part only.

    Args:
        X: Time-series data (samples, time) or (samples, time, channels), or a feature or embedding
            matrix (samples, features). Finite values; flattened per sample; not modified.
        domains: One domain ID (string or integer) per sample; at least two domains.
        groups: Optional ID per sample of the unit that must stay together (cell, recording,
            animal, ...), nested within or across domains as in your design.
        y: Optional biological class label per sample, ``None`` where unknown. Enables the class
            composition table and domain predictability within classes shared by domains.
        probes: Probe kinds to run on identical splits: ``"linear"`` and/or ``"knn"``.
        n_splits: Folds of the generated split (default 5, minimum 2). Not allowed with ``splits``.
        n_repeats: Repetitions with different shuffles (default 1). Not allowed with ``splits``.
        splits: Explicit ``(train_indices, test_indices)`` pairs of integer positions instead of
            generated folds.
        n_neighbors, C, balance, max_train_per_domain, standardize: Probe settings, as in
            :func:`domain_probe`. With ``balance=True`` the probe trains on the same number of
            samples per domain.
        max_classes: Budget for the within-class analysis: at most this many shared classes (the
            most labelled first) are assessed; the others are reported as skipped. 0 disables it.
        seed: Seed for the split shuffles and the probes (split ``i`` uses ``seed + i``). Global
            random state is not touched.

    Returns:
        A dictionary with
        ``question``; ``n_samples``, ``n_features``, ``sample_shape``; ``domains`` (original IDs,
        sorted) and ``domain_counts``; ``groups`` (``None`` or ``n_groups``,
        ``groups_per_domain``, ``n_groups_spanning_domains``); ``split`` (``strategy``
        ``"stratified_group_kfold"``, ``"stratified_kfold"`` or ``"explicit"``, ``n_folds``,
        ``n_repeats``, ``n_distinct_repeats``, ``random_states``, ``indices`` as a list of
        ``{"train", "test"}`` arrays, ``n_scored_samples``, ``times_scored``); ``settings``;
        ``probes``, per kind a dict with
        ``pooled`` (scores from all test predictions pooled across splits: ``balanced_accuracy``,
        ``accuracy``, ``log_loss``, ``balanced_log_loss``, baselines ``chance_balanced_accuracy``,
        ``majority_accuracy``, ``uniform_log_loss``, ``prior_log_loss``, ``confusion_matrix``,
        ``classes``, ``n_scored``), ``split_summary`` (per-split balanced accuracy and log loss,
        ``n_splits_above_chance``) and ``per_split`` (the :func:`domain_probe` result of each
        split);
        ``labels`` (``None`` without ``y``; otherwise ``classes``, domain-by-class ``counts``,
        ``unknown_per_domain``, ``domains_per_class``, ``confounded_classes``,
        ``max_class_share_difference`` and ``within_class``, per class ``status`` "assessed" or
        "skipped" with ``reason``, coverage, and pooled probe results); ``messages`` (descriptive
        summary); ``limitations``; ``significance_test`` (``"none"``).
        Each sample is scored once per repeat; split-to-split variation reflects overlapping
        training sets and is not an independent confidence interval.
    """
    kinds: list[str] = list(probes)
    if not kinds or len(set(kinds)) != len(kinds) or any(k not in _PROBE_KINDS for k in kinds):
        raise ValueError(f"probes must be distinct entries of {_PROBE_KINDS}; got {probes!r}.")
    for name, value, low in (("n_neighbors", n_neighbors, 1), ("max_classes", max_classes, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
            raise ValueError(f"{name} must be an integer >= {low}; got {value!r}.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}.")
    sample_shape = [int(s) for s in np.shape(X)[1:]]
    a = _samples(X, "X")
    n = a.shape[0]
    d = _labels(domains, "domains", n, allow_missing=False)
    domain_ids = _ordered(d, "domains")
    if len(domain_ids) < 2:
        raise ValueError("domains must contain at least two domains.")
    dcode = _codes(d, domain_ids)
    k = len(domain_ids)
    g = None if groups is None else _labels(groups, "groups", n, allow_missing=False)
    gcode = None if g is None else _codes(g, _ordered(g, "groups"))
    lab = None if y is None else _labels(y, "y", n, allow_missing=True)

    if splits is not None:
        if n_splits is not None or n_repeats is not None:
            raise ValueError("n_splits and n_repeats apply to generated splits; omit them.")
        split_list = _splits.validate_explicit(splits, n, dcode, gcode, domain_ids)
        states: list[int] = []
        strategy, n_folds, reps, distinct = "explicit", len(split_list), 1, 1
    else:
        n_folds = 5 if n_splits is None else n_splits
        reps = 1 if n_repeats is None else n_repeats
        for name, value, low in (("n_splits", n_folds, 2), ("n_repeats", reps, 1)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < low:
                raise ValueError(f"{name} must be an integer >= {low}; got {value!r}.")
        split_list, strategy, states, distinct = _splits.generate(
            dcode,
            gcode,
            domain_ids,
            n_splits=int(n_folds),
            n_repeats=int(reps),
            seed=int(seed),
            model_selection=_sklearn("model_selection"),
        )

    settings: dict[str, Any] = {
        "n_neighbors": int(n_neighbors),
        "C": float(C),
        "balance": bool(balance),
        "max_train_per_domain": max_train_per_domain,
        "standardize": bool(standardize),
    }
    probe_results = {
        kind: _probe_over_splits(a, d, split_list, domain_ids, kind, settings, int(seed))
        for kind in kinds
    }
    times = np.bincount(np.concatenate([te for _, te in split_list]), minlength=n)
    scored = times[times > 0]
    dom_list = _to_list(domain_ids)

    messages: list[str] = []
    limitations = [
        "Domain predictability does not identify its cause: biological composition, animal, "
        "tissue, experimental condition, compartment and acquisition can all contribute.",
        "A score near chance does not prove that the domains are equivalent; another probe, other "
        "features or more data may separate them.",
        "A lower score after correction is not by itself evidence of a better representation; "
        "assess reconstruction and biological information on the same splits.",
        "Split-to-split variation comes from overlapping training sets; it is not an independent "
        "confidence interval.",
        "If X is an embedding from a model trained on these samples, the probe's split does not "
        "make this an independent test: train the model on training groups only and assess "
        "held-out ones.",
        "No significance test was performed.",
    ]
    group_info: dict[str, Any] | None = None
    if gcode is not None:
        units = _splits.unit_coverage(dcode, gcode, k)
        spanning = int(sum(len(np.unique(dcode[gcode == u])) > 1 for u in np.unique(gcode)))
        group_info = {
            "n_groups": int(np.unique(gcode).size),
            "groups_per_domain": {dom: int(units[i]) for i, dom in enumerate(dom_list)},
            "n_groups_spanning_domains": spanning,
        }
    if strategy == "stratified_group_kfold":
        assert group_info is not None
        per = list(group_info["groups_per_domain"].values())
        messages.append(
            f"Grouped splits: {group_info['n_groups']} groups ({min(per)}-{max(per)} per domain), "
            f"{n_folds} folds x {reps} repeat(s); no group is in both parts of a split."
        )
        limitations.append(
            "Held-out groups come from domains that also appear in training: this measures how "
            "well known domains are recognised in new groups, not identification of unseen "
            "domains. If each domain is a single session, it describes separation between known "
            "sessions."
        )
    elif strategy == "stratified_kfold":
        messages.append(
            f"Sample-level splits (no groups): {n_folds} folds x {reps} repeat(s), stratified by "
            "domain."
        )
        limitations.append(
            "Samples were split individually, which assumes they are independent. Related samples "
            "(the same cell, recording or animal) on both sides can inflate the scores; pass "
            "groups to hold them out together."
        )
    else:
        messages.append(
            f"{len(split_list)} explicit split(s), checked for bounds, duplicates, overlap and "
            f"domain coverage{' and group separation' if gcode is not None else ''}."
        )
    if group_info is not None:
        messages.append(
            f"{n} samples in {group_info['n_groups']} groups: the number of samples is not the "
            "number of independent units."
        )
    if distinct < reps:
        limitations.append(
            f"Only {distinct} of {reps} repeats produced distinct splits (few groups per domain); "
            "identical repeats add no information."
        )
    if reps > 1 or int(np.max(scored)) > 1:
        limitations.append(
            "Some samples are scored more than once (repeats or overlapping test parts); pooled "
            "counts include every appearance."
        )
    for kind in kinds:
        messages.extend(_probe_message(kind, probe_results[kind]))

    label_info: dict[str, Any] | None = None
    if lab is not None:
        label_info, label_msgs = _label_analysis(
            a,
            d,
            dcode,
            domain_ids,
            lab,
            g,
            split_list if strategy == "explicit" else None,
            kinds=kinds,
            settings=settings,
            n_splits=int(n_folds),
            n_repeats=int(reps),
            max_classes=int(max_classes),
            seed=int(seed),
        )
        messages.extend(label_msgs)
        limitations.append(
            "Domain predictability within a class narrows, but does not identify, the cause: "
            "sub-types, animals or conditions within a class can still differ between domains."
        )

    return {
        "question": _QUESTION,
        "n_samples": int(n),
        "n_features": int(a.shape[1]),
        "sample_shape": sample_shape,
        "domains": dom_list,
        "domain_counts": {
            dom: int(c) for dom, c in zip(dom_list, np.bincount(dcode, minlength=k), strict=True)
        },
        "groups": group_info,
        "split": {
            "strategy": strategy,
            "n_folds": int(n_folds),
            "n_repeats": int(reps),
            "n_distinct_repeats": int(distinct),
            "random_states": states,
            "indices": [{"train": tr, "test": te} for tr, te in split_list],
            "n_scored_samples": int(scored.size),
            "times_scored": {"min": int(np.min(scored)), "max": int(np.max(scored))},
        },
        "settings": {
            "probes": kinds,
            **settings,
            "max_classes": int(max_classes),
            "seed": int(seed),
        },
        "probes": probe_results,
        "labels": label_info,
        "messages": messages,
        "limitations": limitations,
        "significance_test": "none",
    }


# --------------------------------------------------------------------------------------------
# response correlation
# --------------------------------------------------------------------------------------------


def response_correlation(
    X: ArrayLike,
    X_hat: ArrayLike,
    *,
    channels: Literal["flatten", "separate"] = "flatten",
) -> dict[str, Any]:
    """Pearson correlation between each recorded sample and its reconstruction, across time.

    Args:
        X, X_hat: Same shape, (samples, time) or (samples, time, channels); finite values only.
        channels: For 3-D input, ``"flatten"`` correlates over all time points and channels
            together; ``"separate"`` gives one correlation per channel.

    Returns:
        ``per_sample``: array (samples,) or (samples, channels). A sample (or channel) that is
        constant in ``X`` or in ``X_hat`` has no defined correlation and gets NaN.
        ``mean`` and ``median`` over the defined values, ``n_defined``, ``n_constant``.
    """
    x = np.asarray(X, dtype=np.float64)
    x_hat = np.asarray(X_hat, dtype=np.float64)
    if x.shape != x_hat.shape or x.ndim not in (2, 3) or x.shape[0] == 0:
        raise ValueError(
            "X and X_hat must share a shape (samples, time[, channels]); "
            f"got {x.shape}, {x_hat.shape}."
        )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(x_hat))):
        raise ValueError("X or X_hat contains NaN or Inf.")
    if x.ndim == 3 and channels == "flatten":
        x, x_hat = x.reshape(x.shape[0], -1), x_hat.reshape(x.shape[0], -1)
    elif channels not in ("flatten", "separate"):
        raise ValueError(f"channels must be 'flatten' or 'separate'; got {channels!r}.")
    xo = x - x.mean(axis=1, keepdims=True)
    xr = x_hat - x_hat.mean(axis=1, keepdims=True)
    den = np.sqrt((xo**2).sum(axis=1) * (xr**2).sum(axis=1))
    num = (xo * xr).sum(axis=1)
    defined = den > 0
    r = np.full(num.shape, np.nan)
    r[defined] = num[defined] / den[defined]
    return {
        "per_sample": r,
        "mean": float(np.nanmean(r)) if defined.any() else float("nan"),
        "median": float(np.nanmedian(r)) if defined.any() else float("nan"),
        "n_defined": int(defined.sum()),
        "n_constant": int((~defined).sum()),
    }


# --------------------------------------------------------------------------------------------
# centroid matching
# --------------------------------------------------------------------------------------------


def _centroids(
    Z: NDArray[np.float64], g: NDArray[Any], min_samples: int
) -> tuple[list[Any], NDArray[np.float64], list[int]]:
    ids, rows, counts = [], [], []
    for k in _ordered(g, "groups"):
        sel = g == k
        if int(sel.sum()) >= min_samples:
            ids.append(k)
            rows.append(Z[sel].mean(axis=0))
            counts.append(int(sel.sum()))
    if not ids:
        raise ValueError(f"no group has at least min_samples={min_samples} samples.")
    return _to_list(ids), np.vstack(rows), counts


def centroid_similarity(
    Z_a: ArrayLike,
    groups_a: ArrayLike,
    Z_b: ArrayLike,
    groups_b: ArrayLike,
    *,
    metric: Literal["cosine", "pearson"] = "cosine",
    min_samples: int = 1,
) -> dict[str, Any]:
    """Similarity between the group centroids of two sets of embeddings.

    The two sides may use unrelated label spaces (e.g. reference cell groups and response clusters
    from another preparation); IDs are never equated across sides.

    Args:
        Z_a, groups_a: Embeddings (samples, features) and group ID per sample for side A;
            ``None`` marks a sample without a group (excluded).
        Z_b, groups_b: The same for side B.
        metric: ``"cosine"`` (cosine of the centroid vectors) or ``"pearson"`` (correlation across
            the features of the two centroids; with few features this is noisy).
        min_samples: Groups with fewer samples are left out and listed in ``dropped_a/b``.

    Returns:
        ``similarity``: array (groups in A, groups in B), rows follow ``groups_a`` and columns
        ``groups_b`` (sorted original IDs). A centroid with zero norm (cosine) or zero variance
        (pearson) gives a row/column of NaN. ``counts_a``, ``counts_b``, ``dropped_a``,
        ``dropped_b``, ``metric``.
    """
    za, zb = _samples(Z_a, "Z_a"), _samples(Z_b, "Z_b")
    if za.shape[1] != zb.shape[1]:
        raise ValueError(f"Z_a has {za.shape[1]} features but Z_b has {zb.shape[1]}.")
    ga = _labels(groups_a, "groups_a", za.shape[0], allow_missing=True)
    gb = _labels(groups_b, "groups_b", zb.shape[0], allow_missing=True)
    ids_a, ca, na = _centroids(za, ga, min_samples)
    ids_b, cb, nb = _centroids(zb, gb, min_samples)
    if metric == "pearson":
        ca = ca - ca.mean(axis=1, keepdims=True)
        cb = cb - cb.mean(axis=1, keepdims=True)
    elif metric != "cosine":
        raise ValueError(f"metric must be 'cosine' or 'pearson'; got {metric!r}.")
    norm_a, norm_b = np.linalg.norm(ca, axis=1), np.linalg.norm(cb, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = (ca @ cb.T) / np.outer(norm_a, norm_b)
    sim[norm_a == 0, :] = np.nan
    sim[:, norm_b == 0] = np.nan
    return {
        "similarity": sim,
        "groups_a": ids_a,
        "groups_b": ids_b,
        "counts_a": dict(zip(ids_a, na, strict=True)),
        "counts_b": dict(zip(ids_b, nb, strict=True)),
        "dropped_a": [k for k in _to_list(_ordered(ga, "groups_a")) if k not in ids_a],
        "dropped_b": [k for k in _to_list(_ordered(gb, "groups_b")) if k not in ids_b],
        "metric": metric,
    }


def best_matches(
    similarity: ArrayLike,
    row_ids: Sequence[Any],
    col_ids: Sequence[Any],
    *,
    top_k: int = 3,
    tie_tolerance: float = 0.0,
) -> dict[str, Any]:
    """The most similar columns for each row of a similarity matrix, with margins.

    To match in the other direction, pass ``similarity.T`` with the ID lists swapped.

    Args:
        similarity: Array (rows, columns); larger means more similar; NaN means "no score".
        row_ids, col_ids: Original IDs of the rows and columns (e.g. ``groups_a`` / ``groups_b``
            from :func:`centroid_similarity`).
        top_k: Number of ranked matches per row (capped at the number of columns).
        tie_tolerance: A row is ``ambiguous`` when its best and second-best scores differ by at
            most this amount.

    Returns:
        ``matches``: list (one entry per row, in ``row_ids`` order) of dicts with ``row`` (ID),
        ``top`` (column IDs, best first), ``scores``, ``margin`` (best minus second-best score; NaN
        with fewer than two scored columns), ``ambiguous`` and ``valid`` (False when the row has no
        finite score; ``top`` is then empty). Equal scores are ordered by column order.
        ``best``: dict row ID -> best column ID (None if invalid). ``n_valid``, ``n_ambiguous``.
    """
    sim = np.asarray(similarity, dtype=np.float64)
    rows, cols = _to_list(list(row_ids)), _to_list(list(col_ids))
    if sim.shape != (len(rows), len(cols)):
        raise ValueError(
            f"similarity has shape {sim.shape}, but there are {len(rows)} row and "
            f"{len(cols)} column IDs."
        )
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    matches = []
    for i, rid in enumerate(rows):
        s = sim[i]
        ok = np.flatnonzero(np.isfinite(s))
        order = ok[np.argsort(-s[ok], kind="stable")][:top_k]
        scores = [float(s[j]) for j in order]
        finite_sorted = np.sort(s[ok])[::-1]
        margin = float(finite_sorted[0] - finite_sorted[1]) if len(ok) >= 2 else float("nan")
        matches.append(
            {
                "row": rid,
                "top": [cols[j] for j in order],
                "scores": scores,
                "margin": margin,
                "ambiguous": bool(len(ok) >= 2 and margin <= tie_tolerance),
                "valid": bool(len(ok) > 0),
            }
        )
    return {
        "matches": matches,
        "best": {m["row"]: (m["top"][0] if m["valid"] else None) for m in matches},
        "n_valid": sum(m["valid"] for m in matches),
        "n_ambiguous": sum(m["ambiguous"] for m in matches),
    }


def matching_agreement(
    results: Sequence[dict[str, Any]],
    *,
    tie_tolerance: float = 0.0,
) -> dict[str, Any]:
    """How consistently the best match of each row group is found across repeated fits.

    Args:
        results: Outputs of :func:`centroid_similarity`, one per fit (e.g. per seed). Rows and
            columns are aligned by their original IDs, so fits may contain different groups.
        tie_tolerance: Passed to :func:`best_matches` to flag near-ties.

    Returns:
        ``row_ids``: union of row IDs across fits (sorted where possible). Per row (lists in that
        order): ``coverage`` (fraction of fits in which the row has a valid best match),
        ``consensus`` (the most frequent best match), ``agreement`` (fraction of the fits covering
        the row that chose the consensus), ``ambiguous_fraction`` (fraction of covering fits where
        the best match was a near-tie). Overall: ``pairwise_agreement`` (mean over pairs of fits of
        the share of commonly covered rows with the same best match; NaN with fewer than two fits),
        ``n_fits``. Low agreement together with a high ambiguous fraction means the matching is not
        determined by the data at this resolution, not that one fit is right.
    """
    if not results:
        raise ValueError("results must contain at least one centroid_similarity output.")
    bests, ambig = [], []
    for r in results:
        bm = best_matches(
            r["similarity"], r["groups_a"], r["groups_b"], top_k=2, tie_tolerance=tie_tolerance
        )
        bests.append({m["row"]: m["top"][0] for m in bm["matches"] if m["valid"]})
        ambig.append({m["row"]: m["ambiguous"] for m in bm["matches"] if m["valid"]})
    all_rows: list[Any] = []
    for b in bests:
        all_rows.extend(k for k in b if k not in all_rows)
    with contextlib.suppress(TypeError):  # mixed ID types keep first-seen order
        all_rows = sorted(all_rows)
    coverage, consensus, agreement, amb_frac = [], [], [], []
    for rid in all_rows:
        chosen = [b[rid] for b in bests if rid in b]
        coverage.append(len(chosen) / len(results))
        top, count = Counter(chosen).most_common(1)[0]
        consensus.append(top)
        agreement.append(count / len(chosen))
        amb_frac.append(float(np.mean([a[rid] for a in ambig if rid in a])))
    pairs = []
    for i in range(len(bests)):
        for j in range(i + 1, len(bests)):
            common = set(bests[i]) & set(bests[j])
            if common:
                pairs.append(np.mean([bests[i][k] == bests[j][k] for k in common]))
    return {
        "row_ids": all_rows,
        "coverage": coverage,
        "consensus": consensus,
        "agreement": agreement,
        "ambiguous_fraction": amb_frac,
        "pairwise_agreement": float(np.mean(pairs)) if pairs else float("nan"),
        "n_fits": len(results),
    }
