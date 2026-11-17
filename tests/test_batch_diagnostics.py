"""assess_batch_effects: split design, leakage, label analyses and output contracts.

Statistical checks use generous tolerances on independent synthetic data; contract checks (split
validity, IDs, determinism, errors) are exact.
"""

import importlib.util

import numpy as np
import pytest

from neuro_rave import evaluation as ev

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="needs the evaluation extra (scikit-learn)"
)


def _recordings(
    seed,
    *,
    domains=("A", "B", "C"),
    recs_per_domain=4,
    cells=25,
    shift=0.0,
    dim=12,
    class_probs=None,
    rec_sd=0.0,
):
    """Synthetic samples nested in recordings nested in domains; 3 classes."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=2.0, size=(3, dim))
    X, d, g, y = [], [], [], []
    for di, dom in enumerate(domains):
        probs = None if class_probs is None else class_probs[di]
        for r in range(recs_per_domain):
            rec_offset = rng.normal(scale=rec_sd, size=dim)
            cls = rng.choice(3, size=cells, p=probs)
            X.append(centers[cls] + rec_offset + shift * di + rng.normal(size=(cells, dim)))
            d += [dom] * cells
            g += [f"{dom}-rec{r}"] * cells
            y += [f"type{c}" for c in cls]
    return np.vstack(X), np.array(d, dtype=object), np.array(g, dtype=object), y


# ------------------------------------------------------------------ scientific behaviour


@pytest.mark.parametrize("kind", ["linear", "knn"])
def test_no_shift_near_chance_and_strong_shift_detected(kind):
    X, d, g, _ = _recordings(0, shift=0.0)
    flat = ev.assess_batch_effects(X, d, groups=g, probes=[kind], n_splits=3)
    pooled = flat["probes"][kind]["pooled"]
    assert pooled["chance_balanced_accuracy"] == pytest.approx(1 / 3)
    assert pooled["balanced_accuracy"] < 1 / 3 + 0.15  # tolerance: finite random data

    X, d, g, _ = _recordings(1, shift=3.0)
    sep = ev.assess_batch_effects(X, d, groups=g, probes=[kind], n_splits=3)
    pooled = sep["probes"][kind]["pooled"]
    assert pooled["balanced_accuracy"] > 0.9
    assert sep["probes"][kind]["split_summary"]["n_splits_above_chance"] == 3
    assert any("detectable" in m for m in sep["messages"])
    assert sep["significance_test"] == "none"


def test_imbalanced_domains_use_balanced_baselines():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(160, 5)), rng.normal(loc=4.0, size=(40, 5))])
    d = ["big"] * 160 + ["small"] * 40
    out = ev.assess_batch_effects(X, d, n_splits=4, probes=["linear"])
    pooled = out["probes"]["linear"]["pooled"]
    assert pooled["chance_balanced_accuracy"] == 0.5
    assert pooled["majority_accuracy"] == pytest.approx(0.8)
    assert pooled["balanced_accuracy"] > 0.9
    assert out["domain_counts"] == {"big": 160, "small": 40}
    # balance=True: each split's probe trains on equal numbers per domain
    for res in out["probes"]["linear"]["per_split"]:
        counts = list(res["train_counts"].values())
        assert counts[0] == counts[1]


def test_group_leakage_inflates_sample_level_but_not_grouped_splits():
    # Domain is unrelated to the response; samples of one group share a strong signature.
    rng = np.random.default_rng(3)
    X, d, g = [], [], []
    for grp in range(24):
        sig = rng.normal(scale=4.0, size=8)
        dom = ["A", "B"][grp % 2]
        X.append(sig + rng.normal(scale=0.3, size=(15, 8)))
        d += [dom] * 15
        g += [grp] * 15
    X = np.vstack(X)
    rowwise = ev.assess_batch_effects(X, d, probes=["knn"], n_splits=3, n_neighbors=5)
    grouped = ev.assess_batch_effects(X, d, groups=g, probes=["knn"], n_splits=3, n_neighbors=5)
    assert rowwise["probes"]["knn"]["pooled"]["balanced_accuracy"] > 0.9  # memorised groups
    assert grouped["probes"]["knn"]["pooled"]["balanced_accuracy"] < 0.75
    assert any("independent" in lim for lim in rowwise["limitations"])


@pytest.mark.parametrize("kind", ["linear", "knn"])
def test_scaling_is_fitted_on_training_part_only(kind):
    X, d, _, _ = _recordings(4, shift=0.6, dim=6, recs_per_domain=2)
    test = np.arange(0, len(X), 4)
    train = np.setdiff1d(np.arange(len(X)), test)
    base = ev.assess_batch_effects(X, d, splits=[(train, test)], probes=[kind])
    X_moved = X.copy()
    X_moved[test[0], 0] = 1e6  # an extreme test sample must not change the scaling of the others
    moved = ev.assess_batch_effects(X_moved, d, splits=[(train, test)], probes=[kind])
    diff = np.abs(
        base["probes"][kind]["pooled"]["confusion_matrix"]
        - moved["probes"][kind]["pooled"]["confusion_matrix"]
    ).sum()
    assert diff <= 2  # at most the perturbed sample changes its predicted domain


def test_composition_only_difference_versus_within_class():
    # Same class responses in both domains, very different class proportions.
    # Tolerances were checked over data seeds 0-9 with scikit-learn 1.3.0 and 1.9.1:
    # overall >= 0.81, sample-weighted within-class mean <= 0.55, single classes <= 0.69.
    X, d, g, y = _recordings(
        5,
        domains=("A", "B"),
        recs_per_domain=6,
        cells=60,
        shift=0.0,
        class_probs=[[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
    )
    out = ev.assess_batch_effects(X, d, groups=g, y=y, n_splits=3, probes=["linear"])
    assert out["probes"]["linear"]["pooled"]["balanced_accuracy"] > 0.7
    within = [v for v in out["labels"]["within_class"].values() if v["status"] == "assessed"]
    assert within
    scores = [
        (v["n_samples"], v["probes"]["linear"]["pooled"]["balanced_accuracy"]) for v in within
    ]
    assert sum(n * s for n, s in scores) / sum(n for n, _ in scores) < 0.62
    assert all(s < 0.8 for _, s in scores)
    assert out["labels"]["max_class_share_difference"] > 0.5


# ------------------------------------------------------------------ split design contracts


def test_grouped_splits_have_no_group_overlap_and_full_domain_coverage():
    X, d, g, _ = _recordings(6, shift=1.0)
    out = ev.assess_batch_effects(X, d, groups=g, n_splits=4, n_repeats=2)
    split = out["split"]
    assert split["strategy"] == "stratified_group_kfold"
    assert len(split["indices"]) == 8 and split["n_repeats"] == 2
    for part in split["indices"]:
        tr, te = part["train"], part["test"]
        assert not set(g[tr]) & set(g[te])
        assert set(d[tr]) == set(d[te]) == {"A", "B", "C"}
    assert split["n_scored_samples"] == len(X) and split["times_scored"] == {"min": 2, "max": 2}
    assert out["groups"]["groups_per_domain"] == {"A": 4, "B": 4, "C": 4}


def test_coinciding_repeats_are_reported():
    # Two groups per domain allow only two distinct 2-fold assignments, so three repeats coincide.
    X, d, g, _ = _recordings(17, domains=("A", "B"), recs_per_domain=2)
    out = ev.assess_batch_effects(X, d, groups=g, n_splits=2, n_repeats=3, probes=["linear"])
    assert out["split"]["n_repeats"] == 3 and out["split"]["n_distinct_repeats"] < 3
    assert any("distinct splits" in lim for lim in out["limitations"])


def test_groups_equal_to_domains_is_rejected_with_explanation():
    X, d, _, _ = _recordings(7)
    with pytest.raises(ValueError, match="groups=domains"):
        ev.assess_batch_effects(X, d, groups=d)


def test_too_few_groups_per_domain_is_rejected_without_fallback():
    X, d, g, _ = _recordings(8, recs_per_domain=2)
    with pytest.raises(ValueError, match="n_splits=3"):
        ev.assess_batch_effects(X, d, groups=g, n_splits=3)
    ok = ev.assess_batch_effects(X, d, groups=g, n_splits=2, probes=["linear"])
    assert ok["split"]["strategy"] == "stratified_group_kfold"


def test_explicit_splits_are_validated():
    X, d, g, _ = _recordings(9)
    n = len(X)
    a_rows, other = np.flatnonzero(d == "A"), np.flatnonzero(d != "A")
    good_tr, good_te = np.arange(0, n, 2), np.arange(1, n, 2)
    bad = {
        "both the training and the test": [(good_tr, np.r_[good_te, good_tr[:1]])],
        "duplicates": [(np.r_[good_tr, good_tr[:1]], good_te)],
        r"lie in \[0": [(good_tr, np.r_[good_te, n])],
        "integers": [(good_tr.astype(float), good_te)],
        "no samples in the test": [(np.r_[good_tr, a_rows[1::2]], np.setdiff1d(other, good_tr))],
        "non-empty": [(good_tr, [])],
    }
    for pattern, splits in bad.items():
        with pytest.raises(ValueError, match=pattern):
            ev.assess_batch_effects(X, d, splits=splits)
    with pytest.raises(ValueError, match="group"):
        ev.assess_batch_effects(X, d, groups=g, splits=[(good_tr, good_te)])
    with pytest.raises(ValueError, match="omit"):
        ev.assess_batch_effects(X, d, splits=[(good_tr, good_te)], n_splits=3)
    out = ev.assess_batch_effects(X, d, splits=[(good_tr, good_te)], probes=["linear"])
    assert out["split"]["strategy"] == "explicit"
    assert np.array_equal(out["split"]["indices"][0]["test"], good_te)


# ------------------------------------------------------------------ labels


def test_missing_absent_and_confounded_labels():
    X, d, g, y = _recordings(10, recs_per_domain=4)
    y = list(y)
    for i in range(0, len(y), 5):
        y[i] = None  # unknown labels
    for i in np.flatnonzero(d == "A"):
        if y[i] == "type2":
            y[i] = "only_in_A"  # a class labelled in one domain only
    for i in np.flatnonzero(d == "C"):
        if y[i] == "type1":
            y[i] = None  # type1 is absent from labelled C samples
    out = ev.assess_batch_effects(X, d, groups=g, y=y, n_splits=2, probes=["linear"])
    labels = out["labels"]
    assert labels["n_unknown"] == sum(v is None for v in y)
    assert labels["confounded_classes"] == ["only_in_A"]
    assert labels["domains_per_class"]["type1"] == ["A", "B"]
    assert "only_in_A" not in labels["within_class"]
    assert any("cannot be disentangled" in m for m in out["messages"])
    assert labels["within_class"]["type1"]["status"] == "assessed"
    assert labels["within_class"]["type1"]["domains"] == ["A", "B"]


def test_within_class_budget_and_skip_reasons():
    X, d, g, y = _recordings(11)
    out = ev.assess_batch_effects(X, d, groups=g, y=y, n_splits=2, probes=["linear"], max_classes=1)
    statuses = [v["status"] for v in out["labels"]["within_class"].values()]
    assert statuses.count("assessed") == 1 and statuses.count("skipped") == 2
    assert all(
        "budget" in v["reason"]
        for v in out["labels"]["within_class"].values()
        if v["status"] == "skipped"
    )
    none = ev.assess_batch_effects(
        X, d, groups=g, y=y, n_splits=2, probes=["linear"], max_classes=0
    )
    assert all(v["status"] == "skipped" for v in none["labels"]["within_class"].values())
    no_labels = ev.assess_batch_effects(X, d, groups=g, n_splits=2, probes=["linear"])
    assert no_labels["labels"] is None


def test_within_class_respects_groups_and_reports_insufficient_units():
    X, d, g, y = _recordings(12, recs_per_domain=3)
    y = list(y)
    keep_one = {"C-rec0"}
    for i in np.flatnonzero(d == "C"):
        if y[i] == "type0" and g[i] not in keep_one:
            y[i] = None  # type0 in C lives in a single recording only
    out = ev.assess_batch_effects(X, d, groups=g, y=y, n_splits=2, probes=["linear"])
    entry = out["labels"]["within_class"]["type0"]
    assert entry["excluded_domains"] == {"C": "1 group(s) < n_splits=2"}
    assert entry["status"] == "assessed" and entry["domains"] == ["A", "B"]


# ------------------------------------------------------------------ inputs and outputs


def test_ids_shapes_and_determinism():
    X, d, g, y = _recordings(13, dim=12)
    ints = np.array([{"A": 10, "B": 20, "C": 30}[v] for v in d])
    X_3d = X.reshape(len(X), 6, 2)  # (samples, time, channels)
    a = ev.assess_batch_effects(X_3d, ints, groups=g, y=y, n_splits=3, seed=4)
    b = ev.assess_batch_effects(X.reshape(len(X), 12), ints, groups=g, y=y, n_splits=3, seed=4)
    assert a["domains"] == [10, 20, 30] and a["sample_shape"] == [6, 2] and a["n_features"] == 12
    assert a["labels"]["classes"] == ["type0", "type1", "type2"]
    for pa, pb in zip(a["split"]["indices"], b["split"]["indices"], strict=True):
        assert np.array_equal(pa["test"], pb["test"])
    for kind in ("linear", "knn"):
        assert a["probes"][kind]["pooled"]["balanced_accuracy"] == pytest.approx(
            b["probes"][kind]["pooled"]["balanced_accuracy"]
        )
    # Different seeds give different splits, though two seeds can coincide with few groups.
    assignments = {
        tuple(
            tuple(p["test"].tolist())
            for p in ev.assess_batch_effects(
                X_3d, ints, groups=g, n_splits=3, seed=s, probes=["linear"]
            )["split"]["indices"]
        )
        for s in range(6)
    }
    assert len(assignments) > 1
    assert a["settings"]["seed"] == 4 and a["settings"]["probes"] == ["linear", "knn"]


def test_inputs_and_global_random_state_are_untouched():
    X, d, g, y = _recordings(14)
    X_copy, y_copy = X.copy(), list(y)
    state = np.random.get_state()
    ev.assess_batch_effects(X, d, groups=g, y=y, n_splits=2)
    after = np.random.get_state()
    assert np.array_equal(X, X_copy) and y == y_copy
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_invalid_inputs():
    X, d, _, _ = _recordings(15)
    with pytest.raises(ValueError, match="NaN"):
        bad = X.copy()
        bad[0, 0] = np.nan
        ev.assess_batch_effects(bad, d)
    with pytest.raises(ValueError, match="entries"):
        ev.assess_batch_effects(X, d[:-1])
    with pytest.raises(ValueError, match="two domains"):
        ev.assess_batch_effects(X, ["A"] * len(X))
    with pytest.raises(ValueError, match="probes"):
        ev.assess_batch_effects(X, d, probes=["svm"])
    with pytest.raises(ValueError, match="n_splits"):
        ev.assess_batch_effects(X, d, n_splits=1)
    with pytest.raises(ValueError, match="n_neighbors"):
        ev.assess_batch_effects(X, d, probes=["knn"], n_neighbors=10_000)
    with pytest.raises(ValueError, match="None"):
        ev.assess_batch_effects(X, [None] + list(d[1:]))


def test_missing_sklearn_gives_install_hint(monkeypatch):
    def no_sklearn(name, *args):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(ev.importlib, "import_module", no_sklearn)
    X, d, g, _ = _recordings(16)
    with pytest.raises(ImportError, match=r"neuro-rave\[evaluation\]"):
        ev.assess_batch_effects(X, d, groups=g)
