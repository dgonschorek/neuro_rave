"""neuro_rave.evaluation: contracts of the public scoring functions."""

import importlib.util

import numpy as np
import pytest

from neuro_rave import evaluation as ev

# ------------------------------------------------------------------ reconstruction_scores


def test_reconstruction_mse_baseline_and_skill_by_hand():
    X = np.array([[1.0, 3.0], [2.0, 2.0], [0.0, 4.0]])
    X_hat = np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 4.0]])
    ref = np.array([1.0, 3.0])
    out = ev.reconstruction_scores(X, X_hat, reference_mean=ref, groups=["a", "b", "a"])
    assert out["mse"] == pytest.approx(2.0 / 6)
    assert out["baseline_mse"] == pytest.approx((0 + 0 + 1 + 1 + 1 + 1) / 6)
    assert out["reconstruction_skill"] == pytest.approx(1 - 2.0 / 4.0)
    assert np.allclose(out["per_sample_mse"], [0.5, 0.0, 0.5])
    assert out["per_group"]["a"] == pytest.approx(
        {"n_samples": 2, "mse": 0.5, "baseline_mse": 0.5, "reconstruction_skill": 0.0}
    )
    assert out["per_group"]["b"]["reconstruction_skill"] == 1.0


def test_reconstruction_skill_is_nan_when_baseline_error_is_zero():
    X = np.array([[1.0, 3.0], [1.0, 3.0]])
    out = ev.reconstruction_scores(X, X + 0.5, reference_mean=[1.0, 3.0])
    assert out["baseline_mse"] == 0.0 and np.isnan(out["reconstruction_skill"])


def test_reconstruction_reference_mean_predictor_scores_zero_and_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5, 2))
    ref = X.mean(axis=0)
    out = ev.reconstruction_scores(X, np.broadcast_to(ref, X.shape), reference_mean=ref)
    assert out["reconstruction_skill"] == pytest.approx(0.0)
    assert out["n_features"] == 10 and "per_group" not in out
    perfect = ev.reconstruction_scores(X, X, reference_mean=ref)
    assert perfect["mse"] == 0.0 and perfect["reconstruction_skill"] == 1.0


def test_reconstruction_without_reference_and_bad_inputs():
    X = np.ones((3, 4))
    out = ev.reconstruction_scores(X, X)
    assert "baseline_mse" not in out and out["mse"] == 0.0
    with pytest.raises(ValueError, match="shape"):
        ev.reconstruction_scores(X, np.ones((3, 5)))
    with pytest.raises(ValueError, match="NaN"):
        ev.reconstruction_scores(X, np.full((3, 4), np.nan))
    with pytest.raises(ValueError, match="reference_mean"):
        ev.reconstruction_scores(X, X, reference_mean=np.ones(3))


# ------------------------------------------------------------------ classification_scores


def test_classification_missing_truth_is_excluded_and_counts_reported():
    y_true = ["on", None, "off", "off", "on", None]
    y_pred = ["on", "off", "on", "off", "on", "on"]
    out = ev.classification_scores(y_true, y_pred)
    assert out["classes"] == ["off", "on"]
    assert (out["n_total"], out["n_scored"], out["n_missing_true"]) == (6, 4, 2)
    assert out["accuracy"] == pytest.approx(3 / 4)
    assert out["balanced_accuracy"] == pytest.approx((1 / 2 + 1) / 2)
    assert out["confusion_matrix"].tolist() == [[1, 1], [0, 2]]
    assert out["majority_accuracy"] == pytest.approx(0.5)


def test_classification_imbalance_and_absent_predicted_class():
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10
    out = ev.classification_scores(y_true, y_pred, classes=[0, 1, 2])
    assert out["accuracy"] == pytest.approx(0.9)
    assert out["balanced_accuracy"] == pytest.approx(0.5)  # mean recall over classes with support
    assert out["chance_balanced_accuracy"] == pytest.approx(0.5)
    assert out["majority_accuracy"] == pytest.approx(0.9)
    assert np.isnan(out["recall_per_class"][2]) and out["support_per_class"][2] == 0
    with pytest.raises(ValueError, match="not in classes"):
        ev.classification_scores([0, 3], [0, 0], classes=[0, 1])


def test_classification_groups_and_probability_alignment():
    y_true = np.array([10, 3, 3, 10])
    y_pred = np.array([10, 3, 10, 10])
    proba = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.3, 0.7]])  # columns: 3, 10
    out = ev.classification_scores(
        y_true, y_pred, groups=["s1", "s1", "s2", "s2"], y_proba=proba, proba_classes=[3, 10]
    )
    expected = -np.mean(np.log([0.9, 0.8, 0.4, 0.7]))
    assert out["log_loss"] == pytest.approx(expected)
    assert out["per_group"]["s1"]["accuracy"] == 1.0 and out["per_group"]["s2"]["accuracy"] == 0.5
    with pytest.raises(ValueError, match="proba_classes"):
        ev.classification_scores(y_true, y_pred, y_proba=proba)
    with pytest.raises(ValueError, match="probability column"):
        ev.classification_scores(y_true, y_pred, y_proba=proba, proba_classes=[3, 11])


def test_classification_rejects_nan_labels():
    with pytest.raises(ValueError, match="None"):
        ev.classification_scores([1.0, float("nan")], [1.0, 1.0])


# ------------------------------------------------------------------ domain_probe

needs_sklearn = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="needs the evaluation extra (scikit-learn)"
)


def _shifted(n_a, n_b, shift, seed, dim=4):
    rng = np.random.default_rng(seed)
    Z = np.vstack([rng.normal(size=(n_a, dim)), rng.normal(loc=shift, size=(n_b, dim))])
    return Z, np.array(["A"] * n_a + ["B"] * n_b)


@needs_sklearn
@pytest.mark.parametrize("kind", ["knn", "linear"])
def test_domain_probe_separable_and_mixed(kind):
    Z_tr, d_tr = _shifted(200, 200, 6.0, 0)
    Z_te, d_te = _shifted(50, 150, 6.0, 1)
    sep = ev.domain_probe(Z_tr, d_tr, Z_te, d_te, kind=kind)
    assert sep["balanced_accuracy"] > 0.95
    assert sep["classes"] == ["A", "B"] and sep["test_counts"] == {"A": 50, "B": 150}
    assert sep["majority_accuracy"] == pytest.approx(0.75)  # not 0.5 for unequal test domains
    assert sep["chance_balanced_accuracy"] == 0.5 and sep["uniform_log_loss"] == pytest.approx(
        np.log(2)
    )
    Z_tr, d_tr = _shifted(200, 200, 0.0, 2)
    Z_te, d_te = _shifted(100, 100, 0.0, 3)
    mixed = ev.domain_probe(Z_tr, d_tr, Z_te, d_te, kind=kind)
    assert 0.3 < mixed["balanced_accuracy"] < 0.7
    assert np.isfinite(mixed["log_loss"]) and mixed["confusion_matrix"].sum() == 200


@needs_sklearn
def test_domain_probe_balances_training_and_scales_on_training_only():
    Z_tr, d_tr = _shifted(300, 60, 3.0, 0)
    Z_te, d_te = _shifted(40, 40, 3.0, 1)
    out = ev.domain_probe(Z_tr, d_tr, Z_te, d_te, kind="knn", seed=0)
    assert out["train_counts"] == {"A": 60, "B": 60} and out["n_train_used"] == 120
    assert out["prior_log_loss"] == pytest.approx(np.log(2))
    # Scaling must not use test samples: shifting all test samples far away changes predictions,
    # whereas test-fitted scaling would remove the shift.
    far = ev.domain_probe(Z_tr, d_tr, Z_te + 100.0, d_te, kind="linear")
    assert far["balanced_accuracy"] == pytest.approx(0.5)
    unbalanced = ev.domain_probe(Z_tr, d_tr, Z_te, d_te, balance=False, max_train_per_domain=100)
    assert unbalanced["train_counts"] == {"A": 100, "B": 60}


@needs_sklearn
def test_domain_probe_rejects_unseen_test_domain_and_single_domain():
    Z, d = _shifted(30, 30, 1.0, 0)
    with pytest.raises(ValueError, match="absent from training"):
        ev.domain_probe(Z, d, Z[:5], np.array(["C"] * 5))
    with pytest.raises(ValueError, match="two domains"):
        ev.domain_probe(Z, np.array(["A"] * 60), Z, np.array(["A"] * 60))


# ------------------------------------------------------------------ response_correlation


def test_response_correlation_matches_numpy_and_flags_constant_samples():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 30))
    X_hat = X + rng.normal(scale=0.5, size=X.shape)
    X_hat[2] = 1.0  # constant reconstruction
    out = ev.response_correlation(X, X_hat)
    ref = [np.corrcoef(a, b)[0, 1] for a, b in zip(X, X_hat, strict=True)]
    assert np.isnan(out["per_sample"][2]) and out["n_constant"] == 1 and out["n_defined"] == 5
    keep = [0, 1, 3, 4, 5]
    assert np.allclose(out["per_sample"][keep], np.array(ref)[keep])
    assert out["mean"] == pytest.approx(np.mean(np.array(ref)[keep]))


def test_response_correlation_channels():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 20, 3))
    assert ev.response_correlation(X, X)["per_sample"].shape == (4,)
    sep = ev.response_correlation(X, X, channels="separate")
    assert sep["per_sample"].shape == (4, 3) and np.allclose(sep["per_sample"], 1.0)


# ------------------------------------------------------------------ matching


def test_centroid_similarity_ids_orientation_and_metrics():
    Z_a = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    g_a = [7, 7, 2]
    Z_b = np.array([[0.0, 1.0], [3.0, 0.0], [5.0, 5.0]])
    g_b = ["x", "y", None]
    out = ev.centroid_similarity(Z_a, g_a, Z_b, g_b)
    assert out["groups_a"] == [2, 7] and out["groups_b"] == ["x", "y"]
    assert out["similarity"].shape == (2, 2)
    assert np.allclose(out["similarity"], [[1.0, 0.0], [0.0, 1.0]])
    assert out["counts_a"] == {2: 1, 7: 2}
    few = ev.centroid_similarity(Z_a, g_a, np.vstack([Z_b, Z_b]), g_b + g_b, min_samples=2)
    assert few["groups_a"] == [7] and few["dropped_a"] == [2]
    rng = np.random.default_rng(0)
    za, zb = rng.normal(size=(40, 6)), rng.normal(size=(40, 6))
    ga, gb = np.arange(40) % 4, np.arange(40) % 5
    p = ev.centroid_similarity(za, ga, zb, gb, metric="pearson")["similarity"]
    ca = np.vstack([za[ga == k].mean(0) for k in range(4)])
    cb = np.vstack([zb[gb == k].mean(0) for k in range(5)])
    assert np.allclose(p, np.corrcoef(ca, cb)[:4, 4:])


def test_best_matches_preserves_ids_margins_ties_and_invalid_rows():
    sim = np.array([[0.2, 0.9, 0.5], [0.4, 0.4, 0.1], [np.nan, np.nan, np.nan]])
    out = ev.best_matches(sim, ["G1", "G2", "G3"], [10, 20, 30], top_k=2, tie_tolerance=0.01)
    m1, m2, m3 = out["matches"]
    assert m1["top"] == [20, 30] and m1["margin"] == pytest.approx(0.4) and not m1["ambiguous"]
    assert m2["top"] == [10, 20] and m2["ambiguous"]  # tie broken by column order, flagged
    assert not m3["valid"] and m3["top"] == [] and out["best"]["G3"] is None
    assert out["best"] == {"G1": 20, "G2": 10, "G3": None}
    assert (out["n_valid"], out["n_ambiguous"]) == (2, 1)


def test_matching_agreement_aligns_ids_across_fits():
    run1 = {
        "similarity": np.array([[0.9, 0.1], [0.2, 0.8]]),
        "groups_a": ["a", "b"],
        "groups_b": [1, 2],
    }
    # same matches, rows and columns stored in a different order and one extra column
    run2 = {
        "similarity": np.array([[0.7, 0.0, 0.1], [0.2, 0.0, 0.95]]),
        "groups_a": ["b", "a"],
        "groups_b": [2, 3, 1],
    }
    run3 = {"similarity": np.array([[0.5, 0.6]]), "groups_a": ["a"], "groups_b": [1, 2]}
    out = ev.matching_agreement([run1, run2, run3])
    assert out["row_ids"] == ["a", "b"]
    assert out["consensus"] == [1, 2]
    assert out["coverage"] == pytest.approx([1.0, 2 / 3])
    assert out["agreement"] == pytest.approx([2 / 3, 1.0])
    assert out["pairwise_agreement"] == pytest.approx((1.0 + 0.0 + 0.0) / 3)


# ------------------------------------------------------------------ optional dependency


def test_missing_sklearn_gives_install_hint(monkeypatch):
    def no_sklearn(name, *args):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(ev.importlib, "import_module", no_sklearn)
    Z, d = _shifted(20, 20, 1.0, 0)
    with pytest.raises(ImportError, match=r"neuro-rave\[evaluation\]"):
        ev.domain_probe(Z, d, Z, d)
