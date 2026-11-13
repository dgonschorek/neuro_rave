"""Public RAVE / RAVEPlus behavior: shapes, label mapping, fitted state, seeding."""

import numpy as np
import pytest
import torch

from neuro_rave import RAVE, RAVEPlus

FAST = {"latent_dim": 3, "hidden_dims": (16,), "epochs": 3, "batch_size": 8}


def _data(shape=(30, 12), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=shape)
    domains = np.array(["s1", "s2", "s3"])[np.arange(shape[0]) % 3]
    return X, domains


@pytest.mark.parametrize("shape", [(30, 12), (30, 12, 2)])
def test_rave_without_labels_shapes_and_dtypes(shape):
    X, domains = _data(shape)
    model = RAVE(**FAST, seed=0)
    assert model.fit(X, domains) is model
    Z = model.transform(X[:7])
    X_rec = model.reconstruct(X[:7])
    assert Z.shape == (7, 3) and Z.dtype == np.float32
    assert X_rec.shape == (7, *shape[1:]) and X_rec.dtype == np.float32
    assert model.input_shape_ == shape[1:]
    assert list(model.domains_) == ["s1", "s2", "s3"]
    assert all(len(v) == FAST["epochs"] for v in model.history_.values())
    assert model.history_["classification"] == [0.0] * FAST["epochs"]


@pytest.mark.parametrize("shape", [(30, 12), (30, 12, 2)])
def test_rave_plus_partial_labels_shapes_and_mapping(shape):
    X, domains = _data(shape)
    labels = np.resize([10, 3, 7, 7], 30)
    y = [lbl if d != "s3" else None for lbl, d in zip(labels, domains, strict=True)]
    model = RAVEPlus(**FAST, seed=0).fit(X, domains, y=y)
    assert model.classes_.tolist() == [3, 7, 10]
    proba = model.predict_proba(X)
    assert proba.shape == (30, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    pred = model.predict(X)
    assert set(pred.tolist()) <= {3, 7, 10}
    np.testing.assert_array_equal(pred, model.classes_[proba.argmax(axis=1)])
    assert model.transform(X).shape == (30, 3)
    assert model.reconstruct(X).shape == X.shape


@pytest.mark.parametrize("cls", [RAVE, RAVEPlus])
def test_default_domain_weight(cls):
    assert cls().domain_weight == 0.1
    assert cls(domain_weight=1.0).domain_weight == 1.0


@pytest.mark.parametrize("window", [(7,), (41, 3), (1, 5)])
def test_arbitrary_window_layout_and_integer_ids(window):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(23, *window))
    domains = np.resize([1001, -4, 17], 23)
    y = [None if i % 3 == 0 else (250, -2)[i % 2] for i in range(23)]
    model = RAVEPlus(**FAST, seed=0).fit(X, domains, y=y)
    assert model.input_shape_ == window
    assert model.domains_.tolist() == [-4, 17, 1001]
    assert model.classes_.tolist() == [-2, 250]
    assert model.transform(X[:5]).shape == (5, 3)
    assert model.reconstruct(X[:5]).shape == (5, *window)
    assert set(model.predict(X).tolist()) <= {-2, 250}


def test_rave_plus_string_labels_known_in_any_domain():
    X, domains = _data()
    y = [None] * 30
    y[0], y[4], y[8] = "on", "off", "on"
    pred = RAVEPlus(**FAST, seed=0).fit(X, domains, y=y).predict(X)
    assert set(pred.tolist()) <= {"off", "on"}


def test_mostly_unlabeled_batches_train_finitely():
    X, domains = _data()
    y = [None] * 30
    y[0], y[1] = "a", "b"
    model = RAVEPlus(**{**FAST, "batch_size": 2}, seed=0).fit(X, domains, y=y)
    assert all(np.isfinite(v).all() for v in model.history_.values())


@pytest.mark.parametrize("batch_size", [1, 29])
def test_final_minibatch_of_one_sample(batch_size):
    X, domains = _data()
    model = RAVE(**{**FAST, "batch_size": batch_size}, seed=0).fit(X, domains)
    assert np.isfinite(model.transform(X)).all()


def test_seed_zero_is_honored_and_reproducible():
    X, domains = _data()
    a = RAVE(**FAST, seed=0).fit(X, domains).transform(X)
    b = RAVE(**FAST, seed=0).fit(X, domains).transform(X)
    c = RAVE(**FAST, seed=1).fit(X, domains).transform(X)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seed_does_not_touch_global_rng():
    X, domains = _data()
    torch.manual_seed(123)
    expected = torch.rand(1)
    torch.manual_seed(123)
    RAVE(**FAST, seed=0).fit(X, domains)
    assert torch.equal(torch.rand(1), expected)


def test_refit_starts_fresh():
    X, domains = _data()
    once = RAVE(**FAST, seed=0).fit(X, domains).transform(X)
    model = RAVE(**FAST, seed=0)
    model.fit(*_data((30, 5, 2), seed=1))
    assert model.input_shape_ == (5, 2)
    np.testing.assert_array_equal(model.fit(X, domains).transform(X), once)
    assert model.input_shape_ == (12,)


def test_inference_preserves_row_order():
    X, domains = _data()
    model = RAVE(**FAST, seed=0).fit(X, domains)
    np.testing.assert_allclose(
        model.transform(X[::-1]), model.transform(X)[::-1], rtol=1e-5, atol=1e-6
    )


def test_float64_and_float32_inputs_agree():
    X, domains = _data()
    model = RAVE(**FAST, seed=0).fit(X, domains)
    np.testing.assert_array_equal(model.transform(X), model.transform(X.astype(np.float32)))


def test_not_fitted_errors():
    with pytest.raises(RuntimeError, match="not fitted"):
        RAVE().transform(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="not fitted"):
        RAVEPlus().predict(np.zeros((2, 3)))


def test_inference_checks_window_layout():
    X, domains = _data((30, 12, 2))
    model = RAVE(**FAST, seed=0).fit(X, domains)
    with pytest.raises(ValueError, match=r"\(samples, 12, 2\)"):
        model.transform(np.zeros((4, 12)))
    with pytest.raises(ValueError, match=r"\(samples, 12, 2\)"):
        model.reconstruct(np.zeros((4, 11, 2)))


def test_fit_validation_errors():
    X, domains = _data()
    with pytest.raises(ValueError, match="at least 2 distinct"):
        RAVE(**FAST).fit(X, ["only"] * 30)
    with pytest.raises(ValueError, match="at least 2 distinct"):
        RAVEPlus(**FAST).fit(X, domains, y=["a"] * 10 + [None] * 20)
    with pytest.raises(ValueError, match="29 entries, but X has 30 rows"):
        RAVEPlus(**FAST).fit(X, domains, y=[None] * 29)
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        RAVE(**FAST).fit(X_bad, domains)


def test_failed_refit_keeps_previous_fit():
    X, domains = _data()
    model = RAVE(**FAST, seed=0).fit(X, domains)
    before = model.transform(X)
    with pytest.raises(ValueError):
        model.fit(X, ["only"] * 30)
    np.testing.assert_array_equal(model.transform(X), before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"latent_dim": 0},
        {"epochs": 1.5},
        {"batch_size": True},
        {"learning_rate": 0.0},
        {"domain_weight": -1.0},
        {"domain_weight": float("nan")},
        {"seed": -1},
        {"device": "mps"},
    ],
)
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        RAVE(**kwargs)


def test_class_weight_validation():
    with pytest.raises(ValueError):
        RAVEPlus(class_weight=-0.1)


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the CUDA-unavailable error")
def test_cuda_unavailable_is_reported():
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        RAVE(device="cuda")


def test_public_api_and_version():
    import neuro_rave

    assert set(neuro_rave.__all__) == {"RAVE", "RAVEPlus", "VAEDANN", "__version__"}
    assert isinstance(neuro_rave.__version__, str) and neuro_rave.__version__
