"""Public VAEDANN behavior: shapes, labels, seeding, gradient reversal, failure reporting."""

import numpy as np
import pytest
import torch

from neuro_rave import VAEDANN
from neuro_rave._vaedann import balanced_batches, ramp, reverse_gradient

FAST = {"latent_dim": 3, "hidden_dim": 16, "epochs": 3, "batch_size": 12}


def _data(shape=(36, 10), domain_ids=("s1", "s2", "s3"), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=shape)
    domains = np.array(domain_ids, dtype=object)[np.arange(shape[0]) % len(domain_ids)]
    return X, domains


@pytest.mark.parametrize("shape", [(36, 10), (36, 7, 3)])
def test_shapes_dtypes_and_fitted_attributes(shape):
    X, domains = _data(shape)
    model = VAEDANN(**FAST, seed=0)
    assert model.fit(X, domains) is model
    Z, X_rec = model.transform(X[:5]), model.reconstruct(X[:5])
    assert Z.shape == (5, 3) and Z.dtype == np.float32
    assert X_rec.shape == (5, *shape[1:]) and X_rec.dtype == np.float32
    assert model.input_shape_ == shape[1:] and model.classes_ is None and model.epochs_ == 3
    assert list(model.domains_) == ["s1", "s2", "s3"]
    assert set(model.history_) == {
        "reconstruction",
        "kl",
        "classification",
        "domain",
        "domain_accuracy",
        "alpha",
    }
    assert all(len(v) == 3 for v in model.history_.values())
    assert model.history_["classification"] == [0.0, 0.0, 0.0]


def test_independent_fits_with_different_layouts():
    a = VAEDANN(**FAST, seed=0).fit(*_data((36, 10)))
    b = VAEDANN(**FAST, seed=0).fit(*_data((36, 25, 2)))
    assert a.input_shape_ == (10,) and b.input_shape_ == (25, 2)
    with pytest.raises(ValueError, match="same time length and channel layout"):
        a.transform(np.zeros((2, 25, 2)))


@pytest.mark.parametrize("domain_ids", [(10, 3, 7), ("left", "right")])
def test_nonconsecutive_integer_and_string_domains(domain_ids):
    X, domains = _data(domain_ids=domain_ids)
    model = VAEDANN(**FAST, seed=0).fit(X, domains)
    assert list(model.domains_) == sorted(domain_ids)


def test_partial_labels_original_values_and_probability_columns():
    X, domains = _data()
    labels = np.resize([10, 3, 7, 7], 36)
    y = [lbl if d != "s3" else None for lbl, d in zip(labels, domains, strict=True)]
    model = VAEDANN(**FAST, seed=0).fit(X, domains, y=y)
    assert model.classes_.tolist() == [3, 7, 10]
    proba = model.predict_proba(X)
    assert proba.shape == (36, 3) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    pred = model.predict(X)
    assert set(pred.tolist()) <= {3, 7, 10}
    assert np.array_equal(pred, model.classes_[proba.argmax(axis=1)])


def test_batches_without_labeled_samples_leave_class_head_untouched():
    X, domains = _data((60, 8), domain_ids=("a", "b"))
    y = [None] * 60
    y[0], y[1] = "on", "off"  # two labeled samples: most minibatches have none
    model = VAEDANN(**{**FAST, "batch_size": 4, "epochs": 2}, seed=0).fit(X, domains, y=y)
    assert np.all(np.isfinite(model.predict_proba(X)))


def test_predict_without_labels_raises_clearly():
    model = VAEDANN(**FAST, seed=0).fit(*_data())
    with pytest.raises(RuntimeError, match="without labels"):
        model.predict(np.zeros((2, 10)))
    with pytest.raises(RuntimeError, match="not fitted"):
        VAEDANN().transform(np.zeros((2, 10)))


def test_single_sample_domain_and_balanced_batches():
    X = np.random.default_rng(0).normal(size=(11, 6))
    domains = ["rare"] + ["common"] * 10
    model = VAEDANN(**FAST, seed=0).fit(X, domains)  # one sample per domain per batch
    assert model.transform(X).shape == (11, 3)
    codes = np.array([0] * 3 + [1] * 7)
    batches = list(balanced_batches(codes, 2, np.random.default_rng(0)))
    assert len(batches) == 1 and sorted(codes[batches[0]].tolist()) == [0, 0, 1, 1]


def test_default_epochs_match_sample_exposure_of_100_passes():
    X, domains = _data((40, 5), domain_ids=("a", "b"))
    domains = np.array(["a"] * 10 + ["b"] * 30)
    model = VAEDANN(latent_dim=2, hidden_dim=8, batch_size=8, seed=0).fit(X, domains)
    # 4 per domain per batch, 2 batches per epoch -> 16 presentations; 100 * 40 / 16 = 250
    assert model.epochs_ == 250


def test_seed_zero_reproducible_refit_and_global_rng_untouched():
    X, domains = _data()
    torch.manual_seed(123)
    np.random.seed(123)
    before_torch, before_np = torch.rand(1), np.random.rand()
    torch.manual_seed(123)
    np.random.seed(123)
    m = VAEDANN(**FAST, seed=0).fit(X, domains)
    assert torch.rand(1) == before_torch and np.random.rand() == before_np
    Z_first = m.transform(X)
    assert np.array_equal(Z_first, m.fit(X, domains).transform(X))  # refit starts fresh
    assert np.array_equal(Z_first, VAEDANN(**FAST, seed=0).fit(X, domains).transform(X))
    assert not np.array_equal(Z_first, VAEDANN(**FAST, seed=1).fit(X, domains).transform(X))


def test_posterior_mean_inference_is_deterministic():
    X, domains = _data()
    m = VAEDANN(**FAST, seed=0).fit(X, domains)
    assert np.array_equal(m.transform(X), m.transform(X))
    with torch.no_grad():
        decoded = m._networks.decoder(torch.from_numpy(m.transform(X))).numpy()
    assert np.allclose(m.reconstruct(X), decoded)


def test_gradient_reversal_direction_and_ramp():
    x = torch.ones(3, requires_grad=True)
    reverse_gradient(x, 0.5).sum().backward()
    assert torch.equal(x.grad, torch.full((3,), -0.5))
    assert ramp(0, 10) == 0.0 and 0 < ramp(5, 10) < ramp(9, 10) < 1.0


def test_zero_domain_weight_never_updates_domain_classifier():
    X, domains = _data()
    kw = {**FAST, "domain_weight": 0.0, "weight_decay": 0.0}
    short = VAEDANN(**{**kw, "epochs": 1}, seed=0).fit(X, domains)
    long = VAEDANN(**{**kw, "epochs": 4}, seed=0).fit(X, domains)
    for a, b in zip(
        short._networks.domain_head.parameters(),
        long._networks.domain_head.parameters(),
        strict=True,
    ):
        assert torch.equal(a, b)
    enc_short = next(short._networks.encoder.parameters())
    enc_long = next(long._networks.encoder.parameters())
    assert not torch.equal(enc_short, enc_long)


def test_divergence_raises_instead_of_skipping():
    X, domains = _data()
    X[0, 0] = 1e30  # squared error overflows float32
    with pytest.raises(FloatingPointError, match="diverged"):
        VAEDANN(**FAST, seed=0).fit(X, domains)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dropout": 1.0}, "dropout"),
        ({"domain_weight": -1}, "domain_weight"),
        ({"epochs": 0}, "epochs"),
    ],
)
def test_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        VAEDANN(**kwargs)
    with pytest.raises(ValueError, match="batch_size"):
        VAEDANN(batch_size=2).fit(*_data())


# ------------------------------------------------------------------ architecture options


def _capture_domain_head_input(adversary_input):
    from neuro_rave import _vaedann as core

    torch.manual_seed(0)
    nets = core.VAEDANNNetworks(10, 3, 16, 2, (), dropout=0.0)
    seen = {}
    nets.domain_head.register_forward_hook(lambda m, inp, out: seen.update(x=inp[0].detach()))
    x = torch.randn(8, 10)
    d = torch.tensor([0, 1] * 4)
    settings = core.VAEDANNSettings(
        epochs=1,
        per_domain=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        domain_weight=1.0,
        kl_weight=0.0,
        class_weight=0.0,
        device=torch.device("cpu"),
        adversary_input=adversary_input,
    )
    core.objective(nets, x, d, torch.full((8, 0), -1), 0.5, settings)
    mu, _ = nets.encode(x)
    return seen["x"], mu.detach()


def test_adversary_sees_posterior_mean_or_sample():
    seen, mu = _capture_domain_head_input("mean")
    assert torch.equal(seen, mu)
    seen, mu = _capture_domain_head_input("sample")
    assert not torch.equal(seen, mu)


def test_domain_conditioned_reconstruction_requires_aligned_known_ids():
    X, domains = _data(domain_ids=("b", "a", "c"))
    m = VAEDANN(**FAST, decoder_conditioning="domain", seed=0).fit(X, domains)
    assert m._networks.decoder[0].in_features == 3 + 3
    with pytest.raises(ValueError, match="conditioned on the domain"):
        m.reconstruct(X)
    with pytest.raises(ValueError, match="not seen during fit"):
        m.reconstruct(X[:2], domains=["a", "zz"])
    with pytest.raises(ValueError, match="one entry per sample"):
        m.reconstruct(X[:3], domains=["a"])
    rec = m.reconstruct(X, domains=domains)
    perm = np.random.default_rng(1).permutation(len(X))
    assert np.allclose(m.reconstruct(X[perm], domains=domains[perm]), rec[perm], atol=1e-6)
    swapped = m.reconstruct(X, domains=np.where(domains == "a", "b", "a"))
    assert not np.allclose(swapped, rec)  # the domain code changes the decoder output
    assert m.transform(X).shape == (36, 3)  # embedding needs no domains


def test_unconditioned_reconstruct_ignores_domains():
    X, domains = _data()
    m = VAEDANN(**FAST, seed=0).fit(X, domains)
    assert np.array_equal(m.reconstruct(X), m.reconstruct(X, domains=["unseen"] * 36))


def test_ramp_rate_and_params_are_honoured():
    X, domains = _data()
    m = VAEDANN(**{**FAST, "epochs": 4}, ramp_rate=1.0, seed=0).fit(X, domains)
    expected = [2 / (1 + np.exp(-1.0 * e / 4)) - 1 for e in range(4)]
    assert np.allclose(m.history_["alpha"], expected)
    params = m.get_params()
    assert params["ramp_rate"] == 1.0 and params["epochs"] == 4 and params["seed"] == 0
    fresh = VAEDANN(**params)
    assert fresh.get_params() == params and fresh.history_ is None
    assert m.config_["epochs_resolved"] == 4
    with pytest.raises(ValueError, match="adversary_input"):
        VAEDANN(adversary_input="z")
    with pytest.raises(ValueError, match="decoder_conditioning"):
        VAEDANN(decoder_conditioning="yes")
    with pytest.raises(ValueError, match="ramp_rate"):
        VAEDANN(ramp_rate=0)


# ------------------------------------------------------------------ checkpoint / resume


def test_resume_after_interruption_equals_uninterrupted_fit(tmp_path, monkeypatch):
    import neuro_rave.models as models

    X, domains = _data()
    y = [
        None if d == "s3" else c
        for d, c in zip(domains, np.resize(["u", "v", "w", "w"], 36), strict=True)
    ]
    kw = {**FAST, "epochs": 4, "decoder_conditioning": "domain", "adversary_input": "mean"}
    full = VAEDANN(**kw, seed=0).fit(X, domains, y=y)

    real_write = models._write_checkpoint

    def write_then_stop(path, payload):
        real_write(path, payload)
        if payload["epoch"] == 2:
            raise KeyboardInterrupt("simulated interruption")

    monkeypatch.setattr(models, "_write_checkpoint", write_then_stop)
    ckpt = tmp_path / "fit.ckpt"
    with pytest.raises(KeyboardInterrupt):
        VAEDANN(**kw, seed=0).fit(X, domains, y=y, checkpoint=ckpt, checkpoint_every=2)
    monkeypatch.setattr(models, "_write_checkpoint", real_write)
    resumed = VAEDANN(**kw, seed=0).fit(
        X, domains, y=y, checkpoint=ckpt, checkpoint_every=2, resume=True
    )
    assert np.array_equal(full.transform(X), resumed.transform(X))
    assert np.array_equal(full.predict_proba(X), resumed.predict_proba(X))
    assert full.history_ == resumed.history_
    # a finished checkpoint resumes without training and gives the same model
    again = VAEDANN(**kw, seed=0).fit(X, domains, y=y, checkpoint=ckpt, resume=True)
    assert np.array_equal(again.transform(X), full.transform(X))


def test_incompatible_or_unrequested_resume_is_refused(tmp_path):
    from neuro_rave._vaedann import load_checkpoint

    X, domains = _data()
    ckpt = tmp_path / "fit.ckpt"
    VAEDANN(**FAST, seed=0).fit(X, domains, checkpoint=ckpt)
    payload = load_checkpoint(ckpt)
    assert payload["domains"] == ["s1", "s2", "s3"] and payload["epoch"] == 3
    assert payload["config"]["domain_weight"] == 0.2 and len(payload["data_fingerprint"]) == 64
    with pytest.raises(FileExistsError, match="resume=True"):
        VAEDANN(**FAST, seed=0).fit(X, domains, checkpoint=ckpt)
    with pytest.raises(ValueError, match="settings differ"):
        VAEDANN(**FAST, seed=0, domain_weight=1.0).fit(X, domains, checkpoint=ckpt, resume=True)
    with pytest.raises(ValueError, match="differ from the checkpoint"):
        VAEDANN(**FAST, seed=0).fit(X + 1.0, domains, checkpoint=ckpt, resume=True)
    with pytest.raises(ValueError, match="needs checkpoint"):
        VAEDANN(**FAST, seed=0).fit(X, domains, resume=True)


def test_notebook_state_mapping_renames_without_dropping():
    from neuro_rave import _vaedann as core

    nets = core.VAEDANNNetworks(12, 3, 16, 2, (4, 5), 0.2)
    inverse = {}
    for key, value in nets.state_dict().items():
        old = key
        for nb_prefix, pkg_prefix in core.NOTEBOOK_KEY_MAP:
            if key.startswith(pkg_prefix):
                old = nb_prefix + key[len(pkg_prefix) :]
        inverse[old] = value
    assert any(k.startswith("type_head_sc.") for k in inverse)
    mapped = core.from_notebook_state(inverse)
    assert mapped.keys() == nets.state_dict().keys()
    nets.load_state_dict(mapped, strict=True)


def _one_batch_epoch(discriminator_steps, domain_weight=1.0):
    from neuro_rave import _vaedann as core

    torch.manual_seed(0)
    nets = core.VAEDANNNetworks(6, 2, 8, 2)
    x = torch.randn(8, 6)
    d = torch.tensor([0, 1] * 4)
    settings = core.VAEDANNSettings(
        epochs=2,
        per_domain=4,
        learning_rate=1e-2,
        weight_decay=0.0,
        domain_weight=domain_weight,
        kl_weight=1e-2,
        class_weight=1.0,
        device=torch.device("cpu"),
        discriminator_steps=discriminator_steps,
    )
    core.train(nets, x, d, torch.full((8, 0), -1), settings, np.random.default_rng(0))
    return nets.state_dict()


def test_extra_discriminator_steps_update_only_the_domain_classifier():
    # One epoch of one batch: the extra updates follow the main step, so only the domain
    # classifier may differ from a fit without them.
    from neuro_rave import _vaedann as core

    def first_epoch(steps):
        torch.manual_seed(0)
        nets = core.VAEDANNNetworks(6, 2, 8, 2)
        settings = core.VAEDANNSettings(
            1, 4, 1e-2, 0.0, 1.0, 1e-2, 1.0, torch.device("cpu"), discriminator_steps=steps
        )
        torch.manual_seed(1)
        x, d = torch.randn(8, 6), torch.tensor([0, 1] * 4)
        core.train(nets, x, d, torch.full((8, 0), -1), settings, np.random.default_rng(0))
        return nets.state_dict()

    plain, extra = first_epoch(0), first_epoch(3)
    for key in plain:
        same = torch.equal(plain[key], extra[key])
        assert same != key.startswith("domain_head."), key
    # Without an adversary the option changes nothing.
    a, b = _one_batch_epoch(0, domain_weight=0.0), _one_batch_epoch(3, domain_weight=0.0)
    assert all(torch.equal(a[k], b[k]) for k in a)
