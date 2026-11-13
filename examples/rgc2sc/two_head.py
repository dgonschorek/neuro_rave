"""The source notebook's two-head VAE-DANN on neuro_rave's VAEDANN core (research adapter).

``notebook_model.py`` keeps the notebook's original code for reference. This adapter runs the same
configuration through the package's private core (``neuro_rave._vaedann``), which was checked
against the notebook code: identical forward outputs on the shipped checkpoint, identical initial
weights for the same seed, and loss terms, gradients and two training epochs equal to
floating-point precision.

Two heads with **separate label spaces**: head A is trained on domain-0 samples (RGC group labels),
head B on domain-1 samples (SC cluster labels). They are never merged. The public
``neuro_rave.VAEDANN`` has one head and is not identical to this configuration.

Seeding: ``torch.manual_seed(seed)`` before building the networks (as the notebook's ``set_seed``),
so seed 42 gives the notebook's initial weights; latent noise and dropout then come from torch's
generator and the batch order from ``numpy.random.default_rng(seed)``. The notebook shuffled
batches with Python's ``random`` and was run on CUDA; that random sequence is not reproduced here,
so fresh fits are not expected to retrace the shipped checkpoint's training trajectory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from neuro_rave import _vaedann as core

REFERENCE = {
    "latent_dim": 8,
    "hidden_dim": 128,
    "dropout": 0.2,
    "epochs": 1000,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "kl_weight": 1e-2,
    "class_weight": 1.0,
    "domain_weight": 20.0,
    "ramp_rate": 5.0,
    "adversary_input": "sample",
    "decoder_conditioning": "none",
    "discriminator_steps": 0,
}
# Research configuration for the RGC/SC recordings from the 2026 follow-up (selected on validation
# recordings, confirmed on three seeds; not a default for other data): the domain classifier sees
# the posterior mean, the decoder also receives a one-hot domain code, and the adversary weight is
# 2. On validation recordings it lowered the k-NN domain probe and kept more reconstruction skill
# than REFERENCE (see README.md).
IMPROVED = {
    **REFERENCE,
    "adversary_input": "mean",
    "decoder_conditioning": "domain",
    "domain_weight": 2.0,
}
# Settings added after runs had been checkpointed: their value for those checkpoints.
_ADDED_SETTINGS = {"discriminator_steps": 0}
FORMAT = "neuro_rave.rgc2sc.TwoHeadVAEDANN/1"


class TwoHeadVAEDANN:
    """Two class heads, separate label spaces, exactly two domains (codes 0 and 1)."""

    def __init__(self, *, seed=0, **overrides):
        unknown = set(overrides) - set(REFERENCE)
        if unknown:
            raise ValueError(f"unknown settings: {sorted(unknown)}")
        self.seed = seed
        self.settings = {**REFERENCE, **overrides}
        self.nets = None
        self.history_ = None

    def _build(self, n_features, n_a, n_b):
        s = self.settings
        return core.VAEDANNNetworks(
            n_features,
            s["latent_dim"],
            s["hidden_dim"],
            2,
            (n_a, n_b),
            s["dropout"],
            decoder_domains=2 if s["decoder_conditioning"] == "domain" else 0,
        )

    def fit(
        self,
        X,
        domain,
        y_a,
        y_b,
        *,
        on_epoch_end=None,
        checkpoint=None,
        checkpoint_every=50,
        resume=False,
    ):
        """``domain`` 0/1 per sample; ``y_a`` / ``y_b`` integer codes, -1 where a head has no label.

        ``on_epoch_end(epoch, model)`` is called after every epoch (e.g. for learning curves).
        ``checkpoint`` saves the full training state every ``checkpoint_every`` epochs;
        ``resume=True`` continues an existing checkpoint with the same settings and data.
        """
        s = self.settings
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        d = torch.as_tensor(np.asarray(domain, dtype=np.int64))
        labels = torch.stack(
            [
                torch.as_tensor(np.asarray(y_a, dtype=np.int64)),
                torch.as_tensor(np.asarray(y_b, dtype=np.int64)),
            ],
            1,
        )
        if set(np.unique(d.numpy()).tolist()) != {0, 1}:
            raise ValueError("domain must contain exactly the codes 0 and 1")
        smallest = int(np.bincount(d.numpy()).min())
        settings = core.VAEDANNSettings(
            epochs=s["epochs"],
            per_domain=min(s["batch_size"] // 2, smallest),
            learning_rate=s["learning_rate"],
            weight_decay=s["weight_decay"],
            domain_weight=s["domain_weight"],
            kl_weight=s["kl_weight"],
            class_weight=s["class_weight"],
            device=torch.device("cpu"),
            adversary_input=s["adversary_input"],
            ramp_rate=s["ramp_rate"],
            discriminator_steps=s["discriminator_steps"],
        )
        fingerprint = core_fingerprint(X, d, labels)
        path = None if checkpoint is None else Path(checkpoint)
        saved = None
        if path is not None and path.exists():
            if not resume:
                raise FileExistsError(f"{path} exists; pass resume=True to continue it")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if (
                saved.get("format") != FORMAT
                or {**_ADDED_SETTINGS, **saved["settings"]} != s
                or saved["seed"] != self.seed
            ):
                raise ValueError(f"cannot resume {path}: different format, settings or seed")
            if saved["fingerprint"] != fingerprint:
                raise ValueError(f"cannot resume {path}: different data")
        rng = np.random.default_rng(self.seed)
        n_a, n_b = int(labels[:, 0].max()) + 1, int(labels[:, 1].max()) + 1

        def epoch_end(epoch, nets, opt, hist):
            self.nets = nets
            if path is not None and (epoch % checkpoint_every == 0 or epoch == s["epochs"]):
                core.save_checkpoint(
                    path,
                    {
                        "format": FORMAT,
                        "settings": s,
                        "seed": self.seed,
                        "fingerprint": fingerprint,
                        "heads": [n_a, n_b],
                        "n_features": int(X.shape[1]),
                        "epoch": epoch,
                        "model": nets.state_dict(),
                        "optimizer": opt.state_dict(),
                        "rng": core.rng_state(rng, torch.device("cpu")),
                        "history": hist,
                    },
                )
            if on_epoch_end is not None:
                on_epoch_end(epoch, self)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            nets = self._build(int(X.shape[1]), n_a, n_b)
            opt = core.make_optimizer(nets, settings)
            start, hist = 0, None
            if saved is not None:
                nets.load_state_dict(saved["model"])
                opt.load_state_dict(saved["optimizer"])
                core.restore_rng_state(saved["rng"], rng, torch.device("cpu"))
                start, hist = int(saved["epoch"]), saved["history"]
            self.nets = nets
            self.history_ = core.train(
                nets,
                X,
                d,
                labels,
                settings,
                rng,
                optimizer=opt,
                start_epoch=start,
                history=hist,
                on_epoch_end=epoch_end,
            )
        return self

    @torch.no_grad()
    def posterior(self, X):
        """Posterior mean and log-variance (eval mode)."""
        self.nets.eval()
        mu, logvar = self.nets.encode(torch.as_tensor(np.asarray(X, dtype=np.float32)))
        return mu.numpy(), logvar.numpy()

    def transform(self, X):
        return self.posterior(X)[0]

    @torch.no_grad()
    def reconstruct(self, X, domain=None, *, sampled=False, seed=0):
        """Decode the posterior mean (default) or a sampled latent (the notebook's evaluation)."""
        mu, logvar = (torch.as_tensor(a) for a in self.posterior(X))
        z = mu
        if sampled:
            g = torch.Generator().manual_seed(seed)
            z = mu + torch.randn(mu.shape, generator=g) * torch.exp(0.5 * logvar)
        codes = None if domain is None else torch.as_tensor(np.asarray(domain, dtype=np.int64))
        return self.nets.decode(z, codes).numpy()

    @torch.no_grad()
    def predict_proba(self, X, head):
        """Probabilities of head 0 (A) or 1 (B), from the posterior mean."""
        mu = torch.as_tensor(self.transform(X))
        return torch.softmax(self.nets.class_heads[head](mu), 1).numpy()


def core_fingerprint(X, d, labels):
    import hashlib

    h = hashlib.sha256()
    for t in (X, d, labels):
        h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()
