"""The original two-head VAE-DANN from the RGC-to-SC analysis notebook (research configuration).

This is research code for replaying the notebook's checkpoint, not part of the ``neuro_rave`` API.
It keeps what the public ``neuro_rave.VAEDANN`` deliberately does not:

- **two class heads with separate label spaces**: one trained on RGC group labels (RGC samples
  only), one on SC response-cluster labels (SC samples only). The labels are never merged;
- the notebook's settings: ``domain_weight=20``, ``kl_weight=0.01``, ramp rate 5, 1000 epochs;
- **sampled** inference (``reconstruct_sampled``), which the notebook used for its figures.

Exactly two domains (0 and 1); every minibatch holds ``batch_size // 2`` samples of each, and an
epoch ends when the smaller domain is used up. The loss arithmetic and update order are the
notebook's (cells 18-20). Only printing, plotting and DataLoader worker processes were removed.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

NOTEBOOK_SETTINGS = {
    "latent_dim": 8,
    "hidden_dim": 128,
    "dropout": 0.2,
    "epochs": 1000,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "kl_weight": 1e-2,
    "class_weight": 1.0,
    "domain_weight": 20.0,
    "max_alpha": 1.0,
    "ramp_rate": 5.0,
    "batch_size": 512,
}


class _ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):  # noqa: ANN001, D102
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # noqa: ANN001, D102
        return -ctx.alpha * grad_output, None


class TwoHeadVAEDANN(nn.Module):
    """Encoder (mean, log-variance), decoder, domain classifier and one class head per domain."""

    def __init__(
        self, input_dim, n_classes_a, n_classes_b, latent_dim=8, hidden_dim=128, dropout=0.2
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim)
        )
        self.head_a = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, n_classes_a)
        )
        self.head_b = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, n_classes_b)
        )
        self.domain_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.enc_mu(h), self.enc_logvar(h)


def fit_two_head(X, domain, y_a, y_b, *, seed=0, settings=None, log_every=0):
    """Train on float arrays. ``domain`` holds 0/1; ``y_a`` / ``y_b`` hold integer class codes.

    The codes are -1 where a sample has no label for that head (in the notebook: y_a for domain 0,
    y_b for domain 1).

    Returns ``(model, history)``. Seeds Python's, NumPy's and torch's global generators, as the
    notebook did.
    """
    s = {**NOTEBOOK_SETTINGS, **(settings or {})}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(X, dtype=np.float32))
    d = torch.as_tensor(np.asarray(domain, dtype=np.int64))
    ya = torch.as_tensor(np.asarray(y_a, dtype=np.int64))
    yb = torch.as_tensor(np.asarray(y_b, dtype=np.int64))
    model = TwoHeadVAEDANN(
        X.shape[1],
        int(ya.max()) + 1,
        int(yb.max()) + 1,
        s["latent_dim"],
        s["hidden_dim"],
        s["dropout"],
    )
    opt = torch.optim.Adam(
        model.parameters(), lr=s["learning_rate"], weight_decay=s["weight_decay"]
    )
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    pools = [np.flatnonzero(d.numpy() == k).tolist() for k in (0, 1)]
    half = s["batch_size"] // 2
    history = {k: [] for k in ("recon", "kl", "class_a", "class_b", "domain", "alpha")}
    for epoch in range(s["epochs"]):
        model.train()
        alpha = s["max_alpha"] * (
            2.0 / (1.0 + math.exp(-s["ramp_rate"] * epoch / s["epochs"])) - 1.0
        )
        for p in pools:
            random.shuffle(p)
        totals, n = dict.fromkeys(("recon", "kl", "class_a", "class_b", "domain"), 0.0), 0
        for i in range(0, min(map(len, pools)), half):
            idx = pools[0][i : i + half] + pools[1][i : i + half]
            if len(idx) < 2 * half:
                continue
            x, db, a, b = X[idx], d[idx], ya[idx], yb[idx]
            mu, logvar = model.encode(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            recon = mse(model.decoder(z), x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            la = ce(model.head_a(z)[a >= 0], a[a >= 0]) if bool((a >= 0).any()) else torch.zeros(())
            lb = ce(model.head_b(z)[b >= 0], b[b >= 0]) if bool((b >= 0).any()) else torch.zeros(())
            dom = ce(model.domain_head(_ReverseLayer.apply(z, alpha)), db)
            loss = (
                recon
                + s["kl_weight"] * kl
                + s["class_weight"] * (la + lb)
                + s["domain_weight"] * dom
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in zip(totals, (recon, kl, la, lb, dom), strict=True):
                totals[k] += float(v)
            n += 1
        for k, v in totals.items():
            history[k].append(v / max(n, 1))
        history["alpha"].append(alpha)
        if log_every and (epoch + 1) % log_every == 0:
            print(
                f"epoch {epoch + 1}: " + ", ".join(f"{k} {v[-1]:.4f}" for k, v in history.items())
            )
    model.eval()
    return model, history


@torch.no_grad()
def transform(model, X):
    """Posterior mean (what ``neuro_rave.VAEDANN.transform`` returns)."""
    return model.encode(torch.as_tensor(np.asarray(X, dtype=np.float32)))[0].numpy()


@torch.no_grad()
def reconstruct_sampled(model, X, seed=0):
    """Decode a *sampled* latent, as the notebook's evaluation cell did."""
    mu, logvar = model.encode(torch.as_tensor(np.asarray(X, dtype=np.float32)))
    g = torch.Generator().manual_seed(seed)
    z = mu + torch.randn(mu.shape, generator=g) * torch.exp(0.5 * logvar)
    return model.decoder(z).numpy()


def load_notebook_checkpoint(path, n_features=292, n_classes_a=32, n_classes_b=50):
    """Load the notebook's checkpoint (``{"model_state_dict": ...}``) safely into this class."""
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    renamed = {
        k.replace("type_head_rgc.", "head_a.")
        .replace("type_head_sc.", "head_b.")
        .replace("domain_head.net.", "domain_head."): v
        for k, v in state.items()
    }
    model = TwoHeadVAEDANN(n_features, n_classes_a, n_classes_b)
    model.load_state_dict(renamed)
    model.eval()
    return model
