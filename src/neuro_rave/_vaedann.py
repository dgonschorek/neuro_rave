"""Networks, objective and training loop of VAEDANN, the notebook-derived variational baseline.

Kept separate from ``_networks.py`` / ``_training.py`` so RAVE and RAVE+ are untouched.

Per minibatch, one Adam step (with weight decay) over all parameters minimizes

    reconstruction_mse + kl_weight * kl
    + class_weight * sum(class_ce per head) + domain_weight * domain_ce,

where the domain classifier sees either the sampled latent (``adversary_input="sample"``, as in
the source notebook) or the posterior mean (``"mean"``) through a gradient-reversal layer: its own
weights descend on ``domain_ce`` while the encoder receives ``-alpha * domain_weight`` times that
gradient. ``alpha = 2 / (1 + exp(-ramp_rate * epoch / epochs)) - 1`` ramps from 0 towards 1 and is
constant within an epoch. All terms are plain means over the batch (the KL term over batch and
latent dimensions; each class term over the batch's samples labelled for that head). The decoder
sees the sampled latent, optionally concatenated with a one-hot code of the sample's domain
(``decoder_conditioning="domain"``).

Several class heads are supported privately so the source notebook's two-head configuration
(separate label spaces) can run on this same code; the public ``VAEDANN`` uses at most
one head.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch import Tensor, nn

MAX_ALPHA = 1.0
ADVERSARY_INPUTS = ("sample", "mean")
DECODER_CONDITIONINGS = ("none", "domain")


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.alpha * grad_output, None


# ``Function.apply`` is an unannotated method in older PyTorch and a typed builtin in newer
# releases; binding it to an explicitly typed name keeps the call checked under both.
_apply_reverse_gradient: Callable[[Tensor, float], Tensor] = _ReverseGradient.apply


def reverse_gradient(x: Tensor, alpha: float) -> Tensor:
    """Identity forward; multiplies the gradient by ``-alpha`` backward."""
    return _apply_reverse_gradient(x, alpha)


class VAEDANNNetworks(nn.Module):
    """Encoder (mean and log-variance), decoder, class heads and domain classifier.

    Submodules are created in the source notebook's order (encoder, posterior layers, decoder,
    class heads, domain classifier), so the same seed gives the notebook model's initial weights.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        hidden_dim: int,
        n_domains: int,
        head_sizes: Sequence[int] = (),
        dropout: float = 0.2,
        decoder_domains: int = 0,
    ) -> None:
        super().__init__()
        self.decoder_domains = decoder_domains
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + decoder_domains, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )
        self.class_heads = nn.ModuleList(
            nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, k))
            for k in head_sizes
        )
        self.domain_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_domains),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mu: Tensor = self.enc_mu(h)
        logvar: Tensor = self.enc_logvar(h)
        return mu, logvar

    def decode(self, z: Tensor, domains: Tensor | None = None) -> Tensor:
        """Decode latents; ``domains`` (integer codes) is required if the decoder is conditioned."""
        if self.decoder_domains:
            if domains is None:
                raise ValueError("this decoder is conditioned on the domain; pass domain codes")
            z = torch.cat([z, F.one_hot(domains, self.decoder_domains).to(z.dtype)], dim=1)
        out: Tensor = self.decoder(z)
        return out


@dataclass(frozen=True)
class VAEDANNSettings:
    epochs: int
    per_domain: int
    learning_rate: float
    weight_decay: float
    domain_weight: float
    kl_weight: float
    class_weight: float
    device: torch.device
    adversary_input: str = "sample"
    ramp_rate: float = 5.0
    discriminator_steps: int = 0


def ramp(epoch: int, epochs: int, rate: float = 5.0) -> float:
    """Gradient-reversal strength for an epoch (0 in the first epoch, approaching ``MAX_ALPHA``)."""
    return MAX_ALPHA * (2.0 / (1.0 + math.exp(-rate * epoch / epochs)) - 1.0)


def balanced_batches(
    domain_codes: NDArray[np.int64], per_domain: int, rng: np.random.Generator
) -> Iterator[NDArray[np.int64]]:
    """One epoch: each batch holds ``per_domain`` shuffled samples of every domain.

    The epoch ends when the smallest domain is used up; incomplete batches are dropped, so larger
    domains are subsampled differently in every epoch.
    """
    pools = [rng.permutation(np.flatnonzero(domain_codes == k)) for k in np.unique(domain_codes)]
    n_batches = min(len(p) for p in pools) // per_domain
    for b in range(n_batches):
        yield np.concatenate([p[b * per_domain : (b + 1) * per_domain] for p in pools])


@dataclass(frozen=True)
class StepTerms:
    loss: Tensor
    reconstruction: Tensor
    kl: Tensor
    classification: Tensor
    domain: Tensor
    domain_logits: Tensor
    adversary_input: Tensor  # detached input of the domain classifier


def objective(
    nets: VAEDANNNetworks,
    x: Tensor,
    domains: Tensor,
    labels: Tensor,
    alpha: float,
    settings: VAEDANNSettings,
) -> StepTerms:
    """Loss of one minibatch. ``labels`` is (batch, n_heads) with -1 where a head has no label.

    Uses torch's global generator for dropout and latent noise, in the source notebook's order
    (encoder dropout, latent noise, domain-classifier dropout).
    """
    mu, logvar = nets.encode(x)
    z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    reconstruction = F.mse_loss(nets.decode(z, domains), x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    adversary_in = z if settings.adversary_input == "sample" else mu
    domain_logits = nets.domain_head(reverse_gradient(adversary_in, alpha))
    domain = F.cross_entropy(domain_logits, domains)
    classification = torch.zeros((), dtype=x.dtype, device=x.device)
    for h, head in enumerate(nets.class_heads):
        known = labels[:, h] >= 0
        if bool(known.any()):
            # A head without labelled samples stays out of the graph (.grad None): Adam skips it.
            classification = classification + F.cross_entropy(head(z[known]), labels[known, h])
    loss = (
        reconstruction
        + settings.kl_weight * kl
        + settings.class_weight * classification
        + settings.domain_weight * domain
    )
    return StepTerms(
        loss, reconstruction, kl, classification, domain, domain_logits, adversary_in.detach()
    )


HISTORY_KEYS = ("reconstruction", "kl", "classification", "domain", "domain_accuracy", "alpha")
EpochCallback = Callable[
    [int, VAEDANNNetworks, torch.optim.Optimizer, dict[str, list[float]]], None
]


def make_optimizer(nets: VAEDANNNetworks, settings: VAEDANNSettings) -> torch.optim.Adam:
    return torch.optim.Adam(
        nets.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )


def train(
    nets: VAEDANNNetworks,
    x: Tensor,
    domains: Tensor,
    labels: Tensor,
    settings: VAEDANNSettings,
    rng: np.random.Generator,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    start_epoch: int = 0,
    history: dict[str, list[float]] | None = None,
    on_epoch_end: EpochCallback | None = None,
) -> dict[str, list[float]]:
    """Train from ``start_epoch`` to ``settings.epochs``; return per-epoch means over batches.

    ``labels`` is (n, n_heads). ``on_epoch_end(epoch, nets, optimizer, history)`` runs after every
    epoch (``epoch`` counts completed epochs). Raises ``FloatingPointError`` as soon as a loss
    becomes non-finite (no batch is skipped).
    """
    device = settings.device
    opt = optimizer if optimizer is not None else make_optimizer(nets, settings)
    hist = history if history is not None else {k: [] for k in HISTORY_KEYS}
    codes = domains.numpy()
    nets.train()
    for epoch in range(start_epoch, settings.epochs):
        alpha = ramp(epoch, settings.epochs, settings.ramp_rate)
        totals = dict.fromkeys(HISTORY_KEYS[:5], 0.0)
        n_batches = 0
        for idx_np in balanced_batches(codes, settings.per_domain, rng):
            idx = torch.from_numpy(idx_np)
            xb, db, yb = x[idx].to(device), domains[idx].to(device), labels[idx].to(device)
            t = objective(nets, xb, db, yb, alpha, settings)
            if not bool(torch.isfinite(t.loss)):
                raise FloatingPointError(
                    f"VAEDANN training diverged: non-finite loss in epoch {epoch + 1} "
                    f"(reconstruction {t.reconstruction.item():.4g}, KL {t.kl.item():.4g}, "
                    f"domain {t.domain.item():.4g}). Try a smaller domain_weight or learning_rate."
                )
            opt.zero_grad(set_to_none=True)
            torch.autograd.backward(t.loss)
            opt.step()
            for _ in range(settings.discriminator_steps if settings.domain_weight > 0 else 0):
                # Extra domain-classifier-only updates on this batch's (detached) latents, so the
                # adversary keeps up with the encoder. Other parameters have no gradient, so Adam
                # leaves them (and their weight decay) untouched.
                opt.zero_grad(set_to_none=True)
                extra = settings.domain_weight * F.cross_entropy(
                    nets.domain_head(t.adversary_input), db
                )
                torch.autograd.backward(extra)
                opt.step()
            totals["reconstruction"] += t.reconstruction.item()
            totals["kl"] += t.kl.item()
            totals["classification"] += t.classification.item()
            totals["domain"] += t.domain.item()
            totals["domain_accuracy"] += (t.domain_logits.argmax(1) == db).float().mean().item()
            n_batches += 1
        for k, total in totals.items():
            hist[k].append(total / n_batches)
        hist["alpha"].append(alpha)
        if on_epoch_end is not None:
            nets.eval()
            on_epoch_end(epoch + 1, nets, opt, hist)
            nets.train()
    nets.eval()
    return hist


NOTEBOOK_KEY_MAP = (
    ("type_head_rgc.", "class_heads.0."),
    ("type_head_sc.", "class_heads.1."),
    ("domain_head.net.", "domain_head."),
)


def from_notebook_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename the notebook's ``VAE_DANN`` parameters to ``VAEDANNNetworks`` names (none dropped)."""
    out = {}
    for key, value in state.items():
        new = key
        for old_prefix, new_prefix in NOTEBOOK_KEY_MAP:
            if key.startswith(old_prefix):
                new = new_prefix + key[len(old_prefix) :]
        out[new] = value
    return out


CHECKPOINT_FORMAT = "neuro_rave.VAEDANN/1"


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` with ``torch.save`` atomically (temporary file, then rename)."""
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` (``weights_only=True``)."""
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a VAEDANN checkpoint (format {payload.get('format')!r}).")
    return payload


def rng_state(rng: np.random.Generator, device: torch.device) -> dict[str, Any]:
    """Random states needed to continue training identically (torch, CUDA if used, batch order)."""
    state: dict[str, Any] = {"torch": torch.get_rng_state(), "numpy": rng.bit_generator.state}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(
    state: dict[str, Any], rng: np.random.Generator, device: torch.device
) -> None:
    torch.set_rng_state(state["torch"])
    rng.bit_generator.state = state["numpy"]
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)
