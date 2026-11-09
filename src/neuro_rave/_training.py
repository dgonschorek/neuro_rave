"""The shared alternating-update training loop for RAVE and RAVE+.

Per minibatch, in this order (paper Sec. 4.2.3; upstream ``train_model``):

1. Model step (Adam over encoder, decoder, and class head). The discriminator's
   parameters are frozen but stay in the graph, so the encoder receives the
   adversarial gradient through it. Minimizes
   ``reconstruction - domain_weight * domain_ce + class_weight * class_ce``.
2. Discriminator step (separate Adam over the discriminator only). Minimizes
   ``+domain_ce`` on embeddings recomputed with the just-updated encoder and
   detached from it.

Every term is a group-balanced mean: reconstruction and domain cross-entropy
weight each domain present in the batch equally, class cross-entropy weights
each class present among the batch's labeled rows equally.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from neuro_rave._networks import RAVENetworks

ADAM_BETAS = (0.5, 0.999)


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    batch_size: int
    learning_rate: float
    domain_weight: float
    class_weight: float
    device: torch.device


@dataclass(frozen=True)
class LossTerms:
    reconstruction: Tensor
    domain: Tensor
    classification: Tensor


@dataclass(frozen=True)
class Batch:
    """``labels`` is ``-1`` wherever ``known`` is False; those rows never reach the class loss."""

    x: Tensor
    domains: Tensor
    labels: Tensor
    known: Tensor


def group_balanced_mean(values: Tensor, groups: Tensor) -> Tensor:
    """Mean over the groups present of each group's mean value; zero when ``values`` is empty."""
    if values.numel() == 0:
        return values.sum()
    present, inverse = torch.unique(groups, return_inverse=True)
    sums = torch.zeros(len(present), dtype=values.dtype, device=values.device)
    sums = sums.index_add(0, inverse, values)
    counts = torch.bincount(inverse, minlength=len(present)).to(values.dtype)
    return (sums / counts).mean()


def model_loss_terms(nets: RAVENetworks, batch: Batch) -> LossTerms:
    z = nets.encoder(batch.x)
    per_row_mse = (nets.decoder(z) - batch.x).pow(2).mean(dim=1)
    reconstruction = group_balanced_mean(per_row_mse, batch.domains)
    domain_ce = F.cross_entropy(nets.discriminator(z), batch.domains, reduction="none")
    domain = group_balanced_mean(domain_ce, batch.domains)

    if nets.classifier is None or not bool(batch.known.any()):
        # Skipping the head entirely (rather than a zero-weighted term) leaves its
        # .grad as None, so Adam's momentum does not move it on unlabeled batches.
        classification = torch.zeros((), dtype=batch.x.dtype, device=batch.x.device)
    else:
        labels = batch.labels[batch.known]
        class_ce = F.cross_entropy(nets.classifier(z[batch.known]), labels, reduction="none")
        classification = group_balanced_mean(class_ce, labels)
    return LossTerms(reconstruction, domain, classification)


def model_objective(terms: LossTerms, domain_weight: float, class_weight: float) -> Tensor:
    return terms.reconstruction - domain_weight * terms.domain + class_weight * terms.classification


def discriminator_loss(nets: RAVENetworks, batch: Batch) -> Tensor:
    with torch.no_grad():
        z = nets.encoder(batch.x)
    domain_ce = F.cross_entropy(nets.discriminator(z), batch.domains, reduction="none")
    return group_balanced_mean(domain_ce, batch.domains)


def make_optimizers(
    nets: RAVENetworks, learning_rate: float
) -> tuple[torch.optim.Adam, torch.optim.Adam]:
    """Return (model optimizer, discriminator optimizer) over disjoint parameter sets."""
    model_modules: list[nn.Module] = [nets.encoder, nets.decoder]
    if nets.classifier is not None:
        model_modules.append(nets.classifier)
    model_params = [p for module in model_modules for p in module.parameters()]
    model_opt = torch.optim.Adam(model_params, lr=learning_rate, betas=ADAM_BETAS)
    disc_opt = torch.optim.Adam(nets.discriminator.parameters(), lr=learning_rate, betas=ADAM_BETAS)
    return model_opt, disc_opt


def model_step(
    nets: RAVENetworks,
    optimizer: torch.optim.Optimizer,
    batch: Batch,
    domain_weight: float,
    class_weight: float,
) -> LossTerms:
    nets.discriminator.requires_grad_(False)
    try:
        optimizer.zero_grad(set_to_none=True)
        terms = model_loss_terms(nets, batch)
        torch.autograd.backward(model_objective(terms, domain_weight, class_weight))
        optimizer.step()
    finally:
        nets.discriminator.requires_grad_(True)
    return LossTerms(
        terms.reconstruction.detach(), terms.domain.detach(), terms.classification.detach()
    )


def discriminator_step(
    nets: RAVENetworks, optimizer: torch.optim.Optimizer, batch: Batch
) -> Tensor:
    optimizer.zero_grad(set_to_none=True)
    loss = discriminator_loss(nets, batch)
    torch.autograd.backward(loss)
    optimizer.step()
    return loss.detach()


def iterate_minibatches(
    data: Batch, batch_size: int, generator: torch.Generator, device: torch.device
) -> Iterator[Batch]:
    """Shuffled minibatches covering every row once; the last one may hold a single row."""
    order = torch.randperm(data.x.shape[0], generator=generator)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield Batch(
            x=data.x[idx].to(device),
            domains=data.domains[idx].to(device),
            labels=data.labels[idx].to(device),
            known=data.known[idx].to(device),
        )


def train(
    nets: RAVENetworks,
    data: Batch,
    settings: TrainingSettings,
    generator: torch.Generator,
) -> dict[str, list[float]]:
    """Run one fresh training run; return per-epoch, row-weighted mean losses."""
    model_opt, disc_opt = make_optimizers(nets, settings.learning_rate)
    history: dict[str, list[float]] = {
        "reconstruction": [],
        "domain": [],
        "classification": [],
        "discriminator": [],
    }
    nets.train()
    for _ in range(settings.epochs):
        totals = dict.fromkeys(history, 0.0)
        for batch in iterate_minibatches(data, settings.batch_size, generator, settings.device):
            terms = model_step(
                nets, model_opt, batch, settings.domain_weight, settings.class_weight
            )
            disc = discriminator_step(nets, disc_opt, batch)
            n = batch.x.shape[0]
            totals["reconstruction"] += n * terms.reconstruction.item()
            totals["domain"] += n * terms.domain.item()
            totals["classification"] += n * terms.classification.item()
            totals["discriminator"] += n * disc.item()
        n_rows = data.x.shape[0]
        for key, total in totals.items():
            history[key].append(total / n_rows)
    nets.eval()
    return history
