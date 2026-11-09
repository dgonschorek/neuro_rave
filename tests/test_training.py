"""Loss reductions, label masking, optimizer isolation, and the adversarial gradient sign."""

import pytest
import torch

from neuro_rave._networks import RAVENetworks
from neuro_rave._training import (
    Batch,
    discriminator_step,
    group_balanced_mean,
    make_optimizers,
    model_loss_terms,
    model_objective,
    model_step,
)


def _nets(n_classes=3, seed=0):
    torch.manual_seed(seed)
    return RAVENetworks(
        n_features=12, latent_dim=4, hidden_dims=(16,), n_domains=2, n_classes=n_classes
    )


def _batch(n=8, labeled=True, seed=0):
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(3, (n,), generator=gen) if labeled else torch.full((n,), -1)
    return Batch(
        x=torch.randn(n, 12, generator=gen),
        domains=torch.arange(n) % 2,
        labels=labels,
        known=labels >= 0,
    )


def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _unchanged(module, snapshot):
    return all(torch.equal(p, s) for p, s in zip(module.parameters(), snapshot, strict=True))


def test_group_balanced_mean_weights_groups_equally():
    values = torch.tensor([1.0, 1.0, 1.0, 4.0])
    groups = torch.tensor([0, 0, 0, 5])
    assert group_balanced_mean(values, groups).item() == pytest.approx(2.5)


def test_group_balanced_mean_of_nothing_is_zero():
    assert group_balanced_mean(torch.zeros(0), torch.zeros(0, dtype=torch.long)).item() == 0.0


def test_unlabeled_rows_never_reach_the_class_loss():
    nets, batch = _nets(), _batch()
    partial = Batch(
        x=batch.x,
        domains=batch.domains,
        labels=torch.where(torch.arange(8) < 4, batch.labels, torch.full((8,), -1)),
        known=torch.arange(8) < 4,
    )
    labeled_only = Batch(
        x=batch.x[:4], domains=batch.domains[:4], labels=batch.labels[:4], known=batch.known[:4]
    )
    assert torch.allclose(
        model_loss_terms(nets, partial).classification,
        model_loss_terms(nets, labeled_only).classification,
    )


def test_batch_with_zero_labeled_rows_gives_zero_class_loss_and_no_head_gradient():
    nets, batch = _nets(), _batch(labeled=False)
    model_opt, _ = make_optimizers(nets, 1e-2)
    head_before = _snapshot(nets.classifier)
    terms = model_step(nets, model_opt, batch, domain_weight=1.0, class_weight=1.0)
    assert terms.classification.item() == 0.0
    assert torch.isfinite(terms.reconstruction) and torch.isfinite(terms.domain)
    assert all(p.grad is None for p in nets.classifier.parameters())
    assert _unchanged(nets.classifier, head_before)


def test_singleton_batch_trains():
    nets, batch = _nets(), _batch(n=1)
    model_opt, disc_opt = make_optimizers(nets, 1e-2)
    terms = model_step(nets, model_opt, batch, 1.0, 1.0)
    disc = discriminator_step(nets, disc_opt, batch)
    assert all(torch.isfinite(t) for t in (terms.reconstruction, terms.domain, disc))


def test_optimizers_own_disjoint_parameter_sets_covering_everything():
    nets = _nets()
    model_opt, disc_opt = make_optimizers(nets, 1e-3)
    model_ids = {id(p) for g in model_opt.param_groups for p in g["params"]}
    disc_ids = {id(p) for g in disc_opt.param_groups for p in g["params"]}
    assert disc_ids == {id(p) for p in nets.discriminator.parameters()}
    assert model_ids.isdisjoint(disc_ids)
    assert model_ids | disc_ids == {id(p) for p in nets.parameters()}


def test_model_step_leaves_discriminator_untouched_and_unfrozen():
    nets, batch = _nets(), _batch()
    model_opt, _ = make_optimizers(nets, 1e-2)
    disc_before = _snapshot(nets.discriminator)
    model_step(nets, model_opt, batch, 1.0, 1.0)
    assert _unchanged(nets.discriminator, disc_before)
    assert all(p.grad is None for p in nets.discriminator.parameters())
    assert all(p.requires_grad for p in nets.discriminator.parameters())


def test_discriminator_step_leaves_model_untouched():
    nets, batch = _nets(), _batch()
    _, disc_opt = make_optimizers(nets, 1e-2)
    model_parts = [nets.encoder, nets.decoder, nets.classifier]
    before = [_snapshot(m) for m in model_parts]
    disc_before = _snapshot(nets.discriminator)
    discriminator_step(nets, disc_opt, batch)
    assert all(_unchanged(m, s) for m, s in zip(model_parts, before, strict=True))
    assert all(p.grad is None for p in nets.encoder.parameters())
    assert not _unchanged(nets.discriminator, disc_before)


def test_discriminator_step_descends_domain_ce():
    nets, batch = _nets(), _batch(n=32)
    _, disc_opt = make_optimizers(nets, 1e-2)
    losses = [discriminator_step(nets, disc_opt, batch).item() for _ in range(50)]
    assert losses[-1] < losses[0]


def _encoder_grads(loss, nets):
    return torch.autograd.grad(loss, list(nets.encoder.parameters()), retain_graph=True)


def test_adversarial_term_enters_encoder_gradient_with_negative_sign():
    nets, batch = _nets(), _batch()
    nets.discriminator.requires_grad_(False)
    terms = model_loss_terms(nets, batch)
    domain_weight = 0.7

    grad_domain = _encoder_grads(terms.domain, nets)
    grad_rest = _encoder_grads(terms.reconstruction + 2.0 * terms.classification, nets)
    grad_objective = _encoder_grads(model_objective(terms, domain_weight, 2.0), nets)

    assert any(g.abs().sum() > 0 for g in grad_domain), "no gradient path through the frozen head"
    for g_obj, g_rest, g_dom in zip(grad_objective, grad_rest, grad_domain, strict=True):
        torch.testing.assert_close(g_obj - g_rest, -domain_weight * g_dom)


def test_descending_the_adversarial_term_alone_raises_frozen_discriminator_ce():
    nets, batch = _nets(), _batch(n=32)
    nets.discriminator.requires_grad_(False)
    domain_weight = 1.0
    ce_before = model_loss_terms(nets, batch).domain
    adversarial_only = -domain_weight * ce_before
    grads = _encoder_grads(adversarial_only, nets)
    with torch.no_grad():
        for p, g in zip(nets.encoder.parameters(), grads, strict=True):
            p -= 1e-3 * g
    ce_after = model_loss_terms(nets, batch).domain
    assert ce_after.item() > ce_before.item()
