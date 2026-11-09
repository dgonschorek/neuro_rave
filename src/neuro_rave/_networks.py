"""The small MLP encoder, decoder, domain discriminator, and optional class head."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

HEAD_HIDDEN_DIM = 64


def mlp(dims: Sequence[int]) -> nn.Sequential:
    """Linear layers through ``dims`` with ReLU between them and a linear output."""
    layers: list[nn.Module] = []
    for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        if i > 0:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n_in, n_out))
    return nn.Sequential(*layers)


class RAVENetworks(nn.Module):
    """All trainable parts of one fitted model.

    No BatchNorm or dropout, so a minibatch of one sample trains correctly and
    inference is identical in train and eval mode.
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        n_domains: int,
        n_classes: int | None,
    ) -> None:
        super().__init__()
        self.encoder = mlp([n_features, *hidden_dims, latent_dim])
        self.decoder = mlp([latent_dim, *reversed(hidden_dims), n_features])
        self.discriminator = mlp([latent_dim, HEAD_HIDDEN_DIM, n_domains])
        self.classifier: nn.Sequential | None = (
            None if n_classes is None else mlp([latent_dim, HEAD_HIDDEN_DIM, n_classes])
        )
