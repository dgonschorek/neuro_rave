"""The public estimators: RAVE (unsupervised) and RAVE+ (partially labeled)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from neuro_rave._networks import RAVENetworks
from neuro_rave._training import Batch, TrainingSettings, train
from neuro_rave._validation import (
    EncodedCategories,
    as_windows,
    check_layout,
    encode_categories,
)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def _finite_float(name: str, value: float, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number; got {value!r}.")
    number = float(value)
    bound_ok = number > 0 if positive else number >= 0
    if not math.isfinite(number) or not bound_ok:
        requirement = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite, {requirement} number; got {value!r}.")
    return number


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device={device!r} was requested, but CUDA is not available.")
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda' (optionally 'cuda:N'); got {device!r}.")
    return resolved


class _RAVEBase:
    _class_weight: float = 0.0

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        hidden_dims: Sequence[int] = (256, 128),
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        domain_weight: float = 0.1,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.hidden_dims = tuple(_positive_int("hidden_dims entry", h) for h in hidden_dims)
        self.epochs = _positive_int("epochs", epochs)
        self.batch_size = _positive_int("batch_size", batch_size)
        self.learning_rate = _finite_float("learning_rate", learning_rate, positive=True)
        self.domain_weight = _finite_float("domain_weight", domain_weight, positive=False)
        self.device = _resolve_device(device)
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0
        ):
            raise ValueError(f"seed must be None or a non-negative integer; got {seed!r}.")
        self.seed = None if seed is None else int(seed)

        self._networks: RAVENetworks | None = None
        self.input_shape_: tuple[int, ...] | None = None
        self.domains_: NDArray[Any] | None = None
        self.history_: dict[str, list[float]] | None = None

    def _fit(self, X: ArrayLike, domains: ArrayLike, labels: EncodedCategories | None) -> None:
        windows = as_windows(X)
        domain_codes = encode_categories(
            domains, name="domains", n_rows=windows.shape[0], allow_missing=False
        )
        n_rows = windows.shape[0]
        if labels is None:
            label_codes = np.full(n_rows, -1, dtype=np.int64)
            known = np.zeros(n_rows, dtype=bool)
            n_classes = None
        else:
            label_codes, known, n_classes = labels.codes, labels.known, len(labels.categories)

        generator = torch.Generator()
        if self.seed is None:
            generator.seed()
        else:
            generator.manual_seed(self.seed)
        n_features = int(np.prod(windows.shape[1:]))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(2**62, (1,), generator=generator)))
            nets = RAVENetworks(
                n_features=n_features,
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                n_domains=len(domain_codes.categories),
                n_classes=n_classes,
            )
        nets.to(self.device)

        data = Batch(
            x=torch.from_numpy(windows.reshape(n_rows, n_features)),
            domains=torch.from_numpy(domain_codes.codes),
            labels=torch.from_numpy(label_codes),
            known=torch.from_numpy(known),
        )
        settings = TrainingSettings(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            domain_weight=self.domain_weight,
            class_weight=self._class_weight,
            device=self.device,
        )
        history = train(nets, data, settings, generator)

        self._networks = nets
        self.input_shape_ = tuple(int(d) for d in windows.shape[1:])
        self.domains_ = domain_codes.categories
        self.history_ = history

    def _fitted(self) -> tuple[RAVENetworks, tuple[int, ...]]:
        if self._networks is None or self.input_shape_ is None:
            raise RuntimeError(
                f"This {type(self).__name__} instance is not fitted yet; call fit() first."
            )
        return self._networks, self.input_shape_

    def _flat_inputs(self, X: ArrayLike) -> NDArray[np.float32]:
        _, input_shape = self._fitted()
        windows = as_windows(X)
        check_layout(windows, input_shape)
        return windows.reshape(windows.shape[0], -1)

    def _map_batches(
        self, flat: NDArray[np.float32], forward: torch.nn.Module
    ) -> NDArray[np.float32]:
        outputs = []
        with torch.no_grad():
            for start in range(0, flat.shape[0], self.batch_size):
                chunk = torch.from_numpy(flat[start : start + self.batch_size]).to(self.device)
                outputs.append(forward(chunk).cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def transform(self, X: ArrayLike) -> NDArray[np.float32]:
        """Embed windows shaped like the training windows; returns (n_samples, latent_dim)."""
        nets, _ = self._fitted()
        return self._map_batches(self._flat_inputs(X), nets.encoder)

    def reconstruct(self, X: ArrayLike) -> NDArray[np.float32]:
        """Decode each window's embedding back to input space; returns X's shape."""
        nets, input_shape = self._fitted()
        flat = self._flat_inputs(X)
        autoencoder = torch.nn.Sequential(nets.encoder, nets.decoder)
        return self._map_batches(flat, autoencoder).reshape(flat.shape[0], *input_shape)


class RAVE(_RAVEBase):
    """Remove domain (e.g. session) information from time-series windows, without class labels.

    Trains an MLP encoder and decoder against an adversarial domain discriminator
    (Gonschorek et al., NeurIPS 2021). Computation is float32; ``seed`` makes CPU
    runs reproducible, including ``seed=0``.

    Args:
        latent_dim: Size of the embedding.
        hidden_dims: Encoder hidden widths; the decoder mirrors them.
        epochs: Passes over the training data.
        batch_size: Minibatch size; a final minibatch of one row is fine.
        learning_rate: Adam learning rate for both optimizers.
        domain_weight: Weight of the adversarial domain term (0 gives a plain autoencoder).
            The adversarial term is unbounded, so large weights can make reconstruction
            collapse or diverge; check ``reconstruct()`` error after fitting.
        device: ``"cpu"`` (default) or ``"cuda"``.
        seed: Seed for initialization and shuffling; ``None`` is non-deterministic.
    """

    def fit(self, X: ArrayLike, domains: ArrayLike) -> RAVE:
        """Train from scratch on windows ``X`` and one domain ID per row (at least 2 domains).

        ``X`` is (samples, time) or (samples, time, channels), already windowed,
        aligned, and scaled by the caller.
        """
        self._fit(X, domains, labels=None)
        return self


class RAVEPlus(_RAVEBase):
    """RAVE plus a class head on the embedding, trained on whichever rows have labels (RAVE+).

    Every row contributes to reconstruction and domain training; only rows with a
    known label contribute to the class loss. Takes the same arguments as
    :class:`RAVE`, plus ``class_weight`` for the class cross-entropy term.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        hidden_dims: Sequence[int] = (256, 128),
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        domain_weight: float = 0.1,
        class_weight: float = 1.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            domain_weight=domain_weight,
            device=device,
            seed=seed,
        )
        self.class_weight = self._class_weight = _finite_float(
            "class_weight", class_weight, positive=False
        )
        self.classes_: NDArray[Any] | None = None

    def fit(self, X: ArrayLike, domains: ArrayLike, y: ArrayLike) -> RAVEPlus:
        """Train from scratch; ``y`` has one label per row, ``None`` where unknown.

        Labels may be known in any domain; at least two distinct classes must be observed.
        """
        windows = as_windows(X)
        labels = encode_categories(y, name="y", n_rows=windows.shape[0], allow_missing=True)
        self._fit(windows, domains, labels)
        self.classes_ = labels.categories
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float32]:
        """Class probabilities, shape (n_samples, n_classes); columns follow ``classes_``."""
        nets, _ = self._fitted()
        assert nets.classifier is not None
        head = torch.nn.Sequential(nets.encoder, nets.classifier, torch.nn.Softmax(dim=1))
        return self._map_batches(self._flat_inputs(X), head)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Most probable class per row, as the original labels from ``y``."""
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]
