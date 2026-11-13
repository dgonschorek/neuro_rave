"""The public estimators: RAVE, RAVE+ (partially labeled) and the experimental VAEDANN."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from neuro_rave._networks import RAVENetworks
from neuro_rave._training import Batch, TrainingSettings, train
from neuro_rave._vaedann import (
    ADVERSARY_INPUTS,
    CHECKPOINT_FORMAT,
    DECODER_CONDITIONINGS,
    VAEDANNNetworks,
    VAEDANNSettings,
    load_checkpoint,
    make_optimizer,
    restore_rng_state,
    rng_state,
    save_checkpoint,
)
from neuro_rave._vaedann import train as train_vaedann
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
        """Embed time series shaped like the training data; returns (n_samples, latent_dim)."""
        nets, _ = self._fitted()
        return self._map_batches(self._flat_inputs(X), nets.encoder)

    def reconstruct(self, X: ArrayLike) -> NDArray[np.float32]:
        """Decode each sample's embedding back to input space; returns X's shape."""
        nets, input_shape = self._fitted()
        flat = self._flat_inputs(X)
        autoencoder = torch.nn.Sequential(nets.encoder, nets.decoder)
        return self._map_batches(flat, autoencoder).reshape(flat.shape[0], *input_shape)


class RAVE(_RAVEBase):
    """Remove domain (e.g. session) information from time-series data, without class labels.

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
        """Train from scratch on time series ``X`` and one domain ID per row (at least 2 domains).

        ``X`` is (samples, time) or (samples, time, channels), already cut to a common
        length, aligned, and scaled by the caller.
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
        labels: NDArray[Any] = self.classes_[np.argmax(proba, axis=1)]
        return labels


def _data_fingerprint(
    windows: NDArray[np.float32],
    domain_codes: NDArray[np.int64],
    label_codes: NDArray[np.int64],
    categories: list[Any],
) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(windows).tobytes())
    h.update(np.ascontiguousarray(domain_codes).tobytes())
    h.update(np.ascontiguousarray(label_codes).tobytes())
    h.update(repr(categories).encode())
    return h.hexdigest()


def _plain(values: NDArray[Any] | None) -> list[Any] | None:
    """Categories as plain Python values (safe for ``torch.load(weights_only=True)``)."""
    return (
        None if values is None else [v.item() if isinstance(v, np.generic) else v for v in values]
    )


class VAEDANN:
    """Experimental baseline: variational autoencoder with a gradient-reversal domain adversary.

    Derived from an RGC-to-SC analysis notebook (VAE plus domain-adversarial training, "DANN").
    The encoder outputs a Gaussian posterior per sample. Training decodes a sampled latent, adds a
    KL term towards a standard normal prior, and trains a domain classifier through a
    gradient-reversal layer whose strength ramps up over training. With ``y``, a class head on the
    latent is trained on the samples whose label is known. One Adam optimizer (with weight decay)
    updates all parts in a single step per minibatch.

    Unlike :class:`RAVE`, every minibatch holds the same number of samples from each domain
    (``batch_size // n_domains``, at most the smallest domain's size). An epoch ends when the
    smallest domain is used up, so larger domains are subsampled anew in every epoch.

    ``transform``, ``reconstruct``, ``predict`` and ``predict_proba`` use the posterior mean, so
    they are deterministic. (The source notebook decoded a sampled latent at evaluation.)

    Tested on few datasets: see the README for tested settings and where the adversary reduced
    domain predictability. Always compare with a ``domain_weight=0`` fit
    (see ``neuro_rave.evaluation``).

    Args:
        latent_dim: Size of the latent (embedding).
        hidden_dim: Width of the hidden layers of encoder, decoder and domain classifier.
        epochs: Training epochs. ``None`` (default) chooses the number of epochs that presents as
            many training samples as 100 passes over all samples; the value used is in ``epochs_``.
            The source notebook's tested setting is 1000.
        batch_size: Samples per minibatch, split equally across domains.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay (L2), applied to all parameters.
        domain_weight: Weight of the adversarial domain term (0 disables the adversary).
        kl_weight: Weight of the KL term.
        class_weight: Weight of the class cross-entropy (used only when fitted with ``y``).
        dropout: Dropout probability in the encoder and the domain classifier (training only).
        adversary_input: What the domain classifier sees during training: ``"sample"`` (a sampled
            latent, as in the source notebook) or ``"mean"`` (the posterior mean, i.e. the
            representation ``transform`` returns).
        decoder_conditioning: ``"none"`` (decoder sees only the latent) or ``"domain"`` (decoder
            also sees a one-hot code of the sample's domain; ``reconstruct`` then needs domains).
        ramp_rate: Speed of the gradient-reversal ramp
            ``2 / (1 + exp(-ramp_rate * epoch / epochs)) - 1``; smaller values keep the adversary
            weak for longer.
        device: ``"cpu"`` (default) or ``"cuda"``.
        seed: Seeds initialization, dropout, latent sampling and batch order without changing the
            caller's global random state. ``None`` is non-deterministic. CPU runs with the same
            seed and PyTorch version are reproducible; GPU runs may not be bit-identical.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 8,
        hidden_dim: int = 128,
        epochs: int | None = None,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        domain_weight: float = 0.2,
        kl_weight: float = 1e-2,
        class_weight: float = 1.0,
        dropout: float = 0.2,
        adversary_input: str = "sample",
        decoder_conditioning: str = "none",
        ramp_rate: float = 5.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self.latent_dim = _positive_int("latent_dim", latent_dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.epochs = None if epochs is None else _positive_int("epochs", epochs)
        self.batch_size = _positive_int("batch_size", batch_size)
        self.learning_rate = _finite_float("learning_rate", learning_rate, positive=True)
        self.weight_decay = _finite_float("weight_decay", weight_decay, positive=False)
        self.domain_weight = _finite_float("domain_weight", domain_weight, positive=False)
        self.kl_weight = _finite_float("kl_weight", kl_weight, positive=False)
        self.class_weight = _finite_float("class_weight", class_weight, positive=False)
        self.dropout = _finite_float("dropout", dropout, positive=False)
        if self.dropout >= 1:
            raise ValueError(f"dropout must be in [0, 1); got {dropout!r}.")
        if adversary_input not in ADVERSARY_INPUTS:
            raise ValueError(
                f"adversary_input must be one of {ADVERSARY_INPUTS}; got {adversary_input!r}."
            )
        if decoder_conditioning not in DECODER_CONDITIONINGS:
            raise ValueError(
                f"decoder_conditioning must be one of {DECODER_CONDITIONINGS}; "
                f"got {decoder_conditioning!r}."
            )
        self.adversary_input = adversary_input
        self.decoder_conditioning = decoder_conditioning
        self.ramp_rate = _finite_float("ramp_rate", ramp_rate, positive=True)
        self.device = _resolve_device(device)
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0
        ):
            raise ValueError(f"seed must be None or a non-negative integer; got {seed!r}.")
        self.seed = None if seed is None else int(seed)

        self._networks: VAEDANNNetworks | None = None
        self.input_shape_: tuple[int, ...] | None = None
        self.domains_: NDArray[Any] | None = None
        self.classes_: NDArray[Any] | None = None
        self.epochs_: int | None = None
        self.history_: dict[str, list[float]] | None = None
        self.config_: dict[str, Any] | None = None

    def get_params(self) -> dict[str, Any]:
        """Constructor arguments (device as a string); ``VAEDANN(**get_params())`` is unfitted."""
        params = {name: getattr(self, name) for name in _VAEDANN_PARAMS}
        params["device"] = str(self.device)
        return params

    def fit(
        self,
        X: ArrayLike,
        domains: ArrayLike,
        y: ArrayLike | None = None,
        *,
        checkpoint: str | os.PathLike[str] | None = None,
        checkpoint_every: int = 50,
        resume: bool = False,
    ) -> VAEDANN:
        """Train from scratch (or continue a checkpoint); returns the model.

        ``X`` is (samples, time) or (samples, time, channels), already aligned and scaled by the
        caller; ``domains`` has one ID per sample (at least 2 domains). ``y`` is optional: one label
        per sample with ``None`` where unknown (at least two classes observed). Without ``y`` there
        is no class head and ``predict`` / ``predict_proba`` raise.

        With ``checkpoint=path``, the full training state (weights, optimizer, epoch, random states,
        history, settings, category mappings and a fingerprint of the data) is written atomically
        every ``checkpoint_every`` epochs and at the end. An existing checkpoint is only continued
        with ``resume=True``, and only if the settings and the data (X, domains, y) are identical;
        otherwise ``fit`` refuses. A missing checkpoint with ``resume=True`` starts a fresh fit, so
        the same call works before and after an interruption. Continuing gives the same result as
        an uninterrupted fit on the same machine and PyTorch version.
        """
        windows = as_windows(X)
        n_rows = windows.shape[0]
        domain_codes = encode_categories(
            domains, name="domains", n_rows=n_rows, allow_missing=False
        )
        classes: NDArray[Any] | None = None
        label_codes = np.full(n_rows, -1, dtype=np.int64)
        if y is not None:
            labels = encode_categories(y, name="y", n_rows=n_rows, allow_missing=True)
            label_codes, classes = labels.codes, labels.categories
        n_domains = len(domain_codes.categories)
        if self.batch_size < n_domains:
            raise ValueError(
                f"batch_size={self.batch_size} is smaller than the number of domains "
                f"({n_domains}); every minibatch needs at least one sample per domain."
            )
        checkpoint_every = _positive_int("checkpoint_every", checkpoint_every)
        smallest = int(np.min(np.bincount(domain_codes.codes)))
        per_domain = min(self.batch_size // n_domains, smallest)
        presented_per_epoch = (smallest // per_domain) * per_domain * n_domains
        epochs = self.epochs
        if epochs is None:
            epochs = max(1, round(100 * n_rows / presented_per_epoch))
        config = self.get_params()
        del config["device"]
        fingerprint = _data_fingerprint(
            windows,
            domain_codes.codes,
            label_codes,
            [_plain(domain_codes.categories), _plain(classes)],
        )

        path = None if checkpoint is None else Path(checkpoint)
        saved: dict[str, Any] | None = None
        if resume and path is None:
            raise ValueError("resume=True needs checkpoint=<path>.")
        if path is not None and path.exists():
            if not resume:
                raise FileExistsError(
                    f"checkpoint {path} exists; pass resume=True to continue it, or delete it."
                )
            saved = load_checkpoint(path)
            if saved["config"] != config:
                diff = sorted(k for k in config if saved["config"].get(k) != config[k])
                raise ValueError(f"cannot resume {path}: settings differ ({', '.join(diff)}).")
            if saved["data_fingerprint"] != fingerprint:
                raise ValueError(
                    f"cannot resume {path}: X, domains or y differ from the checkpoint."
                )

        generator = torch.Generator()
        if self.seed is None:
            generator.seed()
        else:
            generator.manual_seed(self.seed)
        torch_seed = int(torch.randint(2**62, (1,), generator=generator))
        rng = np.random.default_rng(int(torch.randint(2**62, (1,), generator=generator)))
        n_features = int(np.prod(windows.shape[1:]))
        settings = VAEDANNSettings(
            epochs=epochs,
            per_domain=per_domain,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            domain_weight=self.domain_weight,
            kl_weight=self.kl_weight,
            class_weight=self.class_weight,
            device=self.device,
            adversary_input=self.adversary_input,
            ramp_rate=self.ramp_rate,
        )
        meta = {
            "format": CHECKPOINT_FORMAT,
            "config": config,
            "epochs_total": epochs,
            "data_fingerprint": fingerprint,
            "domains": _plain(domain_codes.categories),
            "classes": _plain(classes),
            "input_shape": [int(d) for d in windows.shape[1:]],
            "package_version": _package_version(),
        }

        def save(
            epoch: int,
            nets: VAEDANNNetworks,
            opt: torch.optim.Optimizer,
            hist: dict[str, list[float]],
        ) -> None:
            if path is not None and (epoch % checkpoint_every == 0 or epoch == epochs):
                payload = {
                    **meta,
                    "epoch": epoch,
                    "model": nets.state_dict(),
                    "optimizer": opt.state_dict(),
                    "rng": rng_state(rng, self.device),
                    "history": hist,
                }
                _write_checkpoint(path, payload)

        devices = [self.device] if self.device.type == "cuda" else []
        # Initialization, dropout and latent noise draw from torch's global generator; forking it
        # makes them reproducible without changing the caller's random state.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(torch_seed)
            nets = VAEDANNNetworks(
                n_features=n_features,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                n_domains=n_domains,
                head_sizes=() if classes is None else (len(classes),),
                dropout=self.dropout,
                decoder_domains=n_domains if self.decoder_conditioning == "domain" else 0,
            ).to(self.device)
            opt = make_optimizer(nets, settings)
            start, history = 0, None
            if saved is not None:
                nets.load_state_dict(saved["model"])
                opt.load_state_dict(saved["optimizer"])
                restore_rng_state(saved["rng"], rng, self.device)
                start, history = int(saved["epoch"]), saved["history"]
            history = train_vaedann(
                nets,
                torch.from_numpy(windows.reshape(n_rows, n_features)),
                torch.from_numpy(domain_codes.codes),
                torch.from_numpy(label_codes).reshape(n_rows, 1),
                settings,
                rng,
                optimizer=opt,
                start_epoch=start,
                history=history,
                on_epoch_end=save,
            )

        self._networks = nets
        self.input_shape_ = tuple(int(d) for d in windows.shape[1:])
        self.domains_ = domain_codes.categories
        self.classes_ = classes
        self.epochs_ = epochs
        self.history_ = history
        self.config_ = {**config, "epochs_resolved": epochs, "per_domain": per_domain}
        return self

    def _fitted(self) -> tuple[VAEDANNNetworks, tuple[int, ...]]:
        if self._networks is None or self.input_shape_ is None:
            raise RuntimeError("This VAEDANN instance is not fitted yet; call fit() first.")
        return self._networks, self.input_shape_

    def _posterior_mean(self, X: ArrayLike) -> torch.Tensor:
        nets, input_shape = self._fitted()
        windows = as_windows(X)
        check_layout(windows, input_shape)
        flat = torch.from_numpy(windows.reshape(windows.shape[0], -1))
        means = []
        with torch.no_grad():
            for start in range(0, flat.shape[0], self.batch_size):
                mu, _ = nets.encode(flat[start : start + self.batch_size].to(self.device))
                means.append(mu)
        return torch.cat(means)

    def _domain_codes(self, domains: ArrayLike, n_rows: int) -> torch.Tensor:
        assert self.domains_ is not None
        values = np.asarray(domains, dtype=object)
        if values.ndim != 1 or len(values) != n_rows:
            raise ValueError(
                f"domains must have one entry per sample ({n_rows}); got {values.shape}."
            )
        lookup = {d: i for i, d in enumerate(_plain(self.domains_) or [])}
        unknown = sorted({repr(v) for v in values.tolist() if v not in lookup})
        if unknown:
            raise ValueError(
                f"domains contains IDs not seen during fit: {', '.join(unknown)}. The decoder is "
                "conditioned on the training domains only."
            )
        return torch.tensor([lookup[v] for v in values.tolist()], dtype=torch.int64)

    def transform(self, X: ArrayLike) -> NDArray[np.float32]:
        """Posterior mean of each sample; returns (n_samples, latent_dim). Needs no domain IDs."""
        out: NDArray[np.float32] = self._posterior_mean(X).cpu().numpy()
        return out

    def reconstruct(self, X: ArrayLike, domains: ArrayLike | None = None) -> NDArray[np.float32]:
        """Decode each sample's posterior mean back to input space; returns X's shape.

        With ``decoder_conditioning="domain"``, ``domains`` (one training-domain ID per sample) is
        required; it is ignored otherwise.
        """
        nets, input_shape = self._fitted()
        mu = self._posterior_mean(X)
        codes = None
        if self.decoder_conditioning == "domain":
            if domains is None:
                raise ValueError(
                    "this model's decoder is conditioned on the domain; "
                    "call reconstruct(X, domains=...)."
                )
            codes = self._domain_codes(domains, mu.shape[0]).to(self.device)
        with torch.no_grad():
            out: NDArray[np.float32] = nets.decode(mu, codes).cpu().numpy()
        return out.reshape(mu.shape[0], *input_shape)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float32]:
        """Class probabilities, shape (n_samples, n_classes); columns follow ``classes_``."""
        nets, _ = self._fitted()
        if len(nets.class_heads) == 0:
            raise RuntimeError(
                "This VAEDANN was fitted without labels, so it has no class head; "
                "fit with y=... to use predict / predict_proba."
            )
        mu = self._posterior_mean(X)
        with torch.no_grad():
            proba: NDArray[np.float32] = torch.softmax(nets.class_heads[0](mu), dim=1).cpu().numpy()
        return proba

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Most probable class per sample, as the original labels from ``y``."""
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        labels: NDArray[Any] = self.classes_[np.argmax(proba, axis=1)]
        return labels


_VAEDANN_PARAMS = (
    "latent_dim",
    "hidden_dim",
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "domain_weight",
    "kl_weight",
    "class_weight",
    "dropout",
    "adversary_input",
    "decoder_conditioning",
    "ramp_rate",
    "device",
    "seed",
)


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    save_checkpoint(path, payload)


def _package_version() -> str:
    from neuro_rave import __version__

    return __version__
