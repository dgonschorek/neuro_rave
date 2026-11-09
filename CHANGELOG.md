# Changelog

## 0.1.0

First experimental release of the Python package for domain adaptation on aligned, fixed-length time series.

- `RAVE` learns embeddings and reconstructs windows using a domain ID for each window.
- `RAVEPlus` also predicts classes from partially labeled training data. Use `None` for unknown labels.
- Both models accept univariate `(samples, time)` and multichannel `(samples, time, channels)` inputs.
- The [quickstart notebook](examples/quickstart.ipynb) walks through both workflows on synthetic data. See the [README](README.md) for installation, examples, and citations.

The default `domain_weight=0.1` is a starting point. On held-out recordings, compare reconstruction error and class separability with a `domain_weight=0` baseline. Use a separate classifier to check how much domain information remains.
