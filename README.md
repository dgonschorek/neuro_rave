# Neuro-RAVE: Reducing Batch Effects in Neuroscientific Time-Series Data

[![CPU CI](https://github.com/dgonschorek/neuro_rave/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dgonschorek/neuro_rave/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

Neuro-RAVE is a PyTorch package for reducing inter-experimental variability (batch effects) in aligned, fixed-length time-series data. Each training sample is a time series with a domain ID that identifies its batch, such as a recording session or experiment.

- **RAVE (unsupervised)** trains an autoencoder to reconstruct the input time series. An adversarial objective pushes the encoder to make the domain harder to predict from the embedding.
- **RAVE+ (semi-supervised)** adds a classification objective on the labeled samples. This encourages the embedding to retain class-relevant information and supports label prediction for unlabeled time series, including in other domains.

The methods were introduced by Gonschorek et al. at NeurIPS 2021. [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/1e5eeb40a3fce716b244599862fd2200-Abstract.html) · [Original research code](https://github.com/eulerlab/rave)

## Installation

Install the tagged release in a Python 3.10 or newer environment. NumPy and PyTorch are the runtime dependencies. [v0.1.0 release](https://github.com/dgonschorek/neuro_rave/releases/tag/v0.1.0)

```bash
python -m pip install "git+https://github.com/dgonschorek/neuro_rave.git@v0.1.0"
```

For CUDA training, install the matching PyTorch build first and set `device="cuda"`. [PyTorch installation](https://pytorch.org/get-started/locally/)

### Quick example

This example uses random time series to illustrate the API. Replace `X`, `domains` and `y` with your own data.

```python
import numpy as np
from neuro_rave import RAVE, RAVEPlus

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 50))            # samples × time points
domains = np.repeat(["s1", "s2"], 100)   # one domain ID per sample

rave = RAVE(latent_dim=8, epochs=5, seed=0).fit(X, domains=domains)
Z = rave.transform(X)                    # (200, 8)
X_reconstructed = rave.reconstruct(X)    # (200, 50)

y = [None] * len(X)                       # None marks an unknown class
y[:50] = ["on"] * 25 + ["off"] * 25
plus = RAVEPlus(latent_dim=8, epochs=5, seed=0).fit(X, domains=domains, y=y)
y_pred = plus.predict(X)
proba = plus.predict_proba(X)             # (200, 2), columns follow plus.classes_
```

## Where should I start?

- **Try both models on synthetic data** in the [quickstart notebook](examples/quickstart.ipynb).
- **Prepare your own recordings** using the [input requirements](#what-your-data-must-look-like).
- **Evaluate a fit** using the [held-out evaluation checklist](#check-the-result-on-your-own-data).
- **Work on the package** using the [development checks](#contributing).
- **See what shipped** in the [changelog](CHANGELOG.md).

<details>
<summary>Run the quickstart notebook locally</summary>

```bash
git clone https://github.com/dgonschorek/neuro_rave.git
cd neuro_rave
python -m venv .venv
```

Activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS and Linux. Then launch JupyterLab from that environment.

```bash
python -m pip install ".[notebook]"
python -m jupyter lab examples/quickstart.ipynb
```

The notebook uses synthetic two-channel waveforms from three sessions and leaves one session unlabeled. To use your own data, replace the data cell with your time-series data, domain IDs and class labels. The first code cell shows which Python interpreter and package installation the notebook is using.

If your Jupyter installation runs outside this environment, register a kernel from the activated environment and select **Python (neuro-rave)** in Jupyter.

```bash
python -m ipykernel install --user --name neuro-rave --display-name "Python (neuro-rave)"
```

</details>

## Contributing

From a clone, activate a virtual environment and run the development checks before opening a pull request.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests examples
python -m ruff format --check src tests examples
python -m mypy src
```

For a locked development and notebook environment, run `uv sync --extra dev --extra notebook`. [uv documentation](https://docs.astral.sh/uv/)

CI runs the installed package tests on Linux with the minimum dependencies and on Windows with Python 3.13. The Windows job also runs the development checks and executes the notebook. [CPU CI workflow](https://github.com/dgonschorek/neuro_rave/actions/workflows/ci.yml)

## Design decisions and structure

The encoder, decoder, domain discriminator and RAVE+ class head are small ReLU networks. The encoder flattens each time series internally and the decoder restores its original shape. Training alternates between making the embedding less useful for domain prediction and improving the domain discriminator.

| Path | Contents |
| --- | --- |
| [`src/neuro_rave/models.py`](src/neuro_rave/models.py) | Public `RAVE` and `RAVEPlus` classes |
| [`src/neuro_rave/`](src/neuro_rave/) | Networks, training loop and input validation |
| [`tests/`](tests/) | CPU tests for the API, training and validation |
| [`examples/quickstart.ipynb`](examples/quickstart.ipynb) | Synthetic walkthrough of both models |

### What your data must look like

| Input | Requirements |
| --- | --- |
| `X` | Real-valued time-series data shaped `(samples, time)` or `(samples, time, channels)`. Each sample is one time series. Samples share the same time grid and channel order. Values must be finite after conversion to float32. |
| `domains` | One string or integer ID per sample, with at least two distinct domains. |
| `y` for RAVE+ | One string or integer class label per sample, with `None` for unknown labels. At least two classes must be observed. |

Align your time series before fitting. Use the same time and channel layout at inference. `transform` returns a NumPy array of embeddings, `reconstruct` returns the reconstructed time series in the input shape, and RAVE+ can return original class labels or probabilities in `classes_` order. Each call to `fit` starts a fresh training run and returns the model.

<details>
<summary>Model settings and fitted attributes</summary>

| Argument | Default | Meaning |
| --- | --- | --- |
| `latent_dim` | `8` | Embedding size |
| `hidden_dims` | `(256, 128)` | Encoder layer widths, mirrored by the decoder |
| `epochs` | `100` | Passes over the training data |
| `batch_size` | `128` | Minibatch size |
| `learning_rate` | `1e-3` | Learning rate for both Adam optimizers |
| `domain_weight` | `0.1` | Adversarial domain weight. Use `0` to disable this term. |
| `class_weight` | `1.0` | Classification weight for RAVE+ |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `seed` | `None` | Seed for initialization and shuffling |

After fitting, `input_shape_` records the time and channel dimensions, `domains_` and `classes_` list sorted IDs and labels, and `history_` contains per-epoch training losses. `classes_` is available on RAVE+.

</details>

<details>
<summary>Training objective and implementation differences</summary>

1. The encoder, decoder and RAVE+ class head minimize `reconstruction_mse − domain_weight × domain_ce + class_weight × class_ce` with the domain discriminator frozen. The class loss uses labeled samples only.
2. The discriminator minimizes domain cross-entropy using detached embeddings from the updated encoder.

The objective follows equations (3) and (4) of the paper. The implementation averages per-domain reconstruction and domain losses and per-class classification losses within each minibatch. The original code sums those group means. The networks support minibatches with one sample. Each minibatch gets one model step and one discriminator step with Adam betas `(0.5, 0.999)`. The v0.1 API works with prepared time-series data and in-memory fitted models.

</details>

### Check the result on your own data

- Hold out whole recordings or sessions within each domain when possible. Compare reconstruction error per domain and, where labels are available, class separability across domains against a `domain_weight=0` fit.
- Train a separate classifier to predict the domain from held-out embeddings. Compare its performance with chance and with the baseline. Use the training discriminator loss as a training diagnostic.
- Test several seeds and weights. If class and domain are confounded, removing domain information can also remove class information.

The adversarial term rewards higher domain cross-entropy, which has no upper bound. Large adversarial weights or long training can destabilize reconstruction. If held-out error approaches or exceeds the variance of your data, try a lower weight or fewer epochs.

## Related papers and data sources

- **Paper** Gonschorek et al. (2021), *Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience*. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/1e5eeb40a3fce716b244599862fd2200-Abstract.html)
- **Recorded-data comparison** uses mouse retinal bipolar cell responses to the local chirp stimulus. The dataset is available under CC BY-NC-SA 4.0. [G-Node dataset](https://doi.org/10.12751/g-node.5iije0)

### How the default was chosen

The default `domain_weight=0.1` comes from one dataset and one train/test split. Use it as a starting point for the [held-out checks](#check-the-result-on-your-own-data).

<details>
<summary>Recorded-data comparison</summary>

The comparison used RAVE+ on the local chirp recordings. A stratified subsample of 3,000 labeled cells from dataset A was split by cell. Dataset B supplied unlabeled bipolar cell responses, with eight recordings for training and three for testing. A separate k-nearest-neighbour classifier on held-out embeddings measured domain predictability. On raw traces, its balanced accuracy was 0.999. Chance is 0.5 balanced accuracy and 0.69 log loss.

The table reports means over five paired seeds.

| `domain_weight` | Reconstruction MSE, A / B | Balanced class accuracy, A test | Domain classifier balanced accuracy / log loss |
| --- | --- | --- | --- |
| 0 | 0.258 / 0.388 | 0.659 | 0.990 / 0.085 |
| 0.01 | 0.257 / 0.388 | 0.659 | 0.988 / 0.088 |
| **0.1** | 0.273 / 0.432 | 0.650 | **0.805 / 0.420** |
| 1.0 | 0.297 / 0.469 | 0.635 | 0.811 / 0.408 |

At `0.1`, domain-classifier accuracy fell in every seed relative to the baseline, while mean balanced class accuracy on A fell by 0.009. Reconstruction error rose by about 6% on A and 11% on B. The domain classifier still reached about 0.8 balanced accuracy at both `0.1` and `1.0`.

In synthetic checks, `1.0` diverged on one dataset where `0.1` stayed stable. On the notebook's data, RAVE+ at `0.1` collapsed in two of five seeds. The notebook uses `1.0`, the setting checked for that example.

</details>

### Citation

If you use the package, cite the paper. If you use the recorded data, cite the dataset as well. [CITATION.cff](CITATION.cff)

<details>
<summary>BibTeX entries</summary>

```bibtex
@inproceedings{gonschorek2021removing,
  title={Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience},
  author={Gonschorek, Dominic and Höfling, Larissa and Szatko, Klaudia P. and Franke, Katrin and Schubert, Timm and Dunn, Benjamin and Berens, Philipp and Klindt, David and Euler, Thomas},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

```bibtex
@misc{gonschorek2021bipolar,
  title={Simulated and recorded responses of mouse retinal bipolar cells to the chirp stimulus},
  author={Gonschorek, Dominic and Höfling, Larissa and Szatko, Klaudia and Franke, Katrin and Schubert, Timm and Dunn, Benjamin and Berens, Philipp and Klindt, David and Euler, Thomas},
  publisher={G-Node},
  year={2021},
  doi={10.12751/g-node.5iije0}
}
```

</details>

## License

The code uses the MIT license. The recorded dataset is available separately under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). [LICENSE](LICENSE)
