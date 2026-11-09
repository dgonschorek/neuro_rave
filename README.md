# neuro-rave

Reduce domain effects, such as session or experiment differences, in time-series windows with **RAVE** and **RAVE+**. These methods come from:

> D. Gonschorek, L. Höfling, K. P. Szatko, K. Franke, T. Schubert, B. Dunn, P. Berens, D. Klindt, and T. Euler.
> **Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience**.
> *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.
> [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/1e5eeb40a3fce716b244599862fd2200-Abstract.html) · [Original code (eulerlab/rave)](https://github.com/eulerlab/rave)

- **RAVE** learns an embedding from your windows `X` and one domain ID per row. It needs no class labels.
- **RAVE+** uses the same core plus a class head on the embedding. It uses whatever class labels you have, in any domain. Rows without a label still help with reconstruction and domain removal.

This package (`neuro_rave` v0.1) is a fresh, small implementation of both methods. It is not a copy of the original research code; [Differences from the paper and original code](#differences-from-the-paper-and-original-code) lists what changed. MIT-licensed; please [cite the paper](#citation) if you use it.

**Start here:** [install](#install) → [minimal example](#minimal-example) → [example notebook](#example-notebook) → [check the result on your own data](#check-the-result-on-your-own-data) → [run the tests](#run-the-tests).

See the [changelog](CHANGELOG.md) for release notes.

## Install

```bash
python -m pip install "git+https://github.com/dgonschorek/neuro_rave.git"
```

This needs Python 3.10 or newer. The only runtime dependencies are NumPy and PyTorch. For GPU training, install the CUDA build of PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)), then run the command above.

For a private repository, Git needs the same credentials it would use to clone the repository. If you use an SSH key for GitHub, install with the SSH URL.

```bash
python -m pip install "git+ssh://git@github.com/dgonschorek/neuro_rave.git"
```

For a reproducible install, append `@` followed by a tag or full commit SHA to either URL.

## Minimal example

Random numbers stand in for your data here; replace `X`, `domains` and `y` with your own arrays (see [What your data must look like](#what-your-data-must-look-like)).

**RAVE**: windows and one domain ID per window, no class labels.

```python
import numpy as np
from neuro_rave import RAVE

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 50))            # 200 windows x 50 time points; (200, 50, channels) also works
domains = np.repeat(["s1", "s2"], 100)    # one domain (e.g. session) ID per window

rave = RAVE(latent_dim=8, seed=0).fit(X, domains=domains)
Z = rave.transform(X)                     # (200, 8) embedding
X_reconstructed = rave.reconstruct(X)     # (200, 50), same shape as X
```

**RAVE+**: the same, plus class labels where you know them (`None` where you don't).

```python
from neuro_rave import RAVEPlus

y = [None] * 200                          # one label per window; None = unknown
y[:50] = ["on"] * 25 + ["off"] * 25       # labels known for 50 windows

plus = RAVEPlus(latent_dim=8, seed=0).fit(X, domains=domains, y=y)
Z = plus.transform(X)                     # embedding and reconstruction work as for RAVE
y_pred = plus.predict(X)                  # values taken from y: "off" / "on"
proba = plus.predict_proba(X)             # (200, 2); columns follow plus.classes_
```

New windows passed to `transform`, `reconstruct`, `predict` or `predict_proba` must have the same `(time[, channels])` layout as the training data. Window length, channel count, domain IDs and class labels are not fixed by the package. Before using the embedding, [check the result on your own data](#check-the-result-on-your-own-data).

## Example notebook

[`examples/quickstart.ipynb`](examples/quickstart.ipynb) runs both workflows end to end on synthetic two-channel waveforms recorded in three sessions, with class labels unknown (`None`) in one session. It explains the input shapes, plots inputs against their reconstructions, and predicts classes for the unlabeled session. To use your own data, replace the data cell with your fixed-length windows, domain IDs and labels. The notebook runs on CPU.

The notebook needs a few extra packages (JupyterLab, ipykernel, matplotlib). They live in the `notebook` extra, so the package itself still depends only on NumPy and PyTorch. Install everything into one virtual environment and start Jupyter **from that environment**, so the default kernel is the Python that has `neuro_rave`:

```bash
git clone https://github.com/dgonschorek/neuro_rave.git
cd neuro_rave
python -m venv .venv
```

Activate the environment in Windows PowerShell.

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use this activation command.

```bash
source .venv/bin/activate
```

Then install the notebook extra and launch JupyterLab.

```bash
python -m pip install ".[notebook]"
python -m jupyter lab examples/quickstart.ipynb
```

Keep the default kernel, **Python 3 (ipykernel)**. The first notebook cell prints the interpreter path and where `neuro_rave` was imported from; both should point into `.venv`. Without cloning, you can install with `python -m pip install "neuro-rave[notebook] @ git+https://github.com/dgonschorek/neuro_rave.git"` and download the notebook file from GitHub.

If you run notebooks from a Jupyter installation outside this environment (for example VS Code or a global JupyterLab), register the environment as a named kernel and select it in the kernel picker:

```bash
python -m ipykernel install --user --name neuro-rave --display-name "Python (neuro-rave)"
```


## What your data must look like

- **`X`** is a NumPy array (or array-like) of real numbers, shaped `(samples, time)` or `(samples, time, channels)`.
  - Each row is one fixed-length window. All rows share the same time grid and channel order.
  - Windowing, resampling, alignment, filtering and scaling are up to you. The paper z-scored each cell's response. Standardizing each time point or channel using the training rows is a reasonable default.
- **Data types:** computation runs in float32. float64 and integer inputs are converted explicitly, and any NaN or Inf after conversion is rejected.
- **`domains`** has one categorical ID per row (strings or any integers), with at least two distinct domains.
- **`y`** (RAVE+ only) has one class label per row, with `None` where the label is unknown.
  - Labels can be strings or integers that don't have to be consecutive, and they may be known in any domain.
  - At least two classes must be observed. A NaN label is rejected: use `None` for missing labels.
- **Inference** (`transform`, `reconstruct`, `predict`, `predict_proba`):
  - Takes only `X`, which must have the same `(time[, channels])` layout as the training data.
  - Returns NumPy arrays in input row order.
  - `predict` returns the original label values, and `predict_proba` columns follow `classes_`.
- **Refitting:** every `fit` call starts a fresh training run and returns the model itself.

### Settings

| Argument | Default | Meaning |
|---|---|---|
| `latent_dim` | 8 | Size of the embedding |
| `hidden_dims` | (256, 128) | Hidden layer widths of the encoder; the decoder mirrors them |
| `epochs` | 100 | Passes over the training data |
| `batch_size` | 128 | Minibatch size (a final minibatch of one sample is fine) |
| `learning_rate` | 1e-3 | Learning rate of both Adam optimizers |
| `domain_weight` | 0.1 | Weight of the adversarial domain term (0 disables this term); see [how the default was chosen](#how-the-default-was-chosen) |
| `class_weight` | 1.0 | Weight of the class term (`RAVEPlus` only) |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `seed` | `None` | Seeds initialization and shuffling, so CPU runs are reproducible (`seed=0` works) |

After `fit`, the model has these attributes:
- `input_shape_`: the training window layout
- `domains_`: the sorted domain IDs
- `classes_`: the sorted class labels (RAVE+ only)
- `history_`: the mean reconstruction, domain, class and discriminator training loss for each epoch. These are averages taken while the weights are still changing, so they can swing a lot from epoch to epoch. To judge the fitted model, compute errors from `reconstruct()` / `predict()` instead.

## How training works

The model has four parts, all small MLPs: an encoder *f* (window → embedding *z*), a decoder *g* (*z* → reconstructed window), a domain discriminator *h* (*z* → domain), and, for RAVE+, a class head (*z* → class). Each window is flattened to `time × channels` features internally, and reconstructions are reshaped back to the window layout.

Each minibatch gets two updates, in the order the paper describes. Each update has its own Adam optimizer:

1. **Model step:** the encoder, decoder and class head minimize

   `reconstruction_mse − domain_weight · domain_ce + class_weight · class_ce`

   while the discriminator's weights are held fixed. Gradients still flow *through* the frozen discriminator into the encoder, which pushes the embedding toward hiding the domain. The class term exists only for RAVE+, and only labeled rows contribute to it. A minibatch with no labeled rows contributes zero class loss.
2. **Discriminator step:** the discriminator minimizes the ordinary domain cross-entropy on embeddings recomputed with the updated encoder. These embeddings are detached, so this step does not change the encoder.

This is the min–max game in the paper's equations (3) and (4).

**Imbalanced data:** each loss term gives every group in the minibatch equal weight. Reconstruction and domain loss weight each domain equally, and the class loss weights each class among the labeled rows equally. This way a large session or a common class doesn't dominate training.

## Removing domain information vs. keeping signal

Removing domain information always costs some signal. If the embedding could be inverted perfectly, it would carry exactly as much domain information as the raw data. Removing domain information therefore discards something, and `domain_weight` sets how much (paper, Section 3).

- **When domain and signal are separable** (for example, a gain or offset that differs by session, applied to the same set of response types), RAVE can strip the session part and keep the class structure.
- **When they are confounded,** removing domain information also removes class information.
  - Example: a cell type that was recorded in only one session, or class proportions that differ strongly between sessions. Then "which session is this from?" partly answers "which class is this?".
  - The adversary will erase that shared part, and RAVE+ cannot tell the two apart.
  - Labels in more than one domain help, because the class term then rewards keeping exactly the class differences that recur across sessions.
  - If a class exists in only one domain, no method that uses only these inputs can separate it from the domain.

### Check the result on your own data

The default `domain_weight=0.1` comes from a single comparison on one recorded dataset ([how the default was chosen](#how-the-default-was-chosen)), and the other defaults were checked only on synthetic data. How much domain information RAVE removes, and how much signal it keeps, depends on your data and on `domain_weight`, so measure both before relying on the embedding:

- **Hold out data the model hasn't seen.** Split by whole recordings or sessions within each domain where you can, rather than by individual windows.
- **Signal retention:** compute reconstruction error from `reconstruct()` on the held-out windows, per domain, and compare it with a `domain_weight=0` fit. If known classes exist, check that they stay separable in the embedding across domains, for example with a classifier trained on one domain and tested on another.
- **Domain removal:** train a separate classifier (for example logistic regression or k-nearest neighbours) to predict the domain from held-out embeddings, and compare its accuracy and log loss with chance and with the `domain_weight=0` fit. The discriminator used during training is not an independent measurement.
- **Compare a few weights,** starting from the default, for example `domain_weight` 0, 0.1 and 1.0 with several seeds, and pick the setting that removes the most domain information at an acceptable cost in reconstruction and class separability.
- **Look at the class composition of each domain** before trusting predictions for an unlabeled domain.

**Reading the discriminator loss:** a training discriminator loss near chance (ln K for K balanced domains) is the target.
- A loss well *below* chance means the domain is still easy to predict.
- A loss well *above* chance can reflect a discriminator that confidently predicts the wrong domains. The training loss alone cannot quantify how much domain information remains.

The domain discriminator used during training is not an independent measurement. Check domain removal afterwards with a separately trained classifier on held-out data.

**Known limitation:** the model step *maximizes* the discriminator's cross-entropy, exactly as in the paper. That term has no upper bound. With large `domain_weight` values or long training, the domain loss can run away. Reconstruction then collapses to the mean (reconstruction error near the variance of your data) or diverges (error above the variance). Whether this happens depends on the data and the seed, and no weight is safe everywhere. On the recorded data above, no run collapsed at 0.1 or 1.0. In synthetic checks, 1.0 diverged on one dataset where 0.1 stayed stable, while on the notebook's data `RAVEPlus` at 0.1 collapsed in 2 of 5 seeds where 1.0 did not. That is why the notebook sets `domain_weight=1.0` explicitly. Even short of that, the per-epoch losses in `history_` oscillate. After every fit, compare reconstruction error from `reconstruct()` with the variance of your data and with a `domain_weight=0` fit. If it has collapsed or diverged, try a different `domain_weight` (usually lower), fewer `epochs`, or another seed.

### How the default was chosen

`domain_weight=0.1` was chosen from a comparison of 0, 0.01, 0.1 and 1.0 with `RAVEPlus`, using five paired seeds per weight, on recorded data from

> D. Gonschorek, L. Höfling, K. Szatko, K. Franke, T. Schubert, B. Dunn, P. Berens, D. Klindt, and T. Euler (2021). **Simulated and recorded responses of mouse retinal bipolar cells to the chirp stimulus.** G-Node. [doi:10.12751/g-node.5iije0](https://doi.org/10.12751/g-node.5iije0). License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**Setup:**
- **Data:** two-photon recordings of mouse bipolar cell responses to the local chirp stimulus.
- **Source:** dataset A, with cell-type labels. We used a stratified subsample of 3,000 cells, split by cell.
- **Target:** dataset B, whose labels were never used in training. It was split by recording into 8 training and 3 test recordings.
- **Domain removal** was scored with a separately trained k-nearest-neighbour domain classifier on held-out embeddings. On the raw traces it reaches balanced accuracy 0.999 (chance 0.5).

Mean over five seeds:

| `domain_weight` | Reconstruction MSE, A / B | Balanced accuracy, A test | Domain classifier accuracy / log loss (chance 0.5 / 0.69) |
|---|---|---|---|
| 0 | 0.258 / 0.388 | 0.659 | 0.990 / 0.085 |
| 0.01 | 0.257 / 0.388 | 0.659 | 0.988 / 0.088 |
| **0.1** | 0.273 / 0.432 | 0.650 | **0.805 / 0.420** |
| 1.0 | 0.297 / 0.469 | 0.635 | 0.811 / 0.408 |

Compared with the 0 baseline, 0.1 lowered domain-classifier accuracy in every seed. Mean balanced class accuracy on A fell by 0.009, while reconstruction error rose by about 6% on A and 11% on B.

At 0.1 and 1.0, the domain classifier still reached about 0.8 balanced accuracy. The two weights had similar domain-classifier scores, with higher reconstruction error and lower class accuracy at 1.0.

This is one dataset, one stimulus and one split, so treat 0.1 as a starting point and [check the result on your own data](#check-the-result-on-your-own-data). The recordings are **not** included in this repository or package. Get them from the DOI above under its CC BY-NC-SA license, which is separate from this package's MIT license.

## Differences from the paper and original code

The objective and alternating schedule follow the paper. The implementation is deliberately simpler than the original research code:

- **Architecture:** plain ReLU MLPs. There are no residual blocks, dropout, or PCA initialization, and the discriminator and class head have no BatchNorm. Dropping BatchNorm keeps single-sample minibatches valid and makes training and inference behave identically.
- **Loss reductions:** the original code *sums* per-domain and per-class mean losses, so the effective weights grow with the number of domains or classes in a batch. This package *averages* them instead, which keeps the same equal weighting per group at a scale that doesn't depend on how many groups there are.
- **Schedule:** one model step and one discriminator step per minibatch. There are no warm-up or cool-down phases without the adversary, no option to train the discriminator more often, no weight decay, and no skipping of NaN gradients. The Adam betas stay (0.5, 0.999) as in the original code.
- **Scope:** no data loaders, preprocessing, hyperparameter search, or evaluation suite. Fitted models cannot be saved or loaded yet.

## Run the tests

The tests are fast and run on CPU in a few seconds. From a clone, in a virtual environment:

```bash
python -m pip install ".[dev]"
python -m pytest
```

GitHub Actions tests the installed package on Linux with the minimum supported dependencies and on Windows with Python 3.13. The Windows job also runs the quickstart notebook and the development checks below.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples && ruff format --check src tests examples
mypy src
```

Repository layout:

| Path | Contents |
|---|---|
| `src/neuro_rave/` | the package: `models.py` (`RAVE`, `RAVEPlus`) and private `_validation.py`, `_networks.py`, `_training.py` |
| `tests/` | CPU test suite |
| `examples/quickstart.ipynb` | walkthrough for both models |

Extras:
- `notebook`: JupyterLab, ipykernel and matplotlib, for the example notebook
- `dev`: pytest, ruff and mypy

With [uv](https://docs.astral.sh/uv/), `uv sync --extra dev --extra notebook` installs the same setup from the checked-in `uv.lock`.

## Citation

If you use this code in your research, please cite the paper (also available as [`CITATION.cff`](CITATION.cff), which GitHub shows under "Cite this repository"):

```bibtex
@inproceedings{gonschorek2021removing,
  title={Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience},
  author={Gonschorek, Dominic and Höfling, Larissa and Szatko, Klaudia P. and Franke, Katrin and Schubert, Timm and Dunn, Benjamin and Berens, Philipp and Klindt, David and Euler, Thomas},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

If you use the recorded data from [how the default was chosen](#how-the-default-was-chosen), cite the dataset as well:

```bibtex
@misc{gonschorek2021bipolar,
  title={Simulated and recorded responses of mouse retinal bipolar cells to the chirp stimulus},
  author={Gonschorek, Dominic and Höfling, Larissa and Szatko, Klaudia and Franke, Katrin and Schubert, Timm and Dunn, Benjamin and Berens, Philipp and Klindt, David and Euler, Thomas},
  publisher={G-Node},
  year={2021},
  doi={10.12751/g-node.5iije0}
}
```

## License

The code is MIT-licensed; see [LICENSE](LICENSE). No recorded data is distributed with it. The bipolar cell dataset referenced above is licensed separately under CC BY-NC-SA 4.0 by its authors. The original research code ([eulerlab/rave](https://github.com/eulerlab/rave)) was used only as a reference; no code was copied from it.
