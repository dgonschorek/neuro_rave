# Neuro-RAVE <img src="assets/rave-logo.png" align="right" width="160" alt="Neuro-RAVE logo: a neuron and merging response traces" />

Reducing batch effects in neuroscientific time-series data.

[![CPU CI](https://github.com/dgonschorek/neuro_rave/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dgonschorek/neuro_rave/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

Neuro-RAVE is a PyTorch package for reducing inter-experimental variability (batch effects) in aligned, fixed-length time-series data. Each training sample is a time series with a domain ID that identifies its batch, such as a recording session or experiment.

- **RAVE (unsupervised)** trains an autoencoder to reconstruct the input time series. An adversarial objective pushes the encoder to make the domain harder to predict from the embedding.
- **RAVE+ (semi-supervised)** adds a classification objective on the labeled samples. This encourages the embedding to retain class-relevant information and supports label prediction for unlabeled time series, including in other domains.
- **VAEDANN (experimental baseline)** is a variational autoencoder with a gradient-reversal domain adversary and an optional class head. It comes from an analysis of retinal ganglion cell and superior colliculus recordings and is included as a further baseline to compare against, [under tested conditions](#vaedann-tested-conditions).
- **`neuro_rave.evaluation`** scores any method on held-out data: reconstruction against a training-mean reference, classification, and how well a separate classifier predicts the domain from the embedding.

RAVE and RAVE+ were introduced by Gonschorek et al. at NeurIPS 2021. [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/1e5eeb40a3fce716b244599862fd2200-Abstract.html) · [Original research code](https://github.com/eulerlab/rave)

## Installation

Install the tagged release in a Python 3.10 or newer environment. NumPy and PyTorch are the runtime dependencies. [v0.2.0 release](https://github.com/dgonschorek/neuro_rave/releases/tag/v0.2.0)

```bash
python -m pip install "git+https://github.com/dgonschorek/neuro_rave.git@v0.2.0"
```

For CUDA training, install the matching PyTorch build first and set `device="cuda"`. [PyTorch installation](https://pytorch.org/get-started/locally/)

`VAEDANN` and `neuro_rave.evaluation` were added in 0.2.0 (see the [changelog](CHANGELOG.md)). The `evaluation` extra adds scikit-learn, which `evaluation.domain_probe` needs; the other evaluation functions need only NumPy.

```bash
python -m pip install "neuro-rave[evaluation] @ git+https://github.com/dgonschorek/neuro_rave.git@v0.2.0"
```

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

Score fitted models on held-out samples with `neuro_rave.evaluation`. Here a second set of random time series stands in for held-out recordings, and the experimental `VAEDANN` is compared with the same model with its adversary switched off.

```python
from neuro_rave import VAEDANN
from neuro_rave import evaluation as ev

X_test = rng.normal(size=(100, 50))
domains_test = np.repeat(["s1", "s2"], 50)

for weight in (0.0, 0.2):                # 0 switches the adversary off
    model = VAEDANN(domain_weight=weight, epochs=20, seed=0).fit(X, domains=domains)
    rec = ev.reconstruction_scores(
        X_test, model.reconstruct(X_test), reference_mean=X.mean(axis=0), groups=domains_test
    )
    probe = ev.domain_probe(model.transform(X), domains, model.transform(X_test), domains_test)
    print(weight, rec["reconstruction_skill"], probe["balanced_accuracy"])

y_test = ["on"] * 50 + [None] * 50       # None: true label unknown, not scored
scores = ev.classification_scores(y_test, plus.predict(X_test), groups=domains_test)
```

## Where should I start?

- **Try both models on synthetic data** in the [quickstart notebook](examples/quickstart.ipynb).
- **Compare models and zero-adversary controls with the evaluation functions** in the [comparison notebook](examples/compare_methods.ipynb), which also shows a resumable grid search scored on validation data.
- **Rerun the RGC-to-SC comparison** with your local copies of the data using the [research example](examples/rgc2sc/README.md).
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

CI runs the installed package tests on Linux with the minimum dependencies and on Windows with Python 3.13. The Windows job also installs the `evaluation`, `dev` and `notebook` extras, runs the development checks and executes both example notebooks. [CPU CI workflow](https://github.com/dgonschorek/neuro_rave/actions/workflows/ci.yml)

## Design decisions and structure

The encoder, decoder, domain discriminator and RAVE+ class head are small ReLU networks. The encoder flattens each time series internally and the decoder restores its original shape. Training alternates between making the embedding less useful for domain prediction and improving the domain discriminator.

| Path | Contents |
| --- | --- |
| [`src/neuro_rave/models.py`](src/neuro_rave/models.py) | Public `RAVE`, `RAVEPlus` and experimental `VAEDANN` classes |
| [`src/neuro_rave/evaluation.py`](src/neuro_rave/evaluation.py) | Evaluation functions for any method's outputs |
| [`src/neuro_rave/`](src/neuro_rave/) | Networks, training loops (`_training.py` for RAVE, `_vaedann.py` for VAEDANN) and input validation |
| [`tests/`](tests/) | CPU tests for the API, training, evaluation and validation |
| [`examples/quickstart.ipynb`](examples/quickstart.ipynb) | Synthetic walkthrough of RAVE and RAVE+ |
| [`examples/compare_methods.ipynb`](examples/compare_methods.ipynb) | Synthetic comparison of all models with the evaluation functions |
| [`examples/rgc2sc/`](examples/rgc2sc/) | Research example for the RGC-to-SC recordings (data not included) |

### What your data must look like

| Input | Requirements |
| --- | --- |
| `X` | Real-valued time-series data shaped `(samples, time)` or `(samples, time, channels)`. Each sample is one time series. Samples share the same time grid and channel order. Values must be finite after conversion to float32. |
| `domains` | One string or integer ID per sample, with at least two distinct domains. |
| `y` for RAVE+ and VAEDANN | One string or integer class label per sample, with `None` for unknown labels. At least two classes must be observed. Required for RAVE+, optional for VAEDANN. |

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
<summary>VAEDANN settings and fitted attributes (experimental)</summary>

`VAEDANN.fit(X, domains, y=None)` takes the same inputs as RAVE and RAVE+; `y` is optional. With `y`, one class head is trained on the samples whose label is known (`None` marks unknown labels, in any domain), and all labels share one label space. Without `y` there is no class head, and `predict` / `predict_proba` raise an error.

`transform` and `predict` need no domain IDs. With `decoder_conditioning="domain"`, the decoder uses the domain, so `reconstruct(X, domains=...)` needs one training-domain ID per sample; IDs not seen in training raise an error.

| Argument | Default | Meaning |
| --- | --- | --- |
| `latent_dim` | `8` | Latent size |
| `hidden_dim` | `128` | Width of the hidden layers |
| `epochs` | `None` | `None` presents as many samples as 100 passes over the data; see below |
| `batch_size` | `512` | Minibatch size, split equally across domains |
| `learning_rate` | `1e-3` | Adam learning rate |
| `weight_decay` | `1e-4` | Adam weight decay |
| `domain_weight` | `0.2` | Adversarial domain weight. Use `0` to disable the adversary. |
| `kl_weight` | `1e-2` | Weight of the KL term |
| `class_weight` | `1.0` | Classification weight (only with `y`) |
| `dropout` | `0.2` | Dropout in the encoder and domain classifier during training |
| `adversary_input` | `"sample"` | What the domain classifier sees: `"sample"` (a sampled latent, as in the source notebook) or `"mean"` (the posterior mean that `transform` returns) |
| `decoder_conditioning` | `"none"` | `"domain"` also feeds a one-hot domain code to the decoder; `reconstruct` then needs `domains=` with training-domain IDs |
| `ramp_rate` | `5.0` | Speed of the reversal ramp `alpha = 2 / (1 + exp(−ramp_rate × epoch / epochs)) − 1`; smaller is slower |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `seed` | `None` | Seeds initialization, dropout, latent sampling and batch order without changing global random state |

`fit(X, domains, y=None, checkpoint=path, checkpoint_every=50, resume=False)` saves the full training state (weights, optimizer, epoch, random states, history, settings, category mappings and a fingerprint of the data) atomically every `checkpoint_every` epochs. After an interruption, the same call with `resume=True` continues from the last saved epoch and gives the same result as an uninterrupted fit on the same machine and PyTorch version; it refuses if the settings or data differ. `get_params()` returns the constructor arguments, so `VAEDANN(**params)` creates a fresh model for each candidate in a parameter search. The [comparison notebook](examples/compare_methods.ipynb) runs such a grid search, scores each candidate on validation data against its zero-adversary fit and skips candidates already finished when rerun.

The public class is the generic, shared-head configuration: one class head for one label space. The source notebook instead trained two heads in separate label spaces (RGC group labels on RGC samples, SC cluster labels on SC samples). That two-head configuration is research code for those recordings in [`examples/rgc2sc/`](examples/rgc2sc/README.md), not part of the package API.

Every minibatch holds `batch_size // n_domains` samples of each domain (at most the smallest domain's size). An epoch ends when the smallest domain is used up, so larger domains are subsampled anew each epoch. `epochs_` records the number of epochs used. `history_` has per-epoch means of the reconstruction, KL, classification and domain losses, the domain classifier's training accuracy and the reversal strength `alpha`. `transform`, `reconstruct` and the predictions use the posterior mean, so they are deterministic. If a loss becomes non-finite, `fit` raises `FloatingPointError` instead of skipping batches.

</details>

<details>
<summary>Training objective and implementation differences</summary>

1. The encoder, decoder and RAVE+ class head minimize `reconstruction_mse − domain_weight × domain_ce + class_weight × class_ce` with the domain discriminator frozen. The class loss uses labeled samples only.
2. The discriminator minimizes domain cross-entropy using detached embeddings from the updated encoder.

The objective follows equations (3) and (4) of the paper. The implementation averages per-domain reconstruction and domain losses and per-class classification losses within each minibatch. The original code sums those group means. The networks support minibatches with one sample. Each minibatch gets one model step and one discriminator step with Adam betas `(0.5, 0.999)`. The v0.1 API works with prepared time-series data and in-memory fitted models.

**VAEDANN** uses one Adam step per minibatch over all parts, minimizing `reconstruction_mse + kl_weight × kl + class_weight × class_ce + domain_weight × domain_ce`. By default the decoder and the domain classifier see a latent sampled from the encoder's Gaussian posterior; `adversary_input="mean"` gives the domain classifier the posterior mean instead, and `decoder_conditioning="domain"` appends a one-hot domain code to the decoder's input. The domain classifier is connected through a gradient-reversal layer: it learns to predict the domain, while the encoder receives the reversed gradient scaled by `alpha = 2 / (1 + exp(−ramp_rate × epoch / epochs)) − 1`, which ramps from 0 towards 1. All terms are plain means over the minibatch. Unlike the source notebook, the public class has one class head for one label space and uses the posterior mean for inference; the notebook's two-head setup is kept in [`examples/rgc2sc/`](examples/rgc2sc/).

</details>

### Check the result on your own data

- Hold out whole recordings or sessions within each domain when possible. Compare reconstruction error per domain and, where labels are available, class separability across domains against a `domain_weight=0` fit (`evaluation.reconstruction_scores`, `evaluation.classification_scores`).
- Train a separate classifier to predict the domain from held-out embeddings (`evaluation.domain_probe`). Compare its balanced accuracy with chance and with the `domain_weight=0` fit. Use the training discriminator loss only as a training diagnostic.
- Test several seeds and weights. If class and domain are confounded, removing domain information can also remove class information.
- Read the scores together. A lower domain probe alone does not show a useful correction: some differences between recordings can be biological. Remaining domain information alone is not a failure either.

The adversarial term rewards higher domain cross-entropy, which has no upper bound. Large adversarial weights or long training can destabilize reconstruction. If held-out error approaches or exceeds the variance of your data, try a lower weight or fewer epochs.

## Related papers and data sources

- **Paper** Gonschorek et al. (2021), *Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience*. [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/1e5eeb40a3fce716b244599862fd2200-Abstract.html)
- **Recorded-data comparison** uses mouse retinal bipolar cell responses to the local chirp stimulus. The dataset is available under CC BY-NC-SA 4.0. [G-Node dataset](https://doi.org/10.12751/g-node.5iije0)
- **VAEDANN** combines a variational autoencoder (Kingma and Welling, *Auto-Encoding Variational Bayes*, ICLR 2014) with domain-adversarial training through gradient reversal (Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 2016).
- **VAEDANN comparison** also uses recordings of retinal ganglion cells and of retinal axon boutons in the superior colliculus from the RGC2SC project. They are not distributed with this package; the [research example](examples/rgc2sc/README.md) lists the files it expects.

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

### VAEDANN tested conditions

`VAEDANN`'s default `domain_weight=0.2` was chosen with acceptance criteria that were written down before the weight comparison was run. A weight qualified if, on validation recordings and against the same model with its adversary off, it met all of the following in all three seeds:
- it lowered the held-out k-NN domain probe;
- it cost at most 0.10 reconstruction skill (skill is `1 − SSE / SSE` of predicting the training mean);
- it cost at most 0.05 balanced accuracy on the available labels.

This is a small comparison on two recorded datasets and should be read as a starting point, not a guarantee.

<details>
<summary>Weight comparison on recorded data</summary>

Data:
- **Bipolar cells:** the dataset above, with dataset A as the labeled source and dataset B split by recording.
- **RGC/SC:** mouse retinal ganglion cell somatic responses and retinal bouton responses recorded in the superior colliculus. These are chirp plus moving-bar responses, split by recording, with RGC group labels as source labels and the SC samples unlabeled. These recordings are not distributed with this package.

Each row compares the model with its zero-adversary fit, using validation recordings and the mean over seeds 0–2. Changes are given as adversarial fit minus zero-adversary fit.

| Data, labels | `domain_weight` | k-NN domain probe change | Reconstruction skill change | Label balanced-accuracy change |
| --- | --- | --- | --- | --- |
| Bipolar cells, none | **0.2** | −0.048 | −0.051 | −0.019 |
| | 2 | −0.018 | −0.140 | −0.077 |
| | 20 (notebook) | −0.027 | −0.304 | −0.159 |
| Bipolar cells, source | **0.2** | −0.237 | −0.033 | −0.010 |
| | 2 | −0.072 | −0.097 | −0.010 |
| | 20 (notebook) | −0.032 | −0.243 | −0.377 |
| RGC/SC, source | **0.2** | −0.236 | −0.074 | −0.005 |
| | 2 | −0.167 | −0.130 | −0.005 |
| | 20 (notebook) | −0.142 | −0.124 | −0.031 |

For reference, the RAVE+ default changes the probe by −0.193 and −0.173 and the skill by −0.027 and −0.071 on the same two labeled settings. The zero-adversary k-NN probe was 0.94–0.98 and stayed well above chance (0.5) in every setting. Test recordings showed the same direction.

Where it did not help:
- On the synthetic data of the [quickstart](examples/quickstart.ipynb), where sessions differ by large offsets, no tested weight lowered the domain probe.
- At weight 20, reconstruction collapsed to below the training-mean baseline in 9 of 10 fits on that data.
- Unsupervised `VAEDANN` on the RGC/SC data was tested only at weight 20, where it cost 0.3 reconstruction skill.
- Best matches between RGC groups and SC clusters in the learned embedding changed between seeds for every model. Treat such matches as hypotheses, not cell-type assignments.

</details>

<details>
<summary>Architecture options: posterior-mean adversary and domain-conditioned decoder</summary>

`adversary_input="mean"` together with `decoder_conditioning="domain"` was compared with the defaults in a separate run of the package class (default budget, `domain_weight=0.2`, seeds 0–2, validation recordings). Each change is the fit minus its own zero-adversary fit with the same decoder:

| Data, labels | Setting | k-NN domain probe change | Reconstruction skill change | Label balanced-accuracy change |
| --- | --- | --- | --- | --- |
| Bipolar cells, source | defaults | −0.229 | −0.033 | −0.001 |
| | mean + domain | −0.276 | −0.018 | +0.003 |
| Bipolar cells, none | defaults | −0.101 | −0.067 | – |
| | mean + domain | **−0.016** | −0.018 | – |
| RGC/SC, source | defaults | −0.249 | −0.068 | −0.008 |
| | mean + domain | −0.246 | −0.024 | −0.004 |

With source labels, the options lowered the probe as much as the defaults and cost less reconstruction skill. Without labels they barely lowered the probe on the bipolar-cell data. On the synthetic quickstart data one of three fits became unstable (the KL term grew to about 300 and reconstruction fell below the training-mean baseline). On the simulated bipolar-cell data no setting lowered the k-NN probe noticeably; accuracy on the target scan's true labels rose from 0.27 (adversary off) to 0.49 with the options and to 0.36 with the defaults, varying widely between seeds. A weight of 2 failed the reconstruction criterion on the bipolar-cell data with either setting. The defaults are therefore unchanged; try the options with your own zero-adversary comparison.

For the source notebook's two-head RGC/SC configuration (1000 epochs), a research configuration (`IMPROVED` in [`examples/rgc2sc/`](examples/rgc2sc/README.md)) combines both options with `domain_weight=2`. Compared with the notebook's settings, it lowered the validation k-NN domain probe from 0.715 to 0.640 and raised validation reconstruction skill from 0.62 to 0.71 (seeds 0–2). It was selected for these recordings only and is not a default for other data. Diagnostics on these recordings suggest that at the notebook's weight of 20 the encoder evades the current domain classifier rather than removing domain information.

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
