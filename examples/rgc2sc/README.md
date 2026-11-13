# RGC-to-SC comparison (research example)

This folder contains the comparison behind the `VAEDANN` baseline on recorded responses of mouse retinal
ganglion cells (RGC) and of retinal axons in the superior colliculus (SC). It is research code:
it needs local data files that are **not** distributed with neuro_rave, and its helpers are specific to these
recordings. For a self-contained walkthrough, use [`../compare_methods.ipynb`](../compare_methods.ipynb).

| File | Contents |
| --- | --- |
| `run_comparison.py` | Command-line comparison of RAVE, `VAEDANN` and the two-head models, with recording-level splits and `neuro_rave.evaluation` scores |
| `two_head.py` | The notebook's two-head configuration on the package's VAEDANN core: `TwoHeadVAEDANN`, the presets `REFERENCE` (the notebook's settings) and `IMPROVED` (a research configuration selected on these recordings), checkpoint/resume |
| `rgc2sc_helpers.py` | Loading with explicit paths, the notebook's preprocessing, recording-level splits, `block_summary` and `band_correlation` |
| `notebook_model.py` | The notebook's original two-head VAE-DANN code with sampled inference; used to score the shipped checkpoint |

## Data

| Argument | File used in the analysis | What it contains |
| --- | --- | --- |
| `--rgc-table` | `all_RGC_table_20250901.pkl` | RGC recordings in the ganglion cell layer of the isolated retina: trial-averaged chirp (`chirp_8Hz_average_norm`, 259 points at 0.128 s) and moving-bar time component (`preproc_bar`, 32 points), functional group labels `group_id` 1-32 assigned by a classifier (`prob_group` gives its confidence), and recording identifiers |
| `--sc-labels` | `sc_retinal_axons_clusters_clean.h5` | Responses of retinal axon boutons recorded in the superior colliculus (`resp_chirp`, 260 points), with response-cluster IDs `clusterID` 0-49 from clustering these responses, animal, session and scan IDs and depth |
| `--sc-all` | `20240207_df_clusterd_identified.pkl` | All SC samples; merged as in the notebook |
| `--sc-os-ds` | `df_os_ds.pkl` | SC moving-bar time component (`time_comps`, 32 points) and direction/orientation statistics |
| `--checkpoint` (optional) | `vae_dann_checkpoint.pt` | Weights of the notebook's trained model; its training settings were not recorded |

The column meanings above are what the files and the notebook show. Calcium indicators, preparation details
and the meaning of the RGC `cond1` codes are not recorded in these files; check them against the original
experiment documentation before reporting results. Files referenced by the notebook but not part of this
release (receptive-field diameters, `meta_data.parquet`) are not used here.

Install the extra packages: `python -m pip install "neuro-rave[evaluation]" pandas scipy tables`.

## Two-head settings

The notebook trains one model with two class heads in **separate label spaces**: RGC group labels on the
RGC samples and SC cluster labels on the SC samples. `two_head.py` keeps that setup; the public
`neuro_rave.VAEDANN` has one class head shared by all labelled samples and is a different configuration.
`IMPROVED` was selected for these recordings and this budget; it is not a default for other data, and the
public class keeps its own defaults.

| Setting | `REFERENCE` (notebook) | `IMPROVED` (selected on these recordings) |
| --- | --- | --- |
| Domain classifier input | sampled latent | posterior mean |
| Decoder input | latent | latent + one-hot domain code |
| Adversary weight | 20 | 2 |
| Other settings | latent 8, hidden 128, dropout 0.2, 1000 epochs, batch 512 (256 per domain), Adam 1e-3, weight decay 1e-4, KL weight 0.01, class weight 1, ramp rate 5 | same |

```python
from two_head import IMPROVED, TwoHeadVAEDANN

# domain 0 = RGC, 1 = SC; labels -1 where a head has no label
model = TwoHeadVAEDANN(seed=0, **IMPROVED).fit(
    X_train,
    domain_train,
    y_rgc_train,
    y_sc_train,
    checkpoint="two_head_seed0.ckpt",  # saved every 50 epochs
    resume=True,  # continue an interrupted fit with the same call
)
Z = model.transform(X_test)
X_hat = model.reconstruct(X_test, domain_test)  # the conditioned decoder needs domains
```

Results on whole held-out recordings (1000 epochs, seeds 0-2, mean ± SD). The settings were chosen on the
validation recordings; the test recordings were scored after that choice, and those results are exploratory:

| Setting | Validation k-NN domain probe | Validation skill | Test k-NN domain probe | Test skill |
| --- | --- | --- | --- | --- |
| `REFERENCE` | 0.715 ± 0.040 | 0.618 ± 0.008 | 0.681 ± 0.033 | 0.660 ± 0.008 |
| `IMPROVED` | 0.640 ± 0.025 | 0.711 ± 0.007 | 0.608 ± 0.025 | 0.733 ± 0.006 |
| `IMPROVED` with the adversary off (`domain_weight=0`) | 0.888 ± 0.004 | 0.736 ± 0.001 | 0.877 ± 0.009 | 0.754 ± 0.003 |

The k-NN probe is the balanced accuracy of predicting RGC versus SC from the embedding of unseen
recordings (0.5 = chance). Skill is `1 − SSE / SSE` of predicting the training mean. The RGC-group head
reached about 0.50 balanced accuracy on unseen recordings in every setting, with or without the adversary.

Why `IMPROVED` helps, as far as these runs show: at weight 20 the reversed gradient dominates the encoder and
the model's own domain classifier stays near chance (0.49-0.56), while a freshly trained classifier of the
same size still finds the domain (0.65-0.80). This suggests that the encoder evades the current classifier
rather than removing the information; the probe also oscillates during training. At weight 2 the model's
classifier keeps up and the information drops. Conditioning the decoder on the domain recovers about 0.09
reconstruction skill. Extra classifier updates per batch did not help.

## Checkpoint replay and fresh training

- **Checkpoint replay.** Replaying the notebook's trained model (`vae_dann_checkpoint.pt`; `notebook_model.py`
  loads it, and `run_comparison.py --checkpoint` scores it) reproduces its recorded outputs: the notebook's
  probe scores, normalized reconstruction errors and RGC-to-SC best matches. Scored with this example's
  evaluation protocol, the same embedding gives the reported k-NN domain probe of 0.680.
- **Two kinds of probe score.** The notebook's own probes report ordinary accuracy with 5-fold
  cross-validation over all samples (k-NN 0.830, linear 0.628; the linear probe's balanced accuracy is 0.500).
  This example reports balanced accuracy of a probe trained on training recordings and scored on held-out
  recordings; 0.680 is the checkpoint's score under that protocol. The two kinds of score are not comparable.
  The checkpoint was trained on all samples, including the recordings held out here.
- **Same training rule.** The package's VAEDANN core was checked against the notebook code: identical
  forward outputs on the checkpoint, identical initial weights for the same seed, and loss terms, gradients
  and two training epochs equal to floating-point precision.
- **What is not recovered exactly** is the checkpoint's training trajectory, and with it an identical
  fresh-training result. The checkpoint's training settings, number of epochs, data order and random history
  were not recorded. Fresh `REFERENCE` fits give test probes of 0.66-0.72 (seeds 0-2, exploratory), and the
  probe varies by 0.1-0.2 during training at weight 20. Different random sequences are one possible
  explanation for the remaining differences between training paths; no controlled test has established
  this.

## What the scores can and cannot show

- RGC groups and SC clusters are **different label spaces**. No target ground truth exists, so target
  classification accuracy cannot be measured here. Centroid matches between the spaces are similarities in a
  learned embedding; they changed between seeds in our tests and are not cell-type assignments.
- RGC somatic responses and SC bouton responses differ in biology as well as recording conditions, and the two
  populations differ in composition (for example, fewer ON-dominant responses among the SC samples). Some of the
  remaining domain predictability may be real; a lower domain probe alone is not evidence of a better correction.
- All three SC animals contribute to training and test; held-out recordings are new scans of the same animals.
- A 1000-epoch two-head fit on the training recordings takes 16-20 minutes on one CPU core (about 27 minutes
  on all samples). 1000 epochs is this example's budget, not a requirement for other data.
