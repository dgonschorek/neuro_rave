# Changelog

## 0.3.0 (2026-11-17)

- `evaluation.assess_batch_effects`: a diagnostic to run before any correction. It measures how well the domain (batch, setup, session) can be predicted from held-out responses with linear and k-NN probes, using `domain_probe` on stratified (group) K-fold splits or on explicit `(train, test)` index splits. With `groups` (cells, recordings, animals), whole groups are held out; every domain must be in every training and test part, and infeasible designs, including `groups=domains`, are rejected instead of falling back to sample-level splits. Scaling and balancing are fitted on each training part only. The result reports pooled held-out scores with chance and majority baselines, per-split results, the splits and settings used, and design notes and limitations.
- With biological labels `y` (`None` for unknown), `assess_batch_effects` also reports the domain-by-class composition, classes labelled in only one domain, and domain predictability within each class shared by several domains, using the same group boundaries; `max_classes` bounds the run time. It describes batch-associated differences; it does not decide whether they are technical or biological, whether correction is needed, or claim statistical significance (no significance test is performed).
- New notebook `examples/batch_effects_workflow.ipynb` (synthetic data): diagnose differences before correction, read class composition and within-class results, decide whether a correction is justified, fit `RAVE` (and optionally `RAVEPlus` and `VAEDANN`) with zero-adversary controls on training recordings only, and compare raw data and representations on the same held-out recordings and splits, together with reconstruction and class prediction. The quickstart and comparison notebooks link to it; CI executes it.
- `domain_probe` now shares its fitting and scoring code with `assess_batch_effects`; its outputs are unchanged. `RAVE`, `RAVEPlus` and `VAEDANN` are unchanged.

## 0.2.0 (2026-11-13)

- `neuro_rave.evaluation`: functions for comparing methods on held-out data. `reconstruction_scores` (MSE and skill relative to a training-mean reference, per group), `classification_scores` (accuracy, balanced accuracy, confusion matrix; `None` marks unknown true labels), `domain_probe` (linear or k-NN domain classifier trained on one set of embeddings and scored on another), `response_correlation`, and `centroid_similarity`, `best_matches` and `matching_agreement` for comparing group centroids across label spaces and fits. `domain_probe` needs the new `evaluation` extra (scikit-learn); the other functions need only NumPy.
- `VAEDANN` (experimental): a variational autoencoder with a gradient-reversal domain adversary and an optional class head trained on partially labeled data (`None` for unknown labels), derived from an RGC-to-SC analysis notebook. `transform`, `reconstruct` and `predict` use the posterior mean. Default `domain_weight=0.2`; the notebook's weight of 20 degraded reconstruction on the tested data. See the README for tested conditions.
- `VAEDANN` options: `adversary_input` (`"sample"` as in the source notebook, or `"mean"`), `decoder_conditioning` (`"none"` or `"domain"`; `reconstruct(X, domains=...)` then needs training-domain IDs) and `ramp_rate`. Defaults are `adversary_input="sample"` and `decoder_conditioning="none"`: the options did not help on every tested dataset (see the README).
- `VAEDANN.fit(..., checkpoint=, checkpoint_every=, resume=)` saves and continues long fits, refusing incompatible settings or data; `get_params()` supports parameter searches.
- New examples: `examples/compare_methods.ipynb` (synthetic; compares models and zero-adversary controls with the evaluation functions, with a resumable grid search on validation data) and `examples/rgc2sc/` (research code for the RGC/SC recordings with explicit data paths; data not included). The latter includes the source notebook's two-head configuration (two class heads in separate label spaces) as `REFERENCE` and a research configuration selected on these recordings as `IMPROVED` (posterior-mean adversary, domain-conditioned decoder, weight 2). `IMPROVED` is not a general default.
- `RAVE` and `RAVEPlus` behavior and defaults are unchanged (seeded outputs identical to 0.1.0). Documentation and the quickstart notebook now describe inputs as time-series data.

## 0.1.0

First experimental release of the Python package for domain adaptation on aligned, fixed-length time series.

- `RAVE` learns embeddings and reconstructs windows using a domain ID for each window.
- `RAVEPlus` also predicts classes from partially labeled training data. Use `None` for unknown labels.
- Both models accept univariate `(samples, time)` and multichannel `(samples, time, channels)` inputs.
- The [quickstart notebook](examples/quickstart.ipynb) walks through both workflows on synthetic data. See the [README](README.md) for installation, examples, and citations.

The default `domain_weight=0.1` is a starting point. On held-out recordings, compare reconstruction error and class separability with a `domain_weight=0` baseline. Use a separate classifier to check how much domain information remains.
