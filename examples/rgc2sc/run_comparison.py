"""Compare models on the RGC-to-SC recorded responses with explicit data paths.

Example (Windows paths work the same way):

    python examples/rgc2sc/run_comparison.py \
        --rgc-table DATA/all_RGC_table_20250901.pkl \
        --sc-labels DATA/sc_retinal_axons_clusters_clean.h5 \
        --sc-all DATA/20240207_df_clusterd_identified.pkl \
        --sc-os-ds DATA/df_os_ds.pkl \
        --seeds 0 1 2 --out rgc2sc_results.json

Two-head models (``two_head.py``, on the package's VAEDANN core): ``--two-head improved`` (default)
fits the research configuration selected on these recordings, ``reference`` the notebook's
configuration, ``both`` both, ``none`` skips them. ``--two-head-epochs N`` changes their budget
(default 1000, about 16-20 minutes per fit on one CPU core); ``--checkpoint-dir DIR`` saves their
training state every 50 epochs, and running the same command again resumes interrupted fits.
``--checkpoint DATA/vae_dann_checkpoint.pt`` also scores the notebook's shipped model.

The data are not distributed with neuro_rave. RGC groups and SC response clusters are different
label spaces: matches between them are similarities in the embedding, not cell-type assignments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import notebook_model as nm  # noqa: E402
import two_head as th  # noqa: E402
from rgc2sc_helpers import load_rgc2sc, split_by_recording  # noqa: E402

from neuro_rave import RAVE, VAEDANN  # noqa: E402
from neuro_rave import evaluation as ev  # noqa: E402


def score(name, seed, Z_train, Z_test, X_test, X_hat, data, tr, te, mean_train):
    out = {"model": name, "seed": seed}
    out["reconstruction"] = {
        k: v
        for k, v in ev.reconstruction_scores(
            X_test, X_hat, reference_mean=mean_train, groups=data["domain"][te]
        ).items()
        if k != "per_sample_mse"
    }
    for kind in ("knn", "linear"):
        r = ev.domain_probe(
            Z_train,
            data["domain"][tr],
            Z_test,
            data["domain"][te],
            kind=kind,
            max_train_per_domain=8000,
            seed=0,
        )
        out[f"domain_probe_{kind}"] = {
            k: r[k]
            for k in (
                "balanced_accuracy",
                "accuracy",
                "log_loss",
                "chance_balanced_accuracy",
                "majority_accuracy",
            )
        }
    rgc_tr = data["domain"][tr] == "RGC"
    sc_tr = data["domain"][tr] == "SC"
    sim = ev.centroid_similarity(
        Z_train[rgc_tr],
        data["rgc_group"][tr][rgc_tr],
        Z_train[sc_tr],
        data["sc_cluster"][tr][sc_tr],
    )
    out["matching"] = {
        "similarity": sim["similarity"].tolist(),
        "groups_a": sim["groups_a"],
        "groups_b": sim["groups_b"],
    }
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("rgc-table", "sc-labels", "sc-all", "sc-os-ds"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument(
        "--two-head", choices=("improved", "reference", "both", "none"), default="improved"
    )
    ap.add_argument("--two-head-epochs", type=int, default=th.REFERENCE["epochs"])
    ap.add_argument("--checkpoint-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("rgc2sc_results.json"))
    a = ap.parse_args()

    data = load_rgc2sc(a.rgc_table, a.sc_labels, a.sc_all, a.sc_os_ds)
    split = split_by_recording(data["recording"], data["domain"])
    tr, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
    X = data["X"]
    mean_train = X[tr].mean(axis=0)
    y_rgc = data["rgc_group"]
    print(
        f"{len(X)} samples; train {len(tr)}, test {len(te)} (whole recordings held out)", flush=True
    )
    results = []

    def add(name, seed, transform, reconstruct):
        results.append(
            score(
                name,
                seed,
                transform(X[tr]),
                transform(X[te]),
                X[te],
                reconstruct(X[te]),
                data,
                tr,
                te,
                mean_train,
            )
        )
        r = results[-1]
        print(
            f"{name:34s} seed {seed}: skill {r['reconstruction']['reconstruction_skill']:.3f}, "
            f"k-NN domain probe {r['domain_probe_knn']['balanced_accuracy']:.3f}",
            flush=True,
        )

    for seed in a.seeds:
        for name, model, labels in [
            ("RAVE, adversary off", RAVE(domain_weight=0.0, seed=seed), None),
            ("RAVE", RAVE(seed=seed), None),
            (
                "VAEDANN + RGC labels, adversary off",
                VAEDANN(domain_weight=0.0, seed=seed),
                y_rgc[tr],
            ),
            ("VAEDANN + RGC labels", VAEDANN(seed=seed), y_rgc[tr]),
        ]:
            t0 = time.time()
            model.fit(X[tr], data["domain"][tr], **({} if labels is None else {"y": list(labels)}))
            add(name, seed, model.transform, model.reconstruct)
            results[-1]["fit_seconds"] = round(time.time() - t0, 1)
        presets = {"reference": th.REFERENCE, "improved": th.IMPROVED}
        chosen = list(presets) if a.two_head == "both" else [a.two_head]
        for preset in [p for p in chosen if p != "none"]:
            dom = (data["domain"][tr] == "SC").astype(int)
            dom_te = (data["domain"][te] == "SC").astype(int)
            ya = np.array([g - 1 if g is not None else -1 for g in y_rgc[tr]])
            yb = np.array([c if c is not None else -1 for c in data["sc_cluster"][tr]])
            settings = {**presets[preset], "epochs": a.two_head_epochs}
            ckpt = None
            if a.checkpoint_dir is not None:
                a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                ckpt = a.checkpoint_dir / f"two_head_{preset}_seed{seed}.ckpt"
            t0 = time.time()
            model = th.TwoHeadVAEDANN(seed=seed, **settings)
            model.fit(X[tr], dom, ya, yb, checkpoint=ckpt, resume=ckpt is not None)
            conditioned = settings["decoder_conditioning"] == "domain"

            def recon(Z, m=model, codes=dom_te if conditioned else None):
                # Z is always X[te] here; a conditioned decoder needs the samples' domains.
                return m.reconstruct(Z, codes)

            add(f"two-head {preset} ({a.two_head_epochs} epochs)", seed, model.transform, recon)
            results[-1]["fit_seconds"] = round(time.time() - t0, 1)
            results[-1]["settings"] = settings
    if a.checkpoint is not None:
        ck = nm.load_notebook_checkpoint(a.checkpoint)
        add(
            "notebook checkpoint (trained on all samples)",
            -1,
            lambda Z: nm.transform(ck, Z),
            lambda Z: nm.reconstruct_sampled(ck, Z),
        )

    for name in {r["model"] for r in results}:
        runs = [r["matching"] for r in results if r["model"] == name]
        if len(runs) > 1:
            agree = ev.matching_agreement(
                [{**m, "similarity": np.array(m["similarity"])} for m in runs], tie_tolerance=0.02
            )
            print(f"{name}: best-match agreement across seeds {agree['pairwise_agreement']:.2f}")
    a.out.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    print("written", a.out)


if __name__ == "__main__":
    main()
