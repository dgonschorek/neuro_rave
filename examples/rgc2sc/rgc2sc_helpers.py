"""Loading, preprocessing and research-specific evaluation for the RGC-to-SC comparison.

Everything specific to this dataset lives here, not in ``neuro_rave``: file layout, the notebook's
preprocessing, recording-level splits, block summaries over hand-defined group blocks and
frequency-band correlations. All data paths, blocks and sampling rates are explicit arguments.

Needs ``pandas``, ``scipy`` and (for the HDF5 file) ``tables``:
``python -m pip install pandas scipy tables``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_rgc2sc(rgc_table, sc_labels, sc_all, sc_os_ds):
    """Load and preprocess the data as the notebook does (cells 8-15). Paths are required.

    Args:
        rgc_table: pickle with one RGC recording per row (``all_RGC_table_*.pkl``); needs columns
            ``chirp_8Hz_average_norm``, ``preproc_bar``, ``group_id``, ``chirp_qidx``, ``bar_qidx``,
            ``animgender``, ``cond1``, ``date``, ``exp_num``, ``field``.
        sc_labels: HDF5 (key ``df``) with the clustered SC samples
            (``sc_retinal_axons_clusters_clean.h5``);
            needs ``resp_chirp``, ``clusterID``, ``animal_id``, ``session_id``, ``scan_id``.
        sc_all: pickle with all SC samples (``20240207_df_clusterd_identified.pkl``), merged as in
            the notebook.
        sc_os_ds: pickle with SC moving-bar components (``df_os_ds.pkl``); needs ``time_comps``.

    Returns:
        dict with ``X`` (samples, 292: chirp resampled to 260 points on a shared 0-32 s axis and
        divided by its maximum absolute value, followed by the 32-point bar time component),
        ``domain`` (``"RGC"``/``"SC"``), ``rgc_group`` (1-32, ``None`` for SC), ``sc_cluster``
        (0-49, ``None`` for RGC), ``recording`` (RGC: date_exp_field; SC: animal_session_scan) and
        ``animal`` (RGC: date_exp; SC: animal ID). Missing files raise ``FileNotFoundError`` naming
        the file.
    """
    import pandas as pd
    from scipy.interpolate import interp1d

    for p in (rgc_table, sc_labels, sc_all, sc_os_ds):
        if not Path(p).exists():
            raise FileNotFoundError(f"required data file not found: {p}")
    lab = pd.read_hdf(sc_labels, key="df")
    df_sc = lab.combine_first(pd.read_pickle(sc_all)).loc[lab.index]
    with open(sc_os_ds, "rb") as f:
        os_ds = pickle.load(f)  # local, trusted research file
    df_sc = df_sc.join(os_ds.loc[df_sc.index, os_ds.columns.difference(df_sc.columns)])

    rgc = pd.read_pickle(rgc_table)
    rgc = rgc[(rgc["chirp_qidx"] > 0.35) | (rgc["bar_qidx"] > 0.6)]
    rgc = rgc[rgc["animgender"].isin(["male", "female"])]
    rgc = rgc[rgc["cond1"].isin(["C1", "C2", "c1", "c3", "c4", "control"])]
    rgc = rgc[rgc["group_id"] < 33].reset_index(drop=True)

    t = np.linspace(0, 32, 260)

    def prep(chirp, bar):
        c = interp1d(np.linspace(0, 32, chirp.shape[1]), chirp, kind="linear", axis=1)(t)
        m = np.max(np.abs(c), axis=1, keepdims=True)
        m[m == 0] = 1.0
        return np.hstack([c / m, bar])

    X_rgc = prep(
        np.stack(rgc["chirp_8Hz_average_norm"].values), np.stack(rgc["preproc_bar"].values)
    )
    X_sc = prep(np.stack(df_sc["resp_chirp"].values), np.stack(df_sc["time_comps"].values))
    n_r, n_s = len(X_rgc), len(X_sc)
    rgc_rec = (
        rgc["date"].astype(str) + "_" + rgc["exp_num"].astype(str) + "_" + rgc["field"].astype(str)
    )
    sc_rec = (
        df_sc["animal_id"].astype(str)
        + "_"
        + df_sc["session_id"].astype(str)
        + "_"
        + df_sc["scan_id"].astype(str)
    )
    return {
        "X": np.vstack([X_rgc, X_sc]).astype(np.float32),
        "domain": np.array(["RGC"] * n_r + ["SC"] * n_s),
        "rgc_group": np.array([int(g) for g in rgc["group_id"]] + [None] * n_s, dtype=object),
        "sc_cluster": np.array([None] * n_r + [int(c) for c in df_sc["clusterID"]], dtype=object),
        "recording": np.concatenate(
            ["RGC:" + rgc_rec.values.astype(str), "SC:" + sc_rec.values.astype(str)]
        ),
        "animal": np.concatenate(
            [
                (rgc["date"].astype(str) + "_" + rgc["exp_num"].astype(str)).values.astype(str),
                df_sc["animal_id"].astype(str).values,
            ]
        ),
    }


def split_by_recording(recording, domain, fractions=(0.6, 0.2, 0.2), seed=0):
    """Assign whole recordings to ``"train"``/``"val"``/``"test"`` within each domain."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(recording), dtype=object)
    for dom in np.unique(domain):
        rows = np.flatnonzero(domain == dom)
        recs, counts = np.unique(recording[rows], return_counts=True)
        target = {"test": fractions[2] * len(rows), "val": fractions[1] * len(rows)}
        filled = {"test": 0, "val": 0}
        assign = {}
        for i in rng.permutation(len(recs)):
            for part in ("test", "val"):
                if filled[part] < target[part] and filled[part] + counts[i] <= 1.15 * target[part]:
                    assign[recs[i]], filled[part] = part, filled[part] + counts[i]
                    break
            else:
                assign[recs[i]] = "train"
        out[rows] = [assign[r] for r in recording[rows]]
    return out.astype(str)


def block_summary(similarity, row_blocks, col_blocks):
    """Summarize a (row groups x column groups) similarity matrix over blocks of groups.

    Blocks are inclusive ``(first, last)`` positions along each axis, e.g. supergroups of RGC groups
    and of SC clusters. Both outputs use the same definition (the notebook mixed inclusive and
    half-open blocks). Returns ``(best_match_fraction, block_mean)``, each (row blocks, col blocks):
    the fraction of rows in a row block whose best column falls in each column block, and the mean
    similarity within each block pair.
    """
    sim = np.asarray(similarity, dtype=float)
    best = np.nanargmax(sim, axis=1)
    frac = np.zeros((len(row_blocks), len(col_blocks)))
    mean = np.zeros_like(frac)
    for i, (r0, r1) in enumerate(row_blocks):
        for j, (c0, c1) in enumerate(col_blocks):
            mean[i, j] = np.nanmean(sim[r0 : r1 + 1, c0 : c1 + 1])
            frac[i, j] = np.mean((best[r0 : r1 + 1] >= c0) & (best[r0 : r1 + 1] <= c1))
    return frac, mean


def band_correlation(X, X_hat, *, sampling_rate_hz, cutoff_hz, order=2):
    """Per-sample Pearson correlation of low-pass and high-pass components (zero-phase Butterworth).

    The notebook used ``sampling_rate_hz=7.8125`` and ``cutoff_hz=2`` for its 260-point chirp.
    Returns ``{"low": array, "high": array}``; constant components give NaN.
    """
    from scipy.signal import butter, filtfilt

    from neuro_rave.evaluation import response_correlation

    b, a = butter(order, cutoff_hz / (sampling_rate_hz / 2), btype="low")
    lo, lo_hat = filtfilt(b, a, X, axis=1), filtfilt(b, a, X_hat, axis=1)
    return {
        "low": response_correlation(lo, lo_hat)["per_sample"],
        "high": response_correlation(X - lo, X_hat - lo_hat)["per_sample"],
    }
