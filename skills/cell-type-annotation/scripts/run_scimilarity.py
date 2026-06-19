#!/usr/bin/env python
"""Optional zero-shot cross-check via SCimilarity — a large-memory HYPOTHESIS (not a verdict).

SCimilarity embeds cells against a large pretrained reference (~tens of millions of cells)
and returns a per-cell nearest-neighbor cell-type call. It needs the model bundle on disk
(tens of GB) and a lot of RAM (~64 GB) — so it belongs to the large-memory profile, not the
CPU default. It is a foundation-model annotator, and 2026 benchmarks show such models rarely
beat scANVI / simple baselines for annotation, so this is strictly an ADDITIONAL cross-check
to reconcile against marker evidence, never a primary annotator (reference/automated_tools.md).

It writes the SAME per-cluster schema reference_map.py emits, so the cards/report pick it
up as one more cross-check automatically.

SCimilarity expects RAW COUNTS (it aligns genes to the model's gene order and applies its
own log-normalization). Point --counts-layer at the query's counts.

Usage:
  python run_scimilarity.py --query Q.h5ad --cluster-key leiden --workdir DIR --level L1 \
      --model-path /path/to/scimilarity_model --counts-layer counts [--label-col scimilarity]

Writes tables/refmap_<level>_<label-col>.csv with columns:
  cluster, top1, frac1, top2, frac2, mean_conf, method
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import scanpy as sc

from cta_io import (eprint, require_obs, resolve_workdir, safe_cluster_id,
                    save_table)


def _aggregate(clusters, labels, conf) -> pd.DataFrame:
    df = pd.DataFrame({"cluster": clusters, "label": labels, "conf": conf})
    rows = []
    for cl, g in df.groupby("cluster", sort=True):
        vc = g["label"].value_counts()
        n = vc.sum()
        top1, frac1 = vc.index[0], round(float(vc.iloc[0] / n), 4)
        top2, frac2 = (vc.index[1], round(float(vc.iloc[1] / n), 4)) if len(vc) > 1 else ("", 0.0)
        mean_conf = round(float(np.nanmean(g["conf"])), 4) if g["conf"].notna().any() else ""
        rows.append(dict(cluster=cl, top1=top1, frac1=frac1, top2=top2, frac2=frac2,
                         mean_conf=mean_conf, method="scimilarity"))
    return pd.DataFrame(rows)


def main():
    args = _parse_args()
    try:
        from scimilarity import CellAnnotation, align_dataset, lognorm_counts
    except ImportError:
        raise SystemExit(
            "ERROR: scimilarity is not installed. It is an optional large-memory extra "
            "for this skill (requirements-gpu.txt). Install with:\n"
            "   pip install scimilarity\n"
            "and download the model bundle (tens of GB; needs ~64GB RAM at runtime)."
        )

    eprint(f">> loading query {args.query}")
    query = sc.read_h5ad(args.query)
    require_obs(query, args.cluster_key, "cluster key")

    # SCimilarity wants raw counts in X.
    if args.counts_layer not in (None, "X"):
        if args.counts_layer not in (query.layers or {}):
            raise SystemExit(f"ERROR: --counts-layer '{args.counts_layer}' not in "
                             f"query.layers {list(query.layers.keys())}.")
        query.X = query.layers[args.counts_layer]

    eprint(f">> loading SCimilarity model: {args.model_path}")
    ca = CellAnnotation(model_path=args.model_path)

    eprint(">> aligning genes to model gene order + log-normalizing")
    query = align_dataset(query, ca.gene_order)
    query = lognorm_counts(query)

    eprint(">> embedding + kNN annotation")
    query.obsm["X_scimilarity"] = ca.get_embeddings(query.X)
    result = ca.get_predictions_knn(query.obsm["X_scimilarity"])
    # API returns (predictions, nn_idxs, nn_dists, nn_stats); be tolerant of arity.
    predictions = result[0] if isinstance(result, (tuple, list)) else result
    pred = np.asarray(getattr(predictions, "values", predictions)).astype(str)

    # Per-cell confidence: invert min nearest-neighbor distance into 0-1 (relative within
    # this query), if distances are available; else leave blank.
    conf = np.full(query.n_obs, np.nan)
    try:
        nn_stats = result[3] if len(result) > 3 else None
        dist = None
        if nn_stats is not None and hasattr(nn_stats, "columns"):
            for c in ("min_dist", "mean_dist", "dist"):
                if c in nn_stats.columns:
                    dist = nn_stats[c].to_numpy(dtype=float)
                    break
        if dist is not None and np.isfinite(dist).any():
            lo, hi = np.nanmin(dist), np.nanmax(dist)
            conf = 1.0 - (dist - lo) / (hi - lo + 1e-12)  # nearer => higher confidence
    except Exception:  # noqa: BLE001
        pass

    clusters = query.obs[args.cluster_key].astype(str).to_numpy()
    out = _aggregate(clusters, pred, conf)
    wd = resolve_workdir(args.workdir)
    save_table(out, wd, f"refmap_{args.level}_{safe_cluster_id(args.label_col)}.csv")
    eprint(f">> done. {len(out)} clusters (zero-shot — treat as a hypothesis).")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", "--adata", dest="query", required=True,
                    help="the query AnnData to annotate (alias: --adata)")
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", default="L1")
    ap.add_argument("--model-path", required=True, help="downloaded SCimilarity model dir")
    ap.add_argument("--counts-layer", default="X",
                    help="query layer holding RAW COUNTS ('X' if X is counts)")
    ap.add_argument("--label-col", default="scimilarity",
                    help="name used in the output table filename")
    return ap.parse_args()


if __name__ == "__main__":
    main()
