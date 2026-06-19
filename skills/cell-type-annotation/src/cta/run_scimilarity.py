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

import numpy as np
import typer

from cta.io import (aggregate_predictions, eprint, require_obs, resolve_workdir, safe_cluster_id,
                    save_table)


def cli(
    query: str = typer.Option(
        ..., "--query", "--adata",
        help="the query AnnData to annotate (alias: --adata)"),
    cluster_key: str = typer.Option(..., help="obs column holding the clusters"),
    workdir: str = typer.Option(..., help="annotation-run working directory"),
    level: str = typer.Option("L1", help="hierarchy level tag for filenames"),
    model_path: str = typer.Option(..., help="downloaded SCimilarity model dir"),
    counts_layer: str = typer.Option(
        "X", help="query layer holding RAW COUNTS ('X' if X is counts)"),
    label_col: str = typer.Option(
        "scimilarity", help="name used in the output table filename"),
) -> None:
    """Optional zero-shot cross-check via SCimilarity — a large-memory hypothesis (not a verdict)."""
    import scanpy as sc

    try:
        from scimilarity import CellAnnotation, align_dataset, lognorm_counts
    except ImportError:
        raise SystemExit(
            "ERROR: scimilarity is not installed. It is in the skill's `gpu` pixi env — run "
            "`pixi install -e gpu`, then download the model bundle (tens of GB; needs "
            "~64GB RAM at runtime)."
        )

    eprint(f">> loading query {query}")
    query_obj = sc.read_h5ad(query)
    require_obs(query_obj, cluster_key, "cluster key")

    # SCimilarity wants raw counts in X.
    if counts_layer not in (None, "X"):
        if counts_layer not in (query_obj.layers or {}):
            raise SystemExit(f"ERROR: --counts-layer '{counts_layer}' not in "
                             f"query.layers {list(query_obj.layers.keys())}.")
        query_obj.X = query_obj.layers[counts_layer]

    eprint(f">> loading SCimilarity model: {model_path}")
    ca = CellAnnotation(model_path=model_path)

    eprint(">> aligning genes to model gene order + log-normalizing")
    query_obj = align_dataset(query_obj, ca.gene_order)
    query_obj = lognorm_counts(query_obj)

    eprint(">> embedding + kNN annotation")
    query_obj.obsm["X_scimilarity"] = ca.get_embeddings(query_obj.X)
    result = ca.get_predictions_knn(query_obj.obsm["X_scimilarity"])
    # API returns (predictions, nn_idxs, nn_dists, nn_stats); be tolerant of arity.
    predictions = result[0] if isinstance(result, (tuple, list)) else result
    pred = np.asarray(getattr(predictions, "values", predictions)).astype(str)

    # Per-cell confidence: invert min nearest-neighbor distance into 0-1 (relative within
    # this query), if distances are available; else leave blank.
    conf = np.full(query_obj.n_obs, np.nan)
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

    clusters = query_obj.obs[cluster_key].astype(str).to_numpy()
    out = aggregate_predictions(clusters, pred, conf, "scimilarity")
    wd = resolve_workdir(workdir)
    save_table(out, wd, f"refmap_{level}_{safe_cluster_id(label_col)}.csv")
    eprint(f">> done. {len(out)} clusters (zero-shot — treat as a hypothesis).")
