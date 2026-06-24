#!/usr/bin/env python
"""Optional zero-shot cross-check via SCimilarity — a HYPOTHESIS (not a verdict).

SCimilarity embeds cells against a large pretrained reference (~7M cells) and returns a
per-cell nearest-neighbor cell-type call. HUMAN-ONLY: the v1.1 model has a fixed human gene
order, so this is for HUMAN queries — on mouse it is weak even with ortholog mapping (skip it).
It is a foundation-model annotator, and benchmarks show such models rarely beat scANVI / simple
baselines, so this is strictly an ADDITIONAL cross-check to reconcile against marker evidence,
never a primary annotator (reference/automated_tools.md). Its real value is on BROAD / multi-
tissue human queries where CellTypist/SingleR have no tissue-matched model and collapse.

Cost (measured on celltype-anno-bench, 2026): ~16-27 GB RAM, CPU, a few minutes; the annotation
path needs only the ~9 GB annotation-only bundle, NOT the full ~41 GB download.

CANONICAL SHAPE. The query is expected in the skill's canonical form (`cta data canonicalize`):
RAW COUNTS in `layers['counts']`, log-norm in `X`, gene SYMBOLS as var_names. SCimilarity needs
raw counts (it aligns genes to the model order, then applies its own tp10k log-norm), so this
reads counts from `layers['counts']` by default (--counts-layer); symbol var_names are what
align_dataset matches against the model's gene order.

It writes the SAME per-cluster schema reference_map.py emits, so the cards/report pick it
up as one more cross-check automatically.

Usage:
  python run_scimilarity.py --query Q.h5ad --cluster-key leiden --workdir DIR --level L1 \
      --model-path /path/to/scimilarity_model [--counts-layer counts] [--label-col scimilarity]

Writes tables/refmap_<level>_<label-col>.csv with columns:
  cluster, top1, frac1, top2, frac2, mean_conf, method
"""
from __future__ import annotations

import numpy as np
import typer

from cta.io import (
    CLUSTER_KEY_OPT, LEVEL_OPT, WORKDIR_OPT, aggregate_predictions, eprint,
    load_adata, require_obs, resolve_workdir, safe_cluster_id, save_table)


def cli(
    query: str = typer.Option(
        ..., "--query", "--adata",
        help="the query AnnData to annotate (alias: --adata)"),
    cluster_key: str = CLUSTER_KEY_OPT,
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    model_path: str = typer.Option(..., help="downloaded SCimilarity model dir"),
    counts_layer: str = typer.Option(
        "counts", help="layer with RAW COUNTS — the canonical 'counts' layer by default "
        "(pass 'X' only if X itself is counts)"),
    label_col: str = typer.Option(
        "scimilarity", help="name used in the output table filename"),
) -> None:
    """Optional zero-shot cross-check via SCimilarity — a hypothesis (not a verdict). HUMAN-only."""
    try:
        from scimilarity import CellAnnotation, align_dataset, lognorm_counts
    except ImportError:
        raise SystemExit(
            "ERROR: scimilarity is not installed. It is in the skill's `gpu` pixi env — run "
            "`pixi install -e gpu`, then download the model bundle (~9 GB annotation-only "
            "subset is enough; needs ~16-27 GB RAM at runtime, CPU is fine)."
        )

    eprint(f">> loading query {query}")
    query_obj = load_adata(query)   # canonical: counts in layers['counts'], symbols as var_names
    require_obs(query_obj, cluster_key, "cluster key")

    # SCimilarity needs RAW COUNTS in layers['counts']: align_dataset carries that layer
    # through its gene-space concat, and lognorm_counts reads counts from there (NOT X) and
    # writes its own tp10k log-norm into X. The canonical query already keeps counts in
    # layers['counts']; --counts-layer X copies X across for a non-canonical input.
    src = query_obj.X if counts_layer in (None, "X") else query_obj.layers.get(counts_layer)
    if src is None:
        raise SystemExit(
            f"ERROR: --counts-layer '{counts_layer}' not in query.layers "
            f"{list(query_obj.layers.keys())}. Run `cta data canonicalize` so raw counts land "
            "in layers['counts'] (and var_names become gene symbols), then retry.")
    query_obj.layers["counts"] = src

    eprint(f">> loading SCimilarity model: {model_path}")
    ca = CellAnnotation(model_path=model_path)

    eprint(">> aligning genes to model gene order + log-normalizing")
    query_obj = align_dataset(query_obj, ca.gene_order)
    query_obj = lognorm_counts(query_obj)

    eprint(">> embedding + kNN annotation")
    query_obj.obsm["X_scimilarity"] = ca.get_embeddings(query_obj.X)
    # weighting=True: distance-weighted kNN vote (the unconstrained-annotation default in
    # SCimilarity's own tutorial); reduces noise vs an unweighted majority.
    result = ca.get_predictions_knn(query_obj.obsm["X_scimilarity"], weighting=True)
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
