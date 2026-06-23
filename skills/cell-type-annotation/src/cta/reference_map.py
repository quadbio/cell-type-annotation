#!/usr/bin/env python
"""CellMapper reference label-transfer as a cross-check HYPOTHESIS (not a verdict).

Transfers one or more reference ``obs`` label columns onto the query via a joint
embedding + k-NN, then aggregates the per-cell transferred labels per query cluster.
The result is a hypothesis to reconcile against marker evidence, never an overwrite:
a targeted/limited panel shrinks the shared gene space and degrades transfer, so
treat low shared-gene counts and low mean confidence with suspicion.

If ``cellmapper`` cannot be imported, falls back to a transparent
centroid-Spearman correlation between per-reference-label and per-query-cluster
mean-expression profiles over the shared genes, writing the same table schema with
``method=correlation`` so downstream consumers can tell the two apart.

Usage:
  python reference_map.py --query Q.h5ad --reference R.h5ad --cluster-key leiden \
      --label-cols cell_type,subtype --workdir /proj/run --level L1 \
      [--query-lognorm-layer X|lognorm] [--ref-counts] \
      [--embedding-method cca|pca] [--n-comps 50] \
      [--knn-method pynndescent] [--max-ref 200000] [--stratify-col donor]

Per --label-cols column, writes tables/refmap_<level>_<col>.csv with columns:
  cluster, top1, frac1, top2, frac2, mean_conf, method
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from scipy.sparse import issparse

from cta.constants import MIN_SHARED_GENES, NORMALIZE_TARGET_SUM
from cta.io import (
    CLUSTER_KEY_OPT, LEVEL_OPT, WORKDIR_OPT, aggregate_predictions, eprint,
    load_adata, require_obs, resolve_expression, resolve_workdir,
    safe_cluster_id, save_table, split_csv, stratified_subsample_positions)


def _looks_like_counts(X) -> bool:
    """Heuristic: integer-valued, non-negative matrix => raw counts."""
    sample = X[:200] if X.shape[0] > 200 else X
    arr = sample.toarray() if issparse(sample) else np.asarray(sample)
    if arr.size == 0:
        return False
    if arr.min() < 0:
        return False
    return bool(np.allclose(arr, np.round(arr)))


def _normalize_reference(ref, ref_counts_flag: bool) -> None:
    """log1p of 1e4-normalized counts, to match the query's lognorm space."""
    import scanpy as sc

    do_norm = ref_counts_flag or _looks_like_counts(ref.X)
    if do_norm:
        eprint("   reference looks like counts -> normalize_total(1e4) + log1p")
        sc.pp.normalize_total(ref, target_sum=NORMALIZE_TARGET_SUM)
        sc.pp.log1p(ref)
    else:
        eprint("   reference treated as already log-normalized (use --ref-counts to force)")


def _subsample_indices(obs, n_obs: int, max_ref: int, stratify_col: str | None):
    """Fixed-seed row positions for a <= max_ref subsample, stratified if a column is given."""
    groups = None
    if stratify_col is not None:
        if stratify_col not in obs:
            raise SystemExit(f"ERROR: stratify column '{stratify_col}' not in reference obs")
        groups = obs[stratify_col].astype(str).values
    sel = stratified_subsample_positions(n_obs, max_ref, groups)
    kind = f"stratified subsample by '{stratify_col}'" if stratify_col else "random subsample"
    eprint(f"   {kind}: {n_obs} -> {sel.size} cells")
    return sel


def _read_reference_subsampled(path, label_cols, max_ref: int, stratify_col: str | None):
    """Read the reference in BACKED mode and subsample BEFORE materializing.

    Large references (e.g. a 1.7M-cell atlas) should never be fully loaded just to take
    a 30k subsample. obs is available in backed mode, so we pick indices from it and only
    then `.to_memory()`.
    """
    import anndata as ad
    p = Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"ERROR: reference not found: {p}")
    ref = ad.read_h5ad(p, backed="r")
    for col in label_cols:
        require_obs(ref, col, "label column")
    if ref.n_obs > max_ref:
        sel = _subsample_indices(ref.obs, ref.n_obs, max_ref, stratify_col)
        ref = ref[sel].to_memory()
    else:
        ref = ref.to_memory()
    return ref




def _shared_genes(query, reference):
    shared = [g for g in query.var_names if g in set(reference.var_names)]
    if len(shared) < MIN_SHARED_GENES:
        raise SystemExit(
            f"ERROR: only {len(shared)} shared genes between query and reference "
            f"(need >= {MIN_SHARED_GENES}). Check that var_names use the same "
            "gene identifier convention in both objects."
        )
    eprint(f"   {len(shared)} shared genes between query and reference")
    return shared


def _run_cellmapper(query, reference, label_cols, embedding_method, n_comps, knn_method):
    """Transfer each label column onto query.obs via CellMapper. Returns method tag."""
    from cellmapper import CellMapper

    cmap = CellMapper(query, reference)
    eprint(f">> building joint embedding via '{embedding_method}'")
    if embedding_method == "cca":
        cmap.compute_fast_cca(n_comps=n_comps, key_added="X_joint")
    else:
        cmap.compute_joint_pca(n_comps=n_comps, key_added="X_joint")

    eprint(f">> computing neighbors (knn_method='{knn_method}')")
    cmap.compute_neighbors(use_rep="X_joint", knn_method=knn_method, random_state=0)
    cmap.compute_mapping_matrix()

    for col in label_cols:
        eprint(f">> transferring reference.obs['{col}'] -> query")
        cmap.map_obs(col, prediction_postfix="_pred", confidence_postfix="_conf")
    return "cellmapper"


def _run_correlation_fallback(query, reference, label_cols, cluster_key, workdir, level, shared):
    """Centroid-Spearman fallback: write transferred label + a pseudo-confidence
    onto query.obs so the same aggregation path can be reused."""
    from scipy.stats import rankdata

    eprint(">> cellmapper unavailable -> centroid-Spearman correlation fallback")
    # Mean-expression centroids over shared genes (cells x shared_genes).
    q = query[:, shared]
    r = reference[:, shared]
    Xq = q.X.toarray() if issparse(q.X) else np.asarray(q.X)
    Xr = r.X.toarray() if issparse(r.X) else np.asarray(r.X)

    qclus = query.obs[cluster_key].astype(str).values
    # Per-query-cluster centroid, rank-transformed for Spearman.
    qcats = np.unique(qclus)
    qcent = np.vstack([Xq[qclus == c].mean(axis=0) for c in qcats])
    qrank = np.vstack([rankdata(row) for row in qcent])

    for col in label_cols:
        rlab = reference.obs[col].astype(str).values
        rcats = np.unique(rlab)
        rcent = np.vstack([Xr[rlab == c].mean(axis=0) for c in rcats])
        rrank = np.vstack([rankdata(row) for row in rcent])
        # Spearman = Pearson on ranks: corr matrix (q_clusters x r_labels).
        qz = qrank - qrank.mean(axis=1, keepdims=True)
        rz = rrank - rrank.mean(axis=1, keepdims=True)
        qn = qz / (np.linalg.norm(qz, axis=1, keepdims=True) + 1e-12)
        rn = rz / (np.linalg.norm(rz, axis=1, keepdims=True) + 1e-12)
        corr = qn @ rn.T  # (n_qclusters x n_rlabels)
        order = np.argsort(-corr, axis=1)
        rows = []
        for i, cl in enumerate(qcats):
            o = order[i]
            top1 = rcats[o[0]]
            frac1 = round(float(corr[i, o[0]]), 4)  # correlation as the "fraction"
            if o.size > 1:
                top2 = rcats[o[1]]
                frac2 = round(float(corr[i, o[1]]), 4)
            else:
                top2, frac2 = "", 0.0
            rows.append({
                "cluster": cl, "top1": top1, "frac1": frac1,
                "top2": top2, "frac2": frac2,
                "mean_conf": frac1, "method": "correlation",
            })
        save_table(pd.DataFrame(rows), resolve_workdir(workdir),
                   f"refmap_{level}_{safe_cluster_id(col)}.csv")
    return "correlation"


def cli(
    query: str = typer.Option(
        ..., "--query", "--adata",
        help="the query AnnData to annotate (alias: --adata)"),
    reference: str = typer.Option(..., help="reference AnnData with label columns"),
    cluster_key: str = CLUSTER_KEY_OPT,
    label_cols: str = typer.Option(
        ..., help="comma-separated reference.obs columns to transfer"),
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    query_lognorm_layer: Optional[str] = typer.Option(
        None, "--query-lognorm-layer", "--lognorm-layer",
        help="query layer holding log1p data ('X' or a layer name); alias: --lognorm-layer"),
    ref_counts: bool = typer.Option(
        False, "--ref-counts",
        help="force normalize_total(1e4)+log1p on the reference"),
    embedding_method: str = typer.Option("cca", help="joint embedding method: cca or pca"),
    n_comps: int = typer.Option(50, help="number of embedding components"),
    knn_method: str = typer.Option(
        "pynndescent",
        help="CellMapper kNN backend. CPU: 'pynndescent' (default), 'sklearn', "
             "'faiss-cpu'. GPU: 'rapids' (seconds over millions of cells; needs "
             "cuml-cu12 from the gpu env, `pixi install -e gpu`) or 'faiss-gpu' (needs a working "
             "faiss-gpu build — prefer the conda one; the pip wheel is often ABI-"
             "broken). On a GPU profile use 'rapids' to scale this cross-check."),
    max_ref: int = typer.Option(200000, help="max reference cells to use (subsampled)"),
    stratify_col: Optional[str] = typer.Option(
        None, help="reference.obs column to stratify the subsample by"),
) -> None:
    """CellMapper reference label-transfer as a cross-check hypothesis (not a verdict)."""
    wd = resolve_workdir(workdir)
    label_cols_list = split_csv(label_cols)
    if not label_cols_list:
        raise SystemExit("ERROR: --label-cols must list at least one column")

    eprint(">> loading query")
    query_obj = load_adata(query)
    require_obs(query_obj, cluster_key, "cluster key")

    eprint(">> loading reference (backed) + subsample-before-materialize")
    reference_obj = _read_reference_subsampled(reference, label_cols_list, max_ref,
                                               stratify_col)

    # Intersect var_names -> shared genes; subset both.
    shared = _shared_genes(query_obj, reference_obj)
    query_obj = query_obj[:, shared].copy()
    reference_obj = reference_obj[:, shared].copy()

    # Query expression: pin .X to the chosen lognorm matrix.
    layer = None if query_lognorm_layer in (None, "X") else query_lognorm_layer
    query_obj.X = resolve_expression(query_obj, layer)

    # Reference normalization to match the query's lognorm space.
    _normalize_reference(reference_obj, ref_counts)

    try:
        method = _run_cellmapper(query_obj, reference_obj, label_cols_list,
                                 embedding_method, n_comps, knn_method)
    except ImportError:
        _run_correlation_fallback(query_obj, reference_obj, label_cols_list,
                                  cluster_key, workdir, level, shared)
        eprint(">> done (correlation fallback)")
        return

    # Aggregate the transferred per-cell labels per query cluster.
    clusters = query_obj.obs[cluster_key].astype(str).values
    for col in label_cols_list:
        pred = query_obj.obs[f"{col}_pred"].astype(str).values
        conf = np.asarray(query_obj.obs[f"{col}_conf"].values, dtype=float)
        out = aggregate_predictions(clusters, pred, conf, method)
        save_table(out, wd, f"refmap_{level}_{safe_cluster_id(col)}.csv")
    eprint(">> done")
