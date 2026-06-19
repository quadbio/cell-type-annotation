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

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from cta.io import (aggregate_predictions, eprint, load_adata, require_obs, resolve_expression,
                 resolve_workdir, safe_cluster_id, save_table, split_csv)

MIN_SHARED_GENES = 30


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
    do_norm = ref_counts_flag or _looks_like_counts(ref.X)
    if do_norm:
        eprint("   reference looks like counts -> normalize_total(1e4) + log1p")
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)
    else:
        eprint("   reference treated as already log-normalized (use --ref-counts to force)")


def _subsample_indices(obs, n_obs: int, max_ref: int, stratify_col: str | None):
    """Fixed-seed indices for a <= max_ref subsample, stratified if a column is given."""
    rng = np.random.default_rng(0)
    if stratify_col is not None:
        if stratify_col not in obs:
            raise SystemExit(f"ERROR: stratify column '{stratify_col}' not in reference obs")
        groups = obs[stratify_col].astype(str).values
        cats, counts = np.unique(groups, return_counts=True)
        alloc = np.maximum(1, np.round(counts / counts.sum() * max_ref).astype(int))
        picks = []
        for cat, n_take in zip(cats, alloc):
            idx = np.flatnonzero(groups == cat)
            picks.append(rng.choice(idx, size=min(n_take, idx.size), replace=False))
        sel = np.concatenate(picks)
        if sel.size > max_ref:
            sel = rng.choice(sel, size=max_ref, replace=False)
        eprint(f"   stratified subsample by '{stratify_col}': {n_obs} -> {sel.size} cells")
    else:
        sel = rng.choice(n_obs, size=max_ref, replace=False)
        eprint(f"   random subsample: {n_obs} -> {sel.size} cells")
    return np.sort(sel)


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


def _run_cellmapper(query, reference, label_cols, args):
    """Transfer each label column onto query.obs via CellMapper. Returns method tag."""
    from cellmapper import CellMapper

    cmap = CellMapper(query, reference)
    eprint(f">> building joint embedding via '{args.embedding_method}'")
    if args.embedding_method == "cca":
        cmap.compute_fast_cca(n_comps=args.n_comps, key_added="X_joint")
    else:
        cmap.compute_joint_pca(n_comps=args.n_comps, key_added="X_joint")

    eprint(f">> computing neighbors (knn_method='{args.knn_method}')")
    cmap.compute_neighbors(use_rep="X_joint", knn_method=args.knn_method, random_state=0)
    cmap.compute_mapping_matrix()

    for col in label_cols:
        eprint(f">> transferring reference.obs['{col}'] -> query")
        cmap.map_obs(col, prediction_postfix="_pred", confidence_postfix="_conf")
    return "cellmapper"


def _run_correlation_fallback(query, reference, label_cols, args, shared):
    """Centroid-Spearman fallback: write transferred label + a pseudo-confidence
    onto query.obs so the same aggregation path can be reused."""
    from scipy.stats import rankdata

    eprint(">> cellmapper unavailable -> centroid-Spearman correlation fallback")
    # Mean-expression centroids over shared genes (cells x shared_genes).
    q = query[:, shared]
    r = reference[:, shared]
    Xq = q.X.toarray() if issparse(q.X) else np.asarray(q.X)
    Xr = r.X.toarray() if issparse(r.X) else np.asarray(r.X)

    qclus = query.obs[args.cluster_key].astype(str).values
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
        save_table(pd.DataFrame(rows), resolve_workdir(args.workdir),
                   f"refmap_{args.level}_{safe_cluster_id(col)}.csv")
    return "correlation"


def main():
    args = _parse_args()
    wd = resolve_workdir(args.workdir)
    label_cols = split_csv(args.label_cols)
    if not label_cols:
        raise SystemExit("ERROR: --label-cols must list at least one column")

    eprint(">> loading query")
    query = load_adata(args.query)
    require_obs(query, args.cluster_key, "cluster key")

    eprint(">> loading reference (backed) + subsample-before-materialize")
    reference = _read_reference_subsampled(args.reference, label_cols, args.max_ref,
                                           args.stratify_col)

    # Intersect var_names -> shared genes; subset both.
    shared = _shared_genes(query, reference)
    query = query[:, shared].copy()
    reference = reference[:, shared].copy()

    # Query expression: pin .X to the chosen lognorm matrix.
    layer = None if args.query_lognorm_layer in (None, "X") else args.query_lognorm_layer
    query.X = resolve_expression(query, layer)

    # Reference normalization to match the query's lognorm space.
    _normalize_reference(reference, args.ref_counts)

    try:
        method = _run_cellmapper(query, reference, label_cols, args)
    except ImportError:
        _run_correlation_fallback(query, reference, label_cols, args, shared)
        eprint(">> done (correlation fallback)")
        return

    # Aggregate the transferred per-cell labels per query cluster.
    clusters = query.obs[args.cluster_key].astype(str).values
    for col in label_cols:
        pred = query.obs[f"{col}_pred"].astype(str).values
        conf = np.asarray(query.obs[f"{col}_conf"].values, dtype=float)
        out = aggregate_predictions(clusters, pred, conf, method)
        save_table(out, wd, f"refmap_{args.level}_{safe_cluster_id(col)}.csv")
    eprint(">> done")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # This script takes TWO objects, so the annotated one is "--query"; "--adata" is
    # accepted as an alias so the flag matches every other script in the suite.
    ap.add_argument("--query", "--adata", dest="query", required=True,
                    help="the query AnnData to annotate (alias: --adata)")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--label-cols", required=True,
                    help="comma-separated reference.obs columns to transfer")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", default="L1")
    ap.add_argument("--query-lognorm-layer", "--lognorm-layer", dest="query_lognorm_layer",
                    default=None,
                    help="query layer holding log1p data ('X' or a layer name); "
                         "alias: --lognorm-layer")
    ap.add_argument("--ref-counts", action="store_true",
                    help="force normalize_total(1e4)+log1p on the reference")
    ap.add_argument("--embedding-method", choices=["cca", "pca"], default="cca")
    ap.add_argument("--n-comps", type=int, default=50)
    ap.add_argument("--knn-method", default="pynndescent",
                    help="CellMapper kNN backend. CPU: 'pynndescent' (default), 'sklearn', "
                         "'faiss-cpu'. GPU: 'rapids' (seconds over millions of cells; needs "
                         "cuml-cu12 from the gpu env, `pixi install -e gpu`) or 'faiss-gpu' (needs a working "
                         "faiss-gpu build — prefer the conda one; the pip wheel is often ABI-"
                         "broken). On a GPU profile use 'rapids' to scale this cross-check.")
    ap.add_argument("--max-ref", type=int, default=200000)
    ap.add_argument("--stratify-col", default=None,
                    help="reference.obs column to stratify the subsample by")
    return ap.parse_args()


if __name__ == "__main__":
    main()
