#!/usr/bin/env python
"""Investigate-before-Unknown: profile clusters by QC, composition, and spatial spread.

Before falling back to an "Unknown" / "ambiguous" label, profile each cluster so the
annotator can tell a real population from low-quality, artifact, or early-state cells.
Three independent profiles, each written as its own table:

  1. QC numeric profile  -- per-cluster median of every QC metric, plus a __global__
     median row, so an off-population cluster (low counts, high mito, ...) stands out.
  2. Categorical composition -- a row-normalized (%) crosstab of cluster x category,
     for EACH user-named obs column (e.g. an unexpected batch/timepoint skew can
     reveal an early-state population rather than an artifact).
  3. Spatial profile (optional) -- per-cluster centroid and spread from an obsm key,
     to catch spatially diffuse vs. focal clusters.

NO-BIAS RULE: QC columns are auto-detected purely by numeric dtype (never by name);
categorical crosstabs are built ONLY for columns the user explicitly names. Nothing
like 'week'/'condition'/'slide'/'n_genes' is ever hardcoded as a default.

Usage:
  python profile_clusters.py --adata path.h5ad --cluster-key leiden \
      --workdir DIR --level L1 \
      [--qc-cols n_genes,total_counts,pct_mito] \
      [--cat-cols sample,condition] \
      [--spatial-key spatial]
"""
from __future__ import annotations

import argparse

import pandas as pd
from pandas.api.types import is_numeric_dtype

from cta.io import (
    eprint,
    load_adata,
    require_obs,
    resolve_workdir,
    save_table,
    split_csv,
)


def _auto_qc_cols(adata, cluster_key: str) -> list[str]:
    """Numeric obs columns (by dtype), excluding the cluster key. No name heuristics."""
    cols = []
    for c in adata.obs.columns:
        if c == cluster_key:
            continue
        if is_numeric_dtype(adata.obs[c]):
            cols.append(c)
    return cols


def qc_profile(adata, cluster_key: str, qc_cols: list[str], workdir, level: str):
    if not qc_cols:
        eprint("   QC profile: no numeric obs columns found; skipping.")
        return
    obs = adata.obs[[cluster_key] + qc_cols].copy()
    per_cluster = obs.groupby(cluster_key, observed=True)[qc_cols].median()
    per_cluster.index = per_cluster.index.astype(str)
    glob = obs[qc_cols].median()
    glob.name = "__global__"
    out = pd.concat([per_cluster, glob.to_frame().T])
    out.index.name = cluster_key
    save_table(out, workdir, f"qc_profile_{level}.csv", index=True)


def categorical_composition(adata, cluster_key: str, cat_cols: list[str], workdir, level: str):
    if not cat_cols:
        eprint("   Categorical composition: no --cat-cols given; skipping "
               "(categorical cols are never auto-detected).")
        return
    for col in cat_cols:
        require_obs(adata, col, what="--cat-cols column")
        ct = pd.crosstab(
            adata.obs[cluster_key].astype(str),
            adata.obs[col].astype(str),
            normalize="index",
        ) * 100.0
        ct.index.name = cluster_key
        save_table(ct, workdir, f"crosstab_{level}_{col}.csv", index=True)


def spatial_profile(adata, cluster_key: str, spatial_key: str | None, workdir, level: str):
    if not spatial_key:
        return
    if spatial_key not in adata.obsm:
        eprint(f"   Spatial profile: obsm key '{spatial_key}' not found; skipping. "
               f"Available: {list(adata.obsm.keys())}")
        return
    coords = adata.obsm[spatial_key]
    if coords.shape[1] < 2:
        eprint(f"   Spatial profile: obsm['{spatial_key}'] has <2 dims; skipping.")
        return
    df = pd.DataFrame(
        {"x": coords[:, 0], "y": coords[:, 1], cluster_key: adata.obs[cluster_key].astype(str).values}
    )
    grp = df.groupby(cluster_key, observed=True)
    out = pd.DataFrame({
        "centroid_x": grp["x"].mean(),
        "centroid_y": grp["y"].mean(),
        "spread_x": grp["x"].std(),
        "spread_y": grp["y"].std(),
        "n_cells": grp.size(),
    })
    out.index.name = cluster_key
    save_table(out, workdir, f"spatial_profile_{level}.csv", index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--qc-cols", help="comma-separated numeric obs columns; auto-detected if omitted")
    ap.add_argument("--cat-cols", help="comma-separated categorical obs columns (user-named only)")
    ap.add_argument("--spatial-key", help="adata.obsm key holding spatial coordinates")
    args = ap.parse_args()

    adata = load_adata(args.adata, backed="r")  # only obs/obsm needed — don't materialize X
    require_obs(adata, args.cluster_key, what="cluster key")
    workdir = resolve_workdir(args.workdir)

    if args.qc_cols:
        qc_cols = split_csv(args.qc_cols)
        for c in qc_cols:
            require_obs(adata, c, what="--qc-cols column")
    else:
        qc_cols = _auto_qc_cols(adata, args.cluster_key)
        eprint(f"   auto-detected {len(qc_cols)} numeric QC cols: {qc_cols}")

    cat_cols = split_csv(args.cat_cols)

    eprint(">> QC numeric profile")
    qc_profile(adata, args.cluster_key, qc_cols, workdir, args.level)
    eprint(">> categorical composition")
    categorical_composition(adata, args.cluster_key, cat_cols, workdir, args.level)
    eprint(">> spatial profile")
    spatial_profile(adata, args.cluster_key, args.spatial_key, workdir, args.level)


if __name__ == "__main__":
    main()
