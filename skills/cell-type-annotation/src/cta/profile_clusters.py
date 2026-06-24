"""Investigate-before-Unknown: profile clusters by QC, composition, and spatial spread.

Before falling back to an "Unknown" / "ambiguous" label, profile each cluster so the
annotator can tell a real population from low-quality, artifact, or early-state cells.
Three independent profiles, each written as its own table:

  1. QC numeric profile  -- per-cluster median of every QC metric, plus a __global__
     median row, so an off-population cluster (low counts, high mito, ...) stands out.
     A companion `qc_deviation_<level>.csv` expresses each cluster median as a robust
     (MAD) z vs the across-cluster norm, so low-quality / cycling clusters pop out.
  2. Categorical composition -- a row-normalized (%) crosstab of cluster x category,
     for EACH user-named obs column (e.g. an unexpected batch/timepoint skew can
     reveal an early-state population rather than an artifact).
  3. Spatial profile (optional) -- per-cluster centroid and spread from an obsm key,
     to catch spatially diffuse vs. focal clusters.

NO-BIAS RULE: QC columns are auto-detected purely by numeric dtype (never by name);
categorical crosstabs are built ONLY for columns the user explicitly names. Nothing
like 'week'/'condition'/'slide'/'n_genes' is ever hardcoded as a default.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import typer
from pandas.api.types import is_numeric_dtype

from cta.io import (
    ADATA_OPT, CLUSTER_KEY_OPT, WORKDIR_OPT, eprint, load_adata, require_obs,
    resolve_workdir, save_table, split_csv)


def _auto_qc_cols(adata, cluster_key: str) -> list[str]:
    """Numeric obs columns (by dtype), excluding the cluster key. No name heuristics."""
    cols = []
    for c in adata.obs.columns:
        if c == cluster_key:
            continue
        if is_numeric_dtype(adata.obs[c]):
            cols.append(c)
    return cols


def _robust_z(per_cluster: pd.DataFrame) -> pd.DataFrame:
    """Per-column robust z of each cluster's median vs the across-cluster spread:
    (median - median_over_clusters) / (1.4826 * MAD). Name-agnostic — applied uniformly to
    every numeric column, so an off-population cluster (low depth, high mito, high cell-cycle
    score) stands out as a large-magnitude z regardless of which metric it is. Columns with no
    spread (MAD == 0) yield 0, not inf/nan."""
    center = per_cluster.median()
    mad = (per_cluster - center).abs().median() * 1.4826
    z = (per_cluster - center).divide(mad.where(mad > 0))
    return z.where(mad > 0, 0.0).round(2)


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

    # Companion deviation table: how far each cluster's median sits from the across-cluster
    # norm, in robust (MAD) units — so low-quality / cycling clusters pop out without the
    # annotator eyeballing absolute medians. Needs >=3 clusters for a meaningful spread.
    if per_cluster.shape[0] >= 3:
        dev = _robust_z(per_cluster)
        dev.index.name = cluster_key
        save_table(dev, workdir, f"qc_deviation_{level}.csv", index=True)
    else:
        eprint("   QC deviation: <3 clusters; skipping robust-z table.")


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


def cli(
    adata: str = ADATA_OPT,
    cluster_key: str = CLUSTER_KEY_OPT,
    workdir: str = WORKDIR_OPT,
    level: str = typer.Option(..., help="hierarchy level tag for filenames"),
    qc_cols: Optional[str] = typer.Option(
        None, help="comma-separated numeric obs columns; auto-detected if omitted"),
    cat_cols: Optional[str] = typer.Option(
        None, help="comma-separated categorical obs columns (user-named only)"),
    spatial_key: Optional[str] = typer.Option(
        None, help="adata.obsm key holding spatial coordinates"),
) -> None:
    """Profile clusters by QC numeric medians, categorical composition, and spatial spread."""
    adata_obj = load_adata(adata, backed="r")  # only obs/obsm needed — don't materialize X
    require_obs(adata_obj, cluster_key, what="cluster key")
    workdir_path = resolve_workdir(workdir)

    if qc_cols:
        qc_col_list = split_csv(qc_cols)
        for c in qc_col_list:
            require_obs(adata_obj, c, what="--qc-cols column")
    else:
        qc_col_list = _auto_qc_cols(adata_obj, cluster_key)
        eprint(f"   auto-detected {len(qc_col_list)} numeric QC cols: {qc_col_list}")

    cat_col_list = split_csv(cat_cols)

    eprint(">> QC numeric profile")
    qc_profile(adata_obj, cluster_key, qc_col_list, workdir_path, level)
    eprint(">> categorical composition")
    categorical_composition(adata_obj, cluster_key, cat_col_list, workdir_path, level)
    eprint(">> spatial profile")
    spatial_profile(adata_obj, cluster_key, spatial_key, workdir_path, level)
