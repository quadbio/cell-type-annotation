"""Job A: per-cluster marker identification.

rank_genes_groups (Wilcoxon, scanpy defaults — NOT the t-test default) on log-normalized data
+ a `filter_rank_genes_groups` specificity pass, written as tidy marker tables to
<workdir>/tables/. The descriptive marker evidence that feeds annotation. NOT condition DE
(Job B = pseudobulk + PyDESeq2; see reference/differential_expression.md).

The Wilcoxon ranking is compute-profile-routed (no user knob): on a CUDA GPU it runs via
rapids-singlecell (atlas-scale: seconds vs ~minutes), otherwise via scanpy on CPU — same
method and same `uns['rank_genes_groups']` output, so the specificity pass and every
downstream reader are identical on both paths.
"""

from __future__ import annotations

import typer

from cta import gpu
from cta.io import (
    ADATA_OPT,
    CLUSTER_KEY_OPT,
    FRAGMENTED_FRAME_WARNING,
    LEVEL_OPT,
    WORKDIR_OPT,
    adata_with_lognorm_X,
    eprint,
    load_adata,
    require_obs,
    resolve_workdir,
    save_table,
    split_csv,
    suppressed_warnings,
)


def cli(
    adata: str = ADATA_OPT,
    cluster_key: str = CLUSTER_KEY_OPT,
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    lognorm_layer: str | None = typer.Option(
        None, help="layer with log-normalized data; default uses adata.X"
    ),
    groups: str | None = typer.Option(None, help="comma-separated subset of clusters; default all"),
    n_genes: int = typer.Option(50, help="top genes per cluster to keep"),
) -> None:
    """Per-cluster markers: rank_genes_groups (Wilcoxon) + filter_rank_genes_groups specificity."""
    import pandas as pd
    import scanpy as sc

    wd = resolve_workdir(workdir)
    adata_obj = load_adata(adata)
    require_obs(adata_obj, cluster_key, "cluster key")

    # Use the log-normalized matrix as adata.X for the tests (don't mutate the file).
    adata_obj = adata_with_lognorm_X(adata_obj, lognorm_layer)

    adata_obj.obs[cluster_key] = adata_obj.obs[cluster_key].astype("category")
    grps = split_csv(groups) or "all"

    # Compute-profile routing via the central cta.gpu mechanism (no user knob): rapids-singlecell
    # on a CUDA GPU (atlas-scale: seconds vs ~minutes), else scanpy on CPU, with automatic CPU
    # fallback. Same wilcoxon method + `uns['rank_genes_groups']` either way, so everything
    # downstream (extraction, #36's filter_rank_genes_groups, reports) is identical.
    def _rgg_cpu(a):
        # scanpy assembles its results frame one column per group, so with many clusters pandas
        # raises 'DataFrame is highly fragmented' once per cluster. It is internal to scanpy
        # (scanpy/tools/_rank_genes_groups.py) and has no effect on the output — silence exactly
        # that message for this one call, leaving every other warning visible.
        with suppressed_warnings(FRAGMENTED_FRAME_WARNING):
            sc.tl.rank_genes_groups(
                a, groupby=cluster_key, groups=grps, method="wilcoxon", pts=True, use_raw=False
            )

    def _rgg_gpu(a, rsc):
        rsc.tl.rank_genes_groups(
            a, groupby=cluster_key, groups=grps, method="wilcoxon", pts=True, use_raw=False
        )

    gpu.run(adata_obj, cpu=_rgg_cpu, gpu=_rgg_gpu, label="rank_genes_groups (wilcoxon, pts=True)")
    frames = []
    grp_list = adata_obj.obs[cluster_key].cat.categories if grps == "all" else grps
    for g in grp_list:
        df = sc.get.rank_genes_groups_df(adata_obj, group=str(g)).head(n_genes)
        df.insert(0, "cluster", str(g))
        frames.append(df)
    save_table(pd.concat(frames, ignore_index=True), wd, f"markers_{level}.csv")

    # Specificity: keep only the cluster-specific markers via scanpy's own filter (replaces COSG).
    # filter_rank_genes_groups NaNs out genes that fail its in-group / out-group fraction + fold-
    # change thresholds (scanpy defaults), leaving the genes that genuinely distinguish each
    # cluster. We emit those as markers_filtered_<level>.csv (cluster, rank, gene, logfoldchange,
    # score, pct_in, pct_out) — the specificity table every downstream step reads in place of COSG.
    eprint(">> filter_rank_genes_groups (specificity)")
    sc.tl.filter_rank_genes_groups(
        adata_obj,
        key="rank_genes_groups",
        key_added="rank_genes_groups_filtered",
        use_raw=False,
    )
    filt_rows = []
    for g in grp_list:
        df = sc.get.rank_genes_groups_df(adata_obj, group=str(g), key="rank_genes_groups_filtered")
        df = df.dropna(subset=["names"]).head(n_genes).reset_index(drop=True)
        if df.empty:
            continue
        out = pd.DataFrame(
            {
                "cluster": str(g),
                "rank": range(len(df)),
                "gene": df["names"].astype(str),
                "logfoldchange": df["logfoldchanges"],
                "score": df["scores"],
                "pct_in": df.get("pct_nz_group"),
                "pct_out": df.get("pct_nz_reference"),
            }
        )
        filt_rows.append(out)
    filt_cols = ["cluster", "rank", "gene", "logfoldchange", "score", "pct_in", "pct_out"]
    filtered = (
        pd.concat(filt_rows, ignore_index=True) if filt_rows else pd.DataFrame(columns=filt_cols)
    )
    save_table(filtered, wd, f"markers_filtered_{level}.csv")

    eprint(">> done")
