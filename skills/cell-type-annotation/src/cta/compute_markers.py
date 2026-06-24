"""Job A: per-cluster marker identification.

rank_genes_groups (Wilcoxon, tie_correct — NOT the t-test default) on log-normalized data
+ COSG specificity, written as tidy marker tables to <workdir>/tables/. The descriptive
marker evidence that feeds annotation. NOT condition DE (Job B = pseudobulk + PyDESeq2;
see reference/differential_expression.md).
"""
from __future__ import annotations

from typing import Optional

import typer

from cta.io import (
    ADATA_OPT, CLUSTER_KEY_OPT, FRAGMENTED_FRAME_WARNING, LEVEL_OPT, WORKDIR_OPT,
    adata_with_lognorm_X, eprint, load_adata, require_obs, resolve_workdir,
    save_table, split_csv, suppressed_warnings)


def cli(
    adata: str = ADATA_OPT,
    cluster_key: str = CLUSTER_KEY_OPT,
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    lognorm_layer: Optional[str] = typer.Option(
        None, help="layer with log-normalized data; default uses adata.X"),
    groups: Optional[str] = typer.Option(
        None, help="comma-separated subset of clusters; default all"),
    n_genes: int = typer.Option(50, help="top genes per cluster to keep"),
    no_cosg: bool = typer.Option(
        False, "--no-cosg/--cosg", help="skip COSG specificity scoring"),
) -> None:
    """Per-cluster markers: rank_genes_groups (Wilcoxon, tie_correct) + COSG specificity."""
    import pandas as pd
    import scanpy as sc

    wd = resolve_workdir(workdir)
    adata_obj = load_adata(adata)
    require_obs(adata_obj, cluster_key, "cluster key")

    # Use the log-normalized matrix as adata.X for the tests (don't mutate the file).
    adata_obj = adata_with_lognorm_X(adata_obj, lognorm_layer)

    adata_obj.obs[cluster_key] = adata_obj.obs[cluster_key].astype("category")
    grps = split_csv(groups) or "all"

    eprint(">> rank_genes_groups (wilcoxon, tie_correct=True, pts=True)")
    # scanpy assembles its results frame one column per group, so with many clusters pandas
    # raises 'DataFrame is highly fragmented' once per cluster. It is internal to scanpy
    # (scanpy/tools/_rank_genes_groups.py) and has no effect on the output — silence exactly
    # that message for this one call, leaving every other warning visible.
    with suppressed_warnings(FRAGMENTED_FRAME_WARNING):
        sc.tl.rank_genes_groups(
            adata_obj, groupby=cluster_key, groups=grps,
            method="wilcoxon", tie_correct=True, pts=True, use_raw=False,
        )
    frames = []
    grp_list = adata_obj.obs[cluster_key].cat.categories if grps == "all" else grps
    for g in grp_list:
        df = sc.get.rank_genes_groups_df(adata_obj, group=str(g)).head(n_genes)
        df.insert(0, "cluster", str(g))
        frames.append(df)
    save_table(pd.concat(frames, ignore_index=True), wd, f"markers_{level}.csv")

    if not no_cosg:
        try:
            import cosg
            eprint(">> COSG (cosine-similarity specificity)")
            cosg.cosg(adata_obj, groupby=cluster_key, key_added="cosg",
                      mu=1.0, n_genes_user=n_genes)
            names = pd.DataFrame(adata_obj.uns["cosg"]["names"])
            scores = pd.DataFrame(adata_obj.uns["cosg"]["scores"])
            rows = []
            for col in names.columns:
                for rank, (gene, score) in enumerate(zip(names[col], scores[col])):
                    rows.append({"cluster": str(col), "rank": rank,
                                 "gene": gene, "cosg_score": float(score)})
            save_table(pd.DataFrame(rows), wd, f"cosg_{level}.csv")
        except Exception as e:  # noqa: BLE001 — tool-level degradation is OK
            eprint(f"   COSG skipped ({type(e).__name__}: {e})")

    eprint(">> done")
