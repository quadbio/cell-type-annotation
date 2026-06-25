"""Sub-cluster one or more existing clusters on the anchor embedding — standardized
cluster refinement (in scope: see SKILL.md). Use when a cluster is marker-incoherent
(e.g. mutually-exclusive markers co-expressed) or suspiciously large/heterogeneous.

Subsets the chosen clusters, builds a kNN graph on the SAME embedding used everywhere,
runs Leiden at a given resolution, and characterizes each sub-cluster by the fraction
of cells expressing a panel of marker genes (and per-sub QC / categorical composition
if asked). Writes a per-cell sub-assignment table + a per-sub summary to <workdir>/tables/.

This does NOT relabel anything — it produces evidence so the annotator can decide whether
a cluster splits into real sub-populations (and how to label them), or is homogeneous.

Either --genes (comma list) or --markers (JSON {type:[genes]}, flattened) supplies the
characterization panel; if neither is given, the top variable genes are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import typer

from cta.io import (
    ADATA_OPT,
    CLUSTER_KEY_OPT,
    LEVEL_OPT,
    WORKDIR_OPT,
    eprint,
    load_adata,
    read_markers_json,
    require_obs,
    resolve_expression,
    resolve_workdir,
    save_table,
    split_csv,
)


def _frac_expressing(adata, genes):
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    if not genes:
        return pd.DataFrame(index=adata.obs_names), []
    X = adata[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return pd.DataFrame(X > 0, columns=genes, index=adata.obs_names), genes


def cli(
    adata: str = ADATA_OPT,
    cluster_key: str = CLUSTER_KEY_OPT,
    embedding_key: str = typer.Option(
        ..., help="obsm key of the anchor embedding (neighbors built on this)"
    ),
    clusters: str = typer.Option(
        ..., help="comma-separated cluster id(s) to subcluster (pooled if >1)"
    ),
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    tag: str | None = typer.Option(
        None, help="short name for this refinement (filenames); default from --clusters"
    ),
    resolution: float = typer.Option(0.5, help="Leiden resolution"),
    n_neighbors: int = typer.Option(15, help="number of neighbors for the kNN graph"),
    genes: str | None = typer.Option(None, help="comma-separated characterization genes"),
    markers: str | None = typer.Option(
        None, help="markers JSON; its genes are flattened into the characterization panel"
    ),
    cat_cols: str | None = typer.Option(
        None, help="comma-separated obs columns to cross-tab per sub-cluster"
    ),
    lognorm_layer: str | None = typer.Option(None, help="layer with log-normalized data"),
) -> None:
    """Sub-cluster existing clusters on the anchor embedding and characterize the splits."""
    import scanpy as sc

    wd = resolve_workdir(workdir)
    targets = split_csv(clusters)
    tag = tag or "_".join(targets).replace("/", "_")

    adata_obj = load_adata(adata)
    require_obs(adata_obj, cluster_key, "cluster key")
    if embedding_key not in adata_obj.obsm:
        raise SystemExit(f"ERROR: embedding '{embedding_key}' not in obsm")
    if lognorm_layer:
        adata_obj.X = resolve_expression(adata_obj, lognorm_layer)

    mask = adata_obj.obs[cluster_key].astype(str).isin(targets).values
    sub = adata_obj[mask].copy()
    if sub.n_obs == 0:
        raise SystemExit(f"ERROR: no cells in clusters {targets}")
    eprint(
        f">> subclustering {sub.n_obs} cells from cluster(s) {targets} "
        f"on '{embedding_key}' @ resolution {resolution}"
    )

    sc.pp.neighbors(sub, use_rep=embedding_key, n_neighbors=n_neighbors)
    # igraph flavor = CPU-friendly, deterministic; never requires GPU/leidenalg quirks
    sc.tl.leiden(
        sub, resolution=resolution, flavor="igraph", n_iterations=2, directed=False, key_added="sub"
    )
    sub.obs["sub"] = sub.obs["sub"].astype(str)
    n_sub = sub.obs["sub"].nunique()
    eprint(f">> {n_sub} sub-clusters")

    # per-cell assignment (real parent id preserved verbatim)
    assign = pd.DataFrame(
        {
            "cell": sub.obs_names,
            "parent_cluster": sub.obs[cluster_key].astype(str).values,
            "sub": sub.obs["sub"].values,
        }
    )
    save_table(assign, wd, f"subcluster_{level}_{tag}_assignments.csv")

    # characterization panel
    gene_panel = []
    if genes:
        gene_panel += split_csv(genes)
    if markers:
        for v in read_markers_json(markers).values():
            gene_panel += v if isinstance(v, list) else v.get("pos", [])
    if not gene_panel:
        sc.pp.highly_variable_genes(sub, n_top_genes=min(30, sub.n_vars))
        gene_panel = list(sub.var_names[sub.var["highly_variable"]])

    frac, used = _frac_expressing(sub, gene_panel)
    summary = frac.assign(sub=sub.obs["sub"].values).groupby("sub").mean().round(3)
    summary.insert(0, "n_cells", sub.obs["sub"].value_counts().reindex(summary.index).astype(int))
    # dominant parent per sub (informative when pooling several clusters)
    par = pd.crosstab(sub.obs["sub"], sub.obs[cluster_key].astype(str), normalize="index").round(2)
    summary["top_parent"] = [
        par.loc[s].idxmax() + f"({par.loc[s].max():.0%})" for s in summary.index
    ]
    for col in split_csv(cat_cols):
        if col in sub.obs:
            ct = pd.crosstab(sub.obs["sub"], sub.obs[col].astype(str), normalize="index").round(2)
            summary[f"{col}_top"] = [
                ct.loc[s].idxmax() + f"({ct.loc[s].max():.0%})" for s in summary.index
            ]
    save_table(summary.reset_index(), wd, f"subcluster_{level}_{tag}_summary.csv", index=False)

    eprint(f">> wrote tables/subcluster_{level}_{tag}_assignments.csv and _summary.csv")
    import json

    print(
        json.dumps(
            {
                "n_cells": int(sub.n_obs),
                "n_subclusters": int(n_sub),
                "genes_used": used or gene_panel,
                "tag": tag,
            },
            indent=2,
        )
    )
