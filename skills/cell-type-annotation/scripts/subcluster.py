#!/usr/bin/env python
"""Sub-cluster one or more existing clusters on the anchor embedding — standardized
cluster refinement (in scope: see SKILL.md). Use when a cluster is marker-incoherent
(e.g. mutually-exclusive markers co-expressed) or suspiciously large/heterogeneous.

Subsets the chosen clusters, builds a kNN graph on the SAME embedding used everywhere,
runs Leiden at a given resolution, and characterizes each sub-cluster by the fraction
of cells expressing a panel of marker genes (and per-sub QC / categorical composition
if asked). Writes a per-cell sub-assignment table + a per-sub summary to <workdir>/tables/.

This does NOT relabel anything — it produces evidence so the annotator can decide whether
a cluster splits into real sub-populations (and how to label them), or is homogeneous.

Usage:
  python subcluster.py --adata path.h5ad --cluster-key leiden --embedding-key X_scpoli \
      --clusters 19,16,2 --workdir DIR --level L3 --tag diencephalon \
      [--resolution 0.5] [--genes FOXG1,NKX2-1,...] [--markers markers.json] \
      [--cat-cols condition,week] [--lognorm-layer lognorm]

Either --genes (comma list) or --markers (JSON {type:[genes]}, flattened) supplies the
characterization panel; if neither is given, the top variable genes are used.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from cta_io import (eprint, load_adata, read_markers_json, require_obs,
                    resolve_expression, resolve_workdir, save_table, split_csv)


def _frac_expressing(adata, genes):
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    if not genes:
        return pd.DataFrame(index=adata.obs_names), []
    X = adata[:, genes].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return pd.DataFrame(X > 0, columns=genes, index=adata.obs_names), genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--embedding-key", required=True,
                    help="obsm key of the anchor embedding (neighbors built on this)")
    ap.add_argument("--clusters", required=True,
                    help="comma-separated cluster id(s) to subcluster (pooled if >1)")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", default="L1", help="hierarchy level tag for filenames")
    ap.add_argument("--tag", default=None,
                    help="short name for this refinement (filenames); default from --clusters")
    ap.add_argument("--resolution", type=float, default=0.5)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--genes", default=None, help="comma-separated characterization genes")
    ap.add_argument("--markers", default=None,
                    help="markers JSON; its genes are flattened into the characterization panel")
    ap.add_argument("--cat-cols", default=None,
                    help="comma-separated obs columns to cross-tab per sub-cluster")
    ap.add_argument("--lognorm-layer", default=None)
    args = ap.parse_args()

    wd = resolve_workdir(args.workdir)
    targets = split_csv(args.clusters)
    tag = args.tag or "_".join(targets).replace("/", "_")

    adata = load_adata(args.adata)
    require_obs(adata, args.cluster_key, "cluster key")
    if args.embedding_key not in adata.obsm:
        raise SystemExit(f"ERROR: embedding '{args.embedding_key}' not in obsm")
    if args.lognorm_layer:
        adata.X = resolve_expression(adata, args.lognorm_layer)

    mask = adata.obs[args.cluster_key].astype(str).isin(targets).values
    sub = adata[mask].copy()
    if sub.n_obs == 0:
        raise SystemExit(f"ERROR: no cells in clusters {targets}")
    eprint(f">> subclustering {sub.n_obs} cells from cluster(s) {targets} "
           f"on '{args.embedding_key}' @ resolution {args.resolution}")

    sc.pp.neighbors(sub, use_rep=args.embedding_key, n_neighbors=args.n_neighbors)
    # igraph flavor = CPU-friendly, deterministic; never requires GPU/leidenalg quirks
    sc.tl.leiden(sub, resolution=args.resolution, flavor="igraph", n_iterations=2,
                 directed=False, key_added="sub")
    sub.obs["sub"] = sub.obs["sub"].astype(str)
    n_sub = sub.obs["sub"].nunique()
    eprint(f">> {n_sub} sub-clusters")

    # per-cell assignment (real parent id preserved verbatim)
    assign = pd.DataFrame({
        "cell": sub.obs_names,
        "parent_cluster": sub.obs[args.cluster_key].astype(str).values,
        "sub": sub.obs["sub"].values,
    })
    save_table(assign, wd, f"subcluster_{args.level}_{tag}_assignments.csv")

    # characterization panel
    genes = []
    if args.genes:
        genes += split_csv(args.genes)
    if args.markers:
        for v in read_markers_json(args.markers).values():
            genes += (v if isinstance(v, list) else v.get("pos", []))
    if not genes:
        sc.pp.highly_variable_genes(sub, n_top_genes=min(30, sub.n_vars))
        genes = list(sub.var_names[sub.var["highly_variable"]])

    frac, used = _frac_expressing(sub, genes)
    summary = frac.assign(sub=sub.obs["sub"].values).groupby("sub").mean().round(3)
    summary.insert(0, "n_cells", sub.obs["sub"].value_counts().reindex(summary.index).astype(int))
    # dominant parent per sub (informative when pooling several clusters)
    par = (pd.crosstab(sub.obs["sub"], sub.obs[args.cluster_key].astype(str),
                       normalize="index").round(2))
    summary["top_parent"] = [par.loc[s].idxmax() + f"({par.loc[s].max():.0%})" for s in summary.index]
    for col in split_csv(args.cat_cols):
        if col in sub.obs:
            ct = pd.crosstab(sub.obs["sub"], sub.obs[col].astype(str), normalize="index").round(2)
            summary[f"{col}_top"] = [ct.loc[s].idxmax() + f"({ct.loc[s].max():.0%})"
                                     for s in summary.index]
    save_table(summary.reset_index(), wd, f"subcluster_{args.level}_{tag}_summary.csv", index=False)

    eprint(f">> wrote tables/subcluster_{args.level}_{tag}_assignments.csv and _summary.csv")
    print(json.dumps({"n_cells": int(sub.n_obs), "n_subclusters": int(n_sub),
                      "genes_used": used or genes, "tag": tag}, indent=2))


if __name__ == "__main__":
    main()
