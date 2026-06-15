#!/usr/bin/env python
"""Job A: per-cluster marker identification.

Runs scanpy rank_genes_groups (Wilcoxon, tie_correct — NOT the t-test default) on
log-normalized data, plus COSG for specificity, and writes tidy marker tables to
<workdir>/tables/. This is the descriptive marker evidence that feeds annotation.

This is NOT condition DE (Job B = pseudobulk + PyDESeq2). See
reference/differential_expression.md.

Usage:
  python compute_markers.py --adata path.h5ad --cluster-key leiden \
      --workdir /proj/ds-annotation-run --level L1 \
      [--lognorm-layer lognorm] [--groups 3,5,7] [--n-genes 50] [--no-cosg]
"""
from __future__ import annotations

import argparse

import pandas as pd
import scanpy as sc

from cta_io import (eprint, load_adata, require_obs, resolve_expression,
                 resolve_workdir, save_table, split_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", default="L1", help="hierarchy level tag for filenames")
    ap.add_argument("--lognorm-layer", default=None,
                    help="layer with log-normalized data; default uses adata.X")
    ap.add_argument("--groups", default=None,
                    help="comma-separated subset of clusters; default all")
    ap.add_argument("--n-genes", type=int, default=50)
    ap.add_argument("--no-cosg", action="store_true")
    args = ap.parse_args()

    wd = resolve_workdir(args.workdir)
    adata = load_adata(args.adata)
    require_obs(adata, args.cluster_key, "cluster key")

    # Use the log-normalized matrix as adata.X for the tests (don't mutate the file).
    X = resolve_expression(adata, args.lognorm_layer)
    if args.lognorm_layer is not None:
        adata = adata.copy()
        adata.X = X

    adata.obs[args.cluster_key] = adata.obs[args.cluster_key].astype("category")
    groups = split_csv(args.groups) or "all"

    eprint(">> rank_genes_groups (wilcoxon, tie_correct=True, pts=True)")
    sc.tl.rank_genes_groups(
        adata, groupby=args.cluster_key, groups=groups,
        method="wilcoxon", tie_correct=True, pts=True, use_raw=False,
    )
    frames = []
    grp_list = (adata.obs[args.cluster_key].cat.categories if groups == "all"
                else groups)
    for g in grp_list:
        df = sc.get.rank_genes_groups_df(adata, group=str(g)).head(args.n_genes)
        df.insert(0, "cluster", str(g))
        frames.append(df)
    markers = pd.concat(frames, ignore_index=True)
    save_table(markers, wd, f"markers_{args.level}.csv")

    if not args.no_cosg:
        try:
            import cosg
            eprint(">> COSG (cosine-similarity specificity)")
            cosg.cosg(adata, groupby=args.cluster_key, key_added="cosg",
                      mu=1.0, n_genes_user=args.n_genes)
            names = pd.DataFrame(adata.uns["cosg"]["names"])
            scores = pd.DataFrame(adata.uns["cosg"]["scores"])
            rows = []
            for col in names.columns:
                for rank, (gene, score) in enumerate(zip(names[col], scores[col])):
                    rows.append({"cluster": str(col), "rank": rank,
                                 "gene": gene, "cosg_score": float(score)})
            save_table(pd.DataFrame(rows), wd, f"cosg_{args.level}.csv")
        except Exception as e:  # noqa: BLE001 - tool-level degradation is OK
            eprint(f"   COSG skipped ({type(e).__name__}: {e})")

    eprint(">> done")


if __name__ == "__main__":
    main()
