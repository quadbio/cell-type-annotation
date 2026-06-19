#!/usr/bin/env python
"""Write a slim copy of a large AnnData once, so downstream scripts stop reloading
the full multi-GB object.

Big objects (millions of cells) carry graphs, alternative embeddings, scaled
matrices, and varm/obsp/varp that the annotation skill never touches. Every script
that re-reads the original pays that cost again. This builds a minimal artifact with
ONLY what the skill needs — the log-normalized matrix, the cluster labels, the
analysis embedding (and optional viz embedding), and (optionally) a counts layer —
then downstream jobs point at the slim file instead.

NO-BIAS RULE: nothing dataset-specific is hardcoded. Every key (cluster, embedding,
viz, layers, obs to keep) is a CLI argument.

Usage:
  python prep_slim.py --adata A.h5ad [--out OUT.h5ad] \
      --cluster-key leiden --embedding-key X_scpoli [--viz-key X_umap] \
      [--lognorm-layer X|<layer>] [--counts-layer counts] \
      [--keep-obs sample,condition,batch]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad

from cta.io import eprint, load_adata, split_csv


def _resolve_lognorm(adata: ad.AnnData, layer: str | None):
    """Matrix to store as slim X. `None`/'X' uses adata.X; otherwise that layer."""
    if layer is None or layer == "X":
        return adata.X, "X"
    if layer not in (adata.layers or {}):
        raise SystemExit(
            f"ERROR: --lognorm-layer '{layer}' not found. Available layers: "
            f"{list(adata.layers.keys())}"
        )
    return adata.layers[layer], f"layers['{layer}']"


def _require_obsm(adata: ad.AnnData, key: str, what: str):
    if key not in adata.obsm:
        raise SystemExit(
            f"ERROR: --{what} '{key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--out", help="default: <dir-of-adata>/<stem>_cta.h5ad")
    ap.add_argument("--cluster-key", required=True,
                    help="obs column with the cluster labels to annotate")
    ap.add_argument("--embedding-key", required=True,
                    help="obsm key for the analysis embedding (e.g. X_scpoli)")
    ap.add_argument("--viz-key", help="obsm key for a 2D viz embedding (e.g. X_umap)")
    ap.add_argument("--lognorm-layer", default="X",
                    help="layer holding log-normalized data; 'X' (default) uses adata.X")
    ap.add_argument("--counts-layer", help="layer to carry over as layers['counts']")
    ap.add_argument("--keep-obs",
                    help="comma-separated obs columns to keep; default keeps ALL obs. "
                         "When given, only these plus --cluster-key are kept.")
    args = ap.parse_args()

    src = Path(args.adata).expanduser()
    out = (Path(args.out).expanduser() if args.out
           else src.with_name(f"{src.stem}_cta.h5ad"))

    eprint(f">> loading {src}")
    adata = load_adata(args.adata, prefer_slim=False)  # always slim from the full original
    orig_shape = (adata.n_obs, adata.n_vars)

    # --- validate keys before building anything ---
    if args.cluster_key not in adata.obs:
        raise SystemExit(
            f"ERROR: --cluster-key '{args.cluster_key}' not in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    _require_obsm(adata, args.embedding_key, "embedding-key")
    if args.viz_key:
        _require_obsm(adata, args.viz_key, "viz-key")

    X, x_src = _resolve_lognorm(adata, args.lognorm_layer)
    eprint(f"   slim X <- {x_src} (log-normalized)")

    # --- obs: all columns by default, or keep-list + cluster-key ---
    if args.keep_obs:
        keep = split_csv(args.keep_obs)
        keep = list(dict.fromkeys([*keep, args.cluster_key]))  # ensure cluster, dedup
        missing = [c for c in keep if c not in adata.obs]
        if missing:
            raise SystemExit(
                f"ERROR: --keep-obs columns not in adata.obs: {missing}. "
                f"Available: {list(adata.obs.columns)}"
            )
        obs = adata.obs[keep].copy()
        eprint(f"   obs <- {len(keep)} columns: {keep}")
    else:
        obs = adata.obs.copy()
        eprint(f"   obs <- all {obs.shape[1]} columns")

    # --- obsm: only the embedding(s) the skill uses ---
    obsm = {args.embedding_key: adata.obsm[args.embedding_key]}
    if args.viz_key:
        obsm[args.viz_key] = adata.obsm[args.viz_key]
    eprint(f"   obsm <- {list(obsm.keys())}")

    # --- layers: only counts, if requested ---
    layers = {}
    if args.counts_layer:
        if args.counts_layer not in (adata.layers or {}):
            raise SystemExit(
                f"ERROR: --counts-layer '{args.counts_layer}' not found. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        layers["counts"] = adata.layers[args.counts_layer]
        eprint(f"   layers['counts'] <- layers['{args.counts_layer}']")

    # --- var: keep var_names + a minimal var frame (drop bulky var columns) ---
    var = adata.var[[]].copy()  # preserves var_names index, drops all var columns

    slim = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers=layers)

    # Carry log1p marker so downstream knows X is logged (drop other uns bulk).
    if "log1p" in adata.uns:
        slim.uns["log1p"] = adata.uns["log1p"]
        eprint("   uns['log1p'] carried over")

    eprint(f">> writing {out}")
    slim.write_h5ad(out)

    eprint("\n>> done")
    eprint(f"   original : n_obs={orig_shape[0]}  n_vars={orig_shape[1]}")
    eprint(f"   slim     : n_obs={slim.n_obs}  n_vars={slim.n_vars}")
    eprint(f"   out      : {out}")
    print(out)


if __name__ == "__main__":
    main()
