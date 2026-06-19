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

from pathlib import Path
from typing import Optional

import typer

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


def cli(
    adata: str = typer.Option(..., help="path to the source .h5ad"),
    out: Optional[str] = typer.Option(
        None, help="default: <dir-of-adata>/<stem>_cta.h5ad"),
    cluster_key: str = typer.Option(
        ..., help="obs column with the cluster labels to annotate"),
    embedding_key: str = typer.Option(
        ..., help="obsm key for the analysis embedding (e.g. X_scpoli)"),
    viz_key: Optional[str] = typer.Option(
        None, help="obsm key for a 2D viz embedding (e.g. X_umap)"),
    lognorm_layer: str = typer.Option(
        "X", help="layer holding log-normalized data; 'X' (default) uses adata.X"),
    counts_layer: Optional[str] = typer.Option(
        None, help="layer to carry over as layers['counts']"),
    keep_obs: Optional[str] = typer.Option(
        None, help="comma-separated obs columns to keep; default keeps ALL obs. "
                   "When given, only these plus --cluster-key are kept."),
) -> None:
    """Write a slim copy of a large AnnData with only what the skill needs."""
    import anndata as ad

    src = Path(adata).expanduser()
    out_path = (Path(out).expanduser() if out
                else src.with_name(f"{src.stem}_cta.h5ad"))

    eprint(f">> loading {src}")
    adata_obj = load_adata(adata, prefer_slim=False)  # always slim from the full original
    orig_shape = (adata_obj.n_obs, adata_obj.n_vars)

    # --- validate keys before building anything ---
    if cluster_key not in adata_obj.obs:
        raise SystemExit(
            f"ERROR: --cluster-key '{cluster_key}' not in adata.obs. "
            f"Available: {list(adata_obj.obs.columns)}"
        )
    _require_obsm(adata_obj, embedding_key, "embedding-key")
    if viz_key:
        _require_obsm(adata_obj, viz_key, "viz-key")

    X, x_src = _resolve_lognorm(adata_obj, lognorm_layer)
    eprint(f"   slim X <- {x_src} (log-normalized)")

    # --- obs: all columns by default, or keep-list + cluster-key ---
    if keep_obs:
        keep = split_csv(keep_obs)
        keep = list(dict.fromkeys([*keep, cluster_key]))  # ensure cluster, dedup
        missing = [c for c in keep if c not in adata_obj.obs]
        if missing:
            raise SystemExit(
                f"ERROR: --keep-obs columns not in adata.obs: {missing}. "
                f"Available: {list(adata_obj.obs.columns)}"
            )
        obs = adata_obj.obs[keep].copy()
        eprint(f"   obs <- {len(keep)} columns: {keep}")
    else:
        obs = adata_obj.obs.copy()
        eprint(f"   obs <- all {obs.shape[1]} columns")

    # --- obsm: only the embedding(s) the skill uses ---
    obsm = {embedding_key: adata_obj.obsm[embedding_key]}
    if viz_key:
        obsm[viz_key] = adata_obj.obsm[viz_key]
    eprint(f"   obsm <- {list(obsm.keys())}")

    # --- layers: only counts, if requested ---
    layers = {}
    if counts_layer:
        if counts_layer not in (adata_obj.layers or {}):
            raise SystemExit(
                f"ERROR: --counts-layer '{counts_layer}' not found. "
                f"Available layers: {list(adata_obj.layers.keys())}"
            )
        layers["counts"] = adata_obj.layers[counts_layer]
        eprint(f"   layers['counts'] <- layers['{counts_layer}']")

    # --- var: keep var_names + a minimal var frame (drop bulky var columns) ---
    var = adata_obj.var[[]].copy()  # preserves var_names index, drops all var columns

    slim = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers=layers)

    # Carry log1p marker so downstream knows X is logged (drop other uns bulk).
    if "log1p" in adata_obj.uns:
        slim.uns["log1p"] = adata_obj.uns["log1p"]
        eprint("   uns['log1p'] carried over")

    eprint(f">> writing {out_path}")
    slim.write_h5ad(out_path)

    eprint("\n>> done")
    eprint(f"   original : n_obs={orig_shape[0]}  n_vars={orig_shape[1]}")
    eprint(f"   slim     : n_obs={slim.n_obs}  n_vars={slim.n_vars}")
    eprint(f"   out      : {out_path}")
    print(out_path)
