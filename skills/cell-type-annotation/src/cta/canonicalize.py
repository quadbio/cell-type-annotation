#!/usr/bin/env python
"""Canonicalize a (possibly huge) AnnData ONCE, into the one shape every downstream
`cta` command expects — so no step has to re-resolve input variability on its own.

This is the single Stage-1 data step (it replaces the old `prep-slim`: it still writes
a slim sidecar, but now also *standardizes* what the slim object contains). Downstream
scripts call `load_adata`, which transparently picks up this sidecar, and can then
assume a fixed contract:

  * `X`               — log-normalized expression
  * `layers['counts']`— integer counts (from a named layer, or pulled out of `.raw`)
  * `var_names`       — gene SYMBOLS (curated / CellTypist / literature markers are
                        symbols, so panel-gating matches); the original IDs are kept in
                        `var['ensembl_id']` and the symbol in `var['gene_symbol']`
  * the analysis embedding (+ optional viz embedding), the cluster key, `uns['log1p']`

Big objects (millions of cells) carry graphs, alternative embeddings, scaled matrices
and varm/obsp/varp the skill never touches; this drops all of that so later jobs don't
re-deserialize the full file.

GENE SYMBOLS. var_names that already look like symbols are kept as-is. var_names that
look like Ensembl IDs are renamed using, in order: a `--symbol-col` you name, an
auto-detected symbol var-column (feature_name / gene_symbols / symbol / gene_name), or
a `--symbol-map` TSV (ensembl<TAB>symbol). When none exists the skill builds that map
via the knowledgebase MCP (symbol -> Ensembl, the tool's natural direction) for the
marker genes and passes it as `--symbol-map`; the CLI itself never calls an MCP.

COUNTS. Pulled into `layers['counts']` from `--counts-layer`, else from `.raw.X` when
`.raw` is present and integer-valued (so counts hidden in `.raw` — common on CELLxGENE
atlases — become visible to the normalization fingerprint and to scANVI).

NO-BIAS RULE: nothing dataset-specific is hardcoded. Every key (cluster, embedding,
viz, layers, symbol column, obs to keep) is a CLI argument.

Usage:
  python canonicalize.py --adata A.h5ad [--out OUT.h5ad] \
      --cluster-key leiden --embedding-key X_scpoli [--viz-key X_umap] \
      [--lognorm-layer X|<layer>] [--counts-layer counts] \
      [--symbol-col feature_name] [--symbol-map ensembl_to_symbol.tsv] \
      [--keep-obs sample,condition,batch]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from cta.io import _looks_like_ensembl, eprint, load_adata, split_csv

# var columns that conventionally hold the gene symbol, best first (CELLxGENE uses
# `feature_name`; the immune benchmark atlas uses `gene_symbols`).
_SYMBOL_COL_CANDIDATES = ("feature_name", "gene_symbols", "gene_symbol", "symbol",
                          "gene_name", "Symbol", "Gene", "genes")


def _detect_symbol_column(adata) -> str | None:
    """First var column that conventionally holds symbols AND doesn't itself look Ensembl."""
    for col in _SYMBOL_COL_CANDIDATES:
        if col in adata.var.columns and not _looks_like_ensembl(adata.var[col].astype(str)):
            return col
    return None


def _load_symbol_map(path: str) -> dict[str, str]:
    """Parse an `ensembl<TAB>symbol` (whitespace-separated) map; '#' lines and blanks skipped."""
    out: dict[str, str] = {}
    for ln in Path(path).expanduser().read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    if not out:
        raise SystemExit(f"ERROR: --symbol-map '{path}' yielded no ensembl->symbol pairs.")
    return out


def _canonical_symbols(adata, symbol_col: str | None, symbol_map: dict[str, str] | None):
    """Decide canonical var_names (symbols) for `adata`, leaving gene ORDER unchanged.

    Returns (symbols, ensembl_ids, info) where `symbols`/`ensembl_ids` are lists aligned
    to `adata.var_names`, and `info` records what happened (for canonicalization.json).
    Genes with no known symbol keep their original id (so nothing is dropped).
    """
    orig = [str(n) for n in adata.var_names]
    is_ens = _looks_like_ensembl(orig)
    if not is_ens:
        # var_names already look like symbols — keep them; surface any Ensembl column.
        ens_col = next((c for c in ("ensembl_id", "ensembl", "gene_ids", "gene_id")
                        if c in adata.var.columns), None)
        ensembl = [str(x) for x in adata.var[ens_col]] if ens_col else [""] * len(orig)
        return orig, ensembl, {"convention": "already symbols", "renamed": 0, "unmapped": 0}

    # var_names look like Ensembl -> we must map to symbols.
    if symbol_col is not None and symbol_col not in adata.var.columns:
        raise SystemExit(
            f"ERROR: --symbol-col '{symbol_col}' not in var. "
            f"Available var columns: {list(adata.var.columns)}")
    col = symbol_col or _detect_symbol_column(adata)
    symbols = list(orig)  # default: keep Ensembl id where no symbol is known
    renamed = 0
    if col is not None:
        eprint(f"   symbols <- var['{col}']")
        vals = adata.var[col].astype(str).tolist()
        for i, (o, s) in enumerate(zip(orig, vals)):
            if s and s.lower() not in ("nan", "none", "") and s != o:
                symbols[i] = s
                renamed += 1
    elif symbol_map is not None:
        eprint(f"   symbols <- --symbol-map ({len(symbol_map)} pairs)")
        for i, o in enumerate(orig):
            # match on the bare Ensembl id (strip a version suffix if the map is unversioned)
            s = symbol_map.get(o) or symbol_map.get(o.split(".")[0])
            if s:
                symbols[i] = s
                renamed += 1
    else:
        raise SystemExit(
            "ERROR: var_names look like Ensembl IDs but no symbol source was found. "
            "Provide --symbol-col, ensure a symbol var column exists, or pass "
            "--symbol-map (the skill builds one via the knowledgebase MCP).")
    unmapped = len(orig) - renamed
    info = {"convention": "ensembl->symbol", "source": col or "symbol-map",
            "renamed": renamed, "unmapped": unmapped}
    if unmapped:
        eprint(f"   WARN: {unmapped}/{len(orig)} genes kept their Ensembl id (no symbol found)")
    return symbols, orig, info


def _resolve_lognorm(adata, layer: str | None):
    """Matrix to store as the canonical X. `None`/'X' uses adata.X; otherwise that layer."""
    if layer is None or layer == "X":
        return adata.X, "X"
    if layer not in (adata.layers or {}):
        raise SystemExit(
            f"ERROR: --lognorm-layer '{layer}' not found. Available layers: "
            f"{list(adata.layers.keys())}")
    return adata.layers[layer], f"layers['{layer}']"


def _is_integer_valued(X, cap: int = 200_000) -> bool:
    import numpy as np
    import scipy.sparse as sp
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if data.size > cap:
        data = data[:cap]
    return bool(data.size and np.allclose(data, np.round(data)))


def _resolve_counts(adata, counts_layer: str | None):
    """Return (counts_matrix, source) aligned to adata's gene ORDER, or (None, reason).

    Resolution order: `--counts-layer`, else an existing `layers['counts']` (so
    re-canonicalizing the sidecar is idempotent), else `.raw.X` when `.raw` covers every
    gene (subset to the same order). We never return a partially-aligned matrix: if `.raw`
    is missing genes we skip counts and say so, so `layers['counts']` is either absent or
    fully aligned to X.
    """
    if counts_layer:
        if counts_layer not in (adata.layers or {}):
            raise SystemExit(
                f"ERROR: --counts-layer '{counts_layer}' not found. "
                f"Available layers: {list(adata.layers.keys())}")
        X = adata.layers[counts_layer]
        if not _is_integer_valued(X):
            eprint(f"   WARN: layers['{counts_layer}'] is not integer-valued; "
                   "storing it as counts anyway (point --counts-layer elsewhere if wrong)")
        return X, f"layers['{counts_layer}']"
    if "counts" in (adata.layers or {}):  # already canonical — carry it through
        return adata.layers["counts"], "layers['counts']"
    if adata.raw is not None:
        raw = adata.raw
        raw_names = [str(n) for n in raw.var_names]
        want = [str(n) for n in adata.var_names]
        pos = {n: i for i, n in enumerate(raw_names)}
        if all(n in pos for n in want):
            idx = [pos[n] for n in want]
            Xraw = raw.X[:, idx]
            if _is_integer_valued(Xraw):
                return Xraw, ".raw.X"
            eprint("   .raw.X present but not integer-valued — not using it as counts")
        else:
            missing = sum(1 for n in want if n not in pos)
            eprint(f"   .raw present but missing {missing} of {len(want)} genes — "
                   "skipping counts-from-.raw (would misalign)")
    return None, "none"


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
        None, help="layer with raw counts; if omitted, counts are pulled from .raw when present"),
    symbol_col: Optional[str] = typer.Option(
        None, help="var column to use as gene symbols (overrides auto-detection)"),
    symbol_map: Optional[str] = typer.Option(
        None, help="TSV 'ensembl<TAB>symbol' to rename Ensembl var_names "
                   "(used when no symbol var column exists; built by the skill via MCP)"),
    keep_obs: Optional[str] = typer.Option(
        None, help="comma-separated obs columns to keep; default keeps ALL obs. "
                   "When given, only these plus --cluster-key are kept."),
) -> None:
    """Canonicalize an AnnData into the one shape every downstream `cta` command expects."""
    import anndata as ad

    src = Path(adata).expanduser()
    out_path = (Path(out).expanduser() if out
                else src.with_name(f"{src.stem}_cta.h5ad"))

    eprint(f">> loading {src}")
    adata_obj = load_adata(adata, prefer_slim=False)  # always canonicalize from the full original
    orig_shape = (adata_obj.n_obs, adata_obj.n_vars)

    # --- validate keys before building anything ---
    if cluster_key not in adata_obj.obs:
        raise SystemExit(
            f"ERROR: --cluster-key '{cluster_key}' not in adata.obs. "
            f"Available: {list(adata_obj.obs.columns)}")
    for key, what in [(embedding_key, "embedding-key"), (viz_key, "viz-key")]:
        if key and key not in adata_obj.obsm:
            raise SystemExit(
                f"ERROR: --{what} '{key}' not found in adata.obsm. "
                f"Available: {list(adata_obj.obsm.keys())}")

    X, x_src = _resolve_lognorm(adata_obj, lognorm_layer)
    eprint(f"   X <- {x_src} (log-normalized)")

    # --- gene symbols (var_names) ---
    smap = _load_symbol_map(symbol_map) if symbol_map else None
    symbols, ensembl_ids, var_info = _canonical_symbols(adata_obj, symbol_col, smap)

    # --- counts ---
    counts, counts_src = _resolve_counts(adata_obj, counts_layer)
    eprint(f"   counts <- {counts_src}")

    # --- obs: all columns by default, or keep-list + cluster-key ---
    if keep_obs:
        keep = list(dict.fromkeys([*split_csv(keep_obs), cluster_key]))  # ensure cluster, dedup
        missing = [c for c in keep if c not in adata_obj.obs]
        if missing:
            raise SystemExit(
                f"ERROR: --keep-obs columns not in adata.obs: {missing}. "
                f"Available: {list(adata_obj.obs.columns)}")
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

    # --- var: canonical symbol index + a kept symbol<->Ensembl map (drop bulky var columns) ---
    import pandas as pd
    var = pd.DataFrame(
        {"gene_symbol": symbols, "ensembl_id": ensembl_ids},
        index=pd.Index(symbols, name=adata_obj.var_names.name))

    layers = {"counts": counts} if counts is not None else {}
    slim = ad.AnnData(X=X, obs=obs, var=var, obsm=obsm, layers=layers)

    # Resolve duplicate symbols (several Ensembl IDs -> one symbol) deterministically.
    n_dup = int(slim.var_names.duplicated().sum())
    if n_dup:
        slim.var_names_make_unique()
        eprint(f"   WARN: {n_dup} duplicate gene symbols made unique (e.g. SYMBOL-1)")
    var_info["deduped"] = n_dup

    # Carry log1p marker so downstream knows X is logged (drop other uns bulk).
    if "log1p" in adata_obj.uns:
        slim.uns["log1p"] = adata_obj.uns["log1p"]
        eprint("   uns['log1p'] carried over")

    eprint(f">> writing {out_path}")
    slim.write_h5ad(out_path)

    # --- machine-readable summary for context.md / run_state.json ---
    summary = {
        "source": str(src),
        "out": str(out_path),
        "n_obs": int(slim.n_obs),
        "n_vars": int(slim.n_vars),
        "var": var_info,
        "lognorm_source": x_src,
        "counts_source": counts_src,
        "has_counts": counts is not None,
    }
    summary_path = out_path.with_name("canonicalization.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    eprint("\n>> done")
    eprint(f"   original : n_obs={orig_shape[0]}  n_vars={orig_shape[1]}")
    eprint(f"   canonical: n_obs={slim.n_obs}  n_vars={slim.n_vars}  "
           f"var_names={var_info['convention']}  counts={counts_src}")
    eprint(f"   summary  : {summary_path}")
    print(out_path)
