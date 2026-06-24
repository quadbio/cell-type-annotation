#!/usr/bin/env python
"""Stage 1: compute standard QC metrics + a cell-cycle signal into the canonical sidecar.

The dominant blind-run failure mode is a cluster that is not a single clean type —
low-quality cells, or cycling cells whose cell-cycle programme masks lineage — getting a
confident single-type label because the (blinded) object carried no QC to make the problem
visible. This step makes those problems DETECTABLE up front, by writing the standard scanpy
QC metrics and a cell-cycle score into `obs` **when they are absent**:

  * `n_genes_by_counts`, `total_counts`     -- depth / complexity (low -> low-quality)
  * `pct_counts_mt` (+ `pct_counts_ribo`)   -- mitochondrial / ribosomal fraction
  * `S_score`, `G2M_score`, `phase`          -- cell-cycle signal (high -> cycling)

This is NOT QC *fixing* and NOT doublet detection — both stay out of scope (the skill
alerts, it does not normalize, filter, or run Scrublet). It only computes read-only metrics
the annotator reads (via `cta cluster profile`) to decide a cluster's `state` axis
(normal / low-quality / cycling / …) alongside its cell-type label.

Operates on the **canonical working sidecar** (`<stem>_cta.h5ad` from `cta data canonicalize`)
and rewrites it in place atomically — the user's original file is never touched. Existing QC
columns are preserved (only absent metrics are filled) unless `--recompute` is given.

NO-BIAS RULE: the mito/ribo gene prefixes and the cell-cycle gene lists are arguments with
standard defaults (`MT-`/`mt-`; the Tirosh et al. 2015 S/G2M lists, matched
case-insensitively so the same lists work for human and mouse). Nothing dataset-specific is
hardcoded; pass `--s-genes`/`--g2m-genes` to override for a custom panel.

Usage:
  python qc.py --adata path_cta.h5ad \
      [--mito-prefix MT-,mt-] [--ribo-prefix RPS,RPL,Rps,Rpl] \
      [--s-genes a,b,c|FILE] [--g2m-genes a,b,c|FILE] [--recompute]
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import typer

from cta.io import eprint, load_adata, split_csv

# Tirosh et al. 2015 (Regev lab) cell-cycle gene lists — the de-facto standard, as shipped
# with scanpy's cell-cycle tutorial. Human symbols; matched case-insensitively so they also
# resolve mouse orthologs (Mcm5 <-> MCM5). A few entries use legacy symbols that modern
# atlases have renamed (MLF1IP->CENPU, FAM64A->PIMREG, HN1->JPT1); those simply go unmatched
# on such objects, which is fine — override with --s-genes/--g2m-genes for a custom panel.
_S_GENES_DEFAULT = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
    "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP",
    "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2",
    "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2",
    "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
]
_G2M_GENES_DEFAULT = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2",
    "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2",
    "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B",
    "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2",
    "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN",
    "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]

# obs columns calculate_qc_metrics produces that we keep (lean: skip the log1p_* / percent_top
# variants). `total_counts_<qc_var>`/`pct_counts_<qc_var>` are added per --qc-vars below.
_BASE_METRICS = ["n_genes_by_counts", "total_counts"]
_CC_COLS = ["S_score", "G2M_score", "phase"]


def _prefix_mask(var_names, prefixes: list[str]):
    """Boolean mask over var_names for genes starting with any of `prefixes` (case-sensitive
    so 'MT-'/'mt-' are both listed explicitly and a human 'MTOR' is not caught by 'mt-')."""
    import numpy as np
    names = [str(n) for n in var_names]
    return np.array([any(n.startswith(p) for p in prefixes) for n in names], dtype=bool)


def _resolve_gene_list(spec: Optional[str], default: list[str]) -> list[str]:
    """A --s-genes/--g2m-genes value: a path to a newline/comma file, a comma list, or None
    (use the Tirosh default)."""
    if not spec:
        return list(default)
    p = Path(spec).expanduser()
    if p.exists():
        text = p.read_text()
        return [t.strip() for t in text.replace(",", "\n").splitlines() if t.strip()]
    return split_csv(spec)


def _present_case_insensitive(genes: list[str], var_names) -> list[str]:
    """Subset `genes` to those present in var_names, matched case-insensitively, returning the
    var_names spelling (so scanpy finds them) — lets one human-symbol list serve mouse too."""
    lut = {str(n).upper(): str(n) for n in var_names}
    out, seen = [], set()
    for g in genes:
        hit = lut.get(str(g).upper())
        if hit is not None and hit not in seen:
            out.append(hit)
            seen.add(hit)
    return out


def _compute_qc_metrics(adata, mito_prefixes, ribo_prefixes, summary):
    """Return a DataFrame of the QC-metric columns (indexed like adata.obs), computed from
    `layers['counts']` when present (else skipped/approximate). Never mutates adata."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc

    counts = adata.layers.get("counts") if adata.layers else None
    if counts is None:
        # n_genes (nonzero genes per cell) is still valid from X; depth/mito need real counts.
        eprint("   WARN: no layers['counts'] — computing n_genes_by_counts from X only; "
               "total_counts / pct_counts_mt need integer counts and are SKIPPED.")
        X = adata.X
        nnz = (X > 0).sum(axis=1)
        nnz = np.asarray(nnz).ravel()
        summary["counts_source"] = "none (n_genes from X)"
        return pd.DataFrame({"n_genes_by_counts": nnz.astype(int)}, index=adata.obs_names)

    summary["counts_source"] = "layers['counts']"
    mt = _prefix_mask(adata.var_names, mito_prefixes)
    ribo = _prefix_mask(adata.var_names, ribo_prefixes)
    summary["n_mito_genes"] = int(mt.sum())
    summary["n_ribo_genes"] = int(ribo.sum())
    eprint(f"   mito genes matched: {int(mt.sum())} ({','.join(mito_prefixes)}); "
           f"ribo genes matched: {int(ribo.sum())} ({','.join(ribo_prefixes)})")

    # Compute on a throwaway object so calculate_qc_metrics never pollutes the canonical
    # var/obs; pull back only the obs columns we keep.
    qc_vars, tmp_var = [], pd.DataFrame(index=adata.var_names)
    if mt.any():
        tmp_var["mt"] = mt
        qc_vars.append("mt")
    if ribo.any():
        tmp_var["ribo"] = ribo
        qc_vars.append("ribo")
    tmp = ad.AnnData(X=counts, var=tmp_var)
    obs_metrics, _ = sc.pp.calculate_qc_metrics(
        tmp, qc_vars=qc_vars, percent_top=None, log1p=False, inplace=False)
    obs_metrics.index = adata.obs_names
    keep = [c for c in (_BASE_METRICS + [f"pct_counts_{v}" for v in qc_vars]
                        + [f"total_counts_{v}" for v in qc_vars]) if c in obs_metrics.columns]
    return obs_metrics[keep]


def _compute_cell_cycle(adata, s_genes, g2m_genes, summary):
    """Return a DataFrame with S_score/G2M_score/phase (indexed like adata.obs), scored on the
    log-norm X, or None when too few cell-cycle genes are on the panel (e.g. targeted panel)."""
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    s_present = _present_case_insensitive(s_genes, adata.var_names)
    g2m_present = _present_case_insensitive(g2m_genes, adata.var_names)
    summary["cell_cycle"] = {"s_genes_on_panel": len(s_present),
                             "g2m_genes_on_panel": len(g2m_present)}
    # Need a handful of each for a meaningful score; below that, scoring is noise (and scanpy
    # can fail to build a control set). Skip honestly rather than emit a junk signal.
    if len(s_present) < 2 or len(g2m_present) < 2:
        eprint(f"   cell-cycle: only {len(s_present)} S / {len(g2m_present)} G2M genes on "
               "the panel — too few to score; SKIPPING (override with --s-genes/--g2m-genes).")
        summary["cell_cycle"]["scored"] = False
        return None

    eprint(f"   cell-cycle: scoring with {len(s_present)} S + {len(g2m_present)} G2M genes")
    tmp = ad.AnnData(X=adata.X.copy(), var=pd.DataFrame(index=adata.var_names))
    sc.tl.score_genes_cell_cycle(tmp, s_genes=s_present, g2m_genes=g2m_present)
    summary["cell_cycle"]["scored"] = True
    out = tmp.obs[["S_score", "G2M_score", "phase"]].copy()
    out.index = adata.obs_names
    return out


def cli(
    adata: str = typer.Option(
        ..., help="path to the canonical sidecar (<stem>_cta.h5ad) to add QC metrics to"),
    mito_prefix: str = typer.Option(
        "MT-,mt-", help="comma-separated gene-name prefixes for mitochondrial genes"),
    ribo_prefix: str = typer.Option(
        "RPS,RPL,Rps,Rpl", help="comma-separated gene-name prefixes for ribosomal genes"),
    s_genes: Optional[str] = typer.Option(
        None, help="override S-phase genes: comma list or a file (default: Tirosh 2015)"),
    g2m_genes: Optional[str] = typer.Option(
        None, help="override G2M genes: comma list or a file (default: Tirosh 2015)"),
    recompute: bool = typer.Option(
        False, help="overwrite QC columns that already exist (default: fill only absent ones)"),
) -> None:
    """Compute standard QC metrics + a cell-cycle signal into the canonical sidecar's obs."""
    import pandas as pd

    path = Path(adata).expanduser()
    # Load the file as-is (it IS the canonical working copy); never reach for another sidecar.
    adata_obj = load_adata(str(path), prefer_slim=False)
    summary: dict = {"adata": str(path), "n_obs": int(adata_obj.n_obs),
                     "n_vars": int(adata_obj.n_vars)}

    eprint(f">> QC metrics on {path.name}  (n_obs={adata_obj.n_obs}, n_vars={adata_obj.n_vars})")
    metrics = _compute_qc_metrics(
        adata_obj, split_csv(mito_prefix), split_csv(ribo_prefix), summary)

    eprint(">> cell-cycle signal")
    cc = _compute_cell_cycle(
        adata_obj, _resolve_gene_list(s_genes, _S_GENES_DEFAULT),
        _resolve_gene_list(g2m_genes, _G2M_GENES_DEFAULT), summary)

    candidate = metrics if cc is None else pd.concat([metrics, cc], axis=1)

    # Fill only absent columns unless --recompute: never clobber QC the user already curated.
    written, skipped = [], []
    for col in candidate.columns:
        if col in adata_obj.obs.columns and not recompute:
            skipped.append(col)
            continue
        adata_obj.obs[col] = candidate[col].values
        written.append(col)
    summary["written"] = written
    summary["skipped_existing"] = skipped

    if skipped:
        eprint(f"   kept {len(skipped)} existing column(s) (use --recompute to overwrite): "
               f"{skipped}")
    if not written:
        eprint(">> nothing to write (all QC columns already present). File left unchanged.")
        print(str(path))
        return

    # Atomic in-place rewrite: temp file in the same dir, then os.replace, so a failed write
    # never corrupts the canonical sidecar.
    dest = str(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".h5ad", dir=os.path.dirname(dest) or ".")
    os.close(fd)
    try:
        adata_obj.write_h5ad(tmp_path)
        os.replace(tmp_path, dest)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    (path.with_name("qc.json")).write_text(json.dumps(summary, indent=2))
    eprint(f"\n>> wrote {len(written)} obs column(s) to {path.name}: {written}")
    eprint(f"   summary: {path.with_name('qc.json')}")
    print(str(path))
