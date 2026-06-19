#!/usr/bin/env python
"""Assemble a candidate marker prior from data-driven sources — a STARTING POINT.

Writes markers_<level>.json as {label: [genes]} by pulling marker sets from one or
more sources and merging them. This spares the annotator hand-typing an entire prior,
but the output is NOT final: it must still be curated (drop implausible genes, rename
labels to your hierarchy) and panel-gated against the query before use.

Sources (combinable; at least one required) — nothing dataset-specific is hardcoded,
every label and gene comes from a source you point at:
  --celltypist-model <Name>          extract_top_markers() from a pretrained model
  --reference R.h5ad --ref-label-col rank_genes_groups (wilcoxon) over the reference

Usage:
  python assemble_markers.py --out markers_L1.json [--n-genes 15] \
      [--celltypist-model Immune_All_Low] \
      [--reference ref.h5ad --ref-label-col cell_type [--ref-counts]] \
      [--adata query.h5ad]

If --adata is given, the merged sets are panel-gated against its var_names: dropped
(off-panel) genes and uncallable labels (<2 on-panel markers) are reported, and a
companion <out stem>_gated.json is written alongside the raw --out.
"""
from __future__ import annotations

from typing import Optional

import typer

from cta.io import eprint, load_adata, panel_gate


def from_celltypist(model_name: str, n_genes: int) -> dict[str, list[str]]:
    from celltypist import models

    from cta.run_celltypist import _ensure_model  # download-on-demand (Model.load won't fetch)
    eprint(f">> CellTypist source: extract_top_markers from model '{model_name}'")
    model_name = _ensure_model(model_name)
    model = models.Model.load(model=model_name)
    out: dict[str, list[str]] = {}
    for ct in model.cell_types:
        out[str(ct)] = [str(g) for g in model.extract_top_markers(ct, n_genes)]
    eprint(f"   {len(out)} labels from CellTypist model")
    return out


def from_reference(path: str, label_col: str, n_genes: int, counts: bool) -> dict[str, list[str]]:
    import scanpy as sc
    eprint(f">> Reference source: rank_genes_groups over '{label_col}' in {path}")
    adata = load_adata(path)
    if label_col not in adata.obs:
        raise SystemExit(
            f"ERROR: --ref-label-col '{label_col}' not in reference obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    if counts:
        eprint("   --ref-counts: normalize_total + log1p before DE")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    # Drop unused categories so empty groups don't break rank_genes_groups.
    adata.obs[label_col] = adata.obs[label_col].astype("category")
    sc.tl.rank_genes_groups(adata, groupby=label_col, method="wilcoxon")
    names = adata.uns["rank_genes_groups"]["names"]
    out: dict[str, list[str]] = {}
    for grp in names.dtype.names:
        out[str(grp)] = [str(g) for g in names[grp][:n_genes]]
    eprint(f"   {len(out)} labels from reference DE")
    return out


def merge(sources: list[dict[str, list[str]]]) -> dict[str, list[str]]:
    """Union per label across sources, dedup preserving first-seen order."""
    merged: dict[str, list[str]] = {}
    for src in sources:
        for label, genes in src.items():
            bucket = merged.setdefault(label, [])
            seen = set(bucket)
            for g in genes:
                if g not in seen:
                    bucket.append(g)
                    seen.add(g)
    return merged


def summarize(markers: dict[str, list[str]], what: str) -> None:
    n = len(markers)
    counts = [len(v) for v in markers.values()]
    mean = sum(counts) / n if n else 0.0
    eprint(f">> {what}: {n} labels, mean {mean:.1f} genes/label")


def cli(
    out: str = typer.Option(..., help="output markers_<level>.json path"),
    n_genes: int = typer.Option(
        15, help="top markers per label per source (default 15)"),
    celltypist_model: Optional[str] = typer.Option(
        None, help="pretrained CellTypist model name (a marker-prior source)"),
    reference: Optional[str] = typer.Option(
        None, help="reference .h5ad for data-driven DE markers"),
    ref_label_col: Optional[str] = typer.Option(
        None, help="obs column in --reference to group by (required with --reference)"),
    ref_counts: bool = typer.Option(
        False, "--ref-counts/--no-ref-counts",
        help="--reference X is raw counts; normalize_total + log1p first"),
    adata: Optional[str] = typer.Option(
        None, help="query .h5ad to panel-gate against (writes a *_gated.json too)"),
) -> None:
    """Assemble a candidate marker prior from data-driven sources — a STARTING POINT."""
    import json
    from pathlib import Path

    sources: list[dict[str, list[str]]] = []
    if celltypist_model:
        sources.append(from_celltypist(celltypist_model, n_genes))
    if reference:
        if not ref_label_col:
            raise SystemExit("ERROR: --reference requires --ref-label-col")
        sources.append(from_reference(reference, ref_label_col,
                                      n_genes, ref_counts))
    if not sources:
        raise SystemExit("ERROR: provide at least one source "
                         "(--celltypist-model and/or --reference)")

    merged = merge(sources)
    summarize(merged, "merged (raw, ungated)")

    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))
    eprint(f"   wrote {out_path}")

    if adata:
        adata_obj = load_adata(adata)
        kept, dropped = panel_gate(merged, adata_obj.var_names)
        if dropped:
            eprint(">> off-panel genes dropped per label:")
            for label, genes in dropped.items():
                eprint(f"   {label}: {len(genes)} dropped -> {genes}")
        uncallable = {lbl: g for lbl, g in kept.items() if len(g) < 2}
        if uncallable:
            eprint(">> UNCALLABLE labels (<2 on-panel markers) — curate or merge these:")
            for label, genes in uncallable.items():
                eprint(f"   {label}: {len(genes)} on-panel -> {genes}")
        gated_path = out_path.with_name(out_path.stem + "_gated" + out_path.suffix)
        gated_path.write_text(json.dumps(kept, indent=2))
        eprint(f"   wrote {gated_path}")
        summarize(kept, "gated (on-panel)")

    eprint(">> done. This is a STARTING POINT — curate labels/genes before annotating.")
