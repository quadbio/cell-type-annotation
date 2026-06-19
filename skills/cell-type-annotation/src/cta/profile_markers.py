#!/usr/bin/env python
"""Profile a CURATED marker panel across clusters — the workhorse of manual annotation.

`compute_markers.py` finds a cluster's top genes bottom-up (unsupervised DE); this is the
top-down complement: given the markers you ALREADY believe define your candidate types,
show how strongly each cluster expresses them. The output answers the question you actually
reason from at the bench — "is marker X on in cluster Y, and where is it most specific?" —
as two clusters x genes matrices plus a dot plot.

- per-cluster **fraction expressing** (pct, 0–1) — what the dot plot's dot SIZE encodes
- per-cluster **mean log-norm expression** — what the dot plot's COLOR encodes

Panel-gates against var_names first (off-panel genes are dropped, never treated as absent),
and preserves your type-grouping so the dot plot reads as candidate-type blocks.

Usage:
  # from a marker JSON ({type: [genes]} or {type: {"pos": [...], "neg": [...]}} or a flat list)
  python profile_markers.py --adata work.h5ad --cluster-key cluster \
      --markers markers_L1.json --workdir /proj/run --level L1 [--dotplot]
  # or an inline / file gene list
  python profile_markers.py --adata work.h5ad --cluster-key cluster \
      --genes Snap25,Aqp4,Mbp,Cx3cr1,Cldn5 --workdir /proj/run --level L1 --dotplot
  python profile_markers.py ... --genes-file panel.txt ...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import typer

from cta.io import (eprint, load_adata, panel_gate, require_obs, resolve_expression,
                    resolve_workdir, save_table, split_csv)


def _load_panel(markers: Optional[str], genes_arg: Optional[str],
                genes_file: Optional[str]) -> dict[str, list[str]]:
    """Return {group: [genes]} from --markers (JSON) / --genes / --genes-file.

    A marker JSON may be {type: [genes]}, {type: {"pos": [...], "neg": [...]}}, or a flat
    list. pos+neg are both profiled (you want to SEE a negative marker is off, not assume it).
    A flat list / inline genes land under a single "panel" group.
    """
    if markers:
        raw = json.loads(Path(markers).expanduser().read_text())
        if isinstance(raw, list):
            return {"panel": [str(g) for g in raw]}
        panel: dict[str, list[str]] = {}
        for ct, v in raw.items():
            if isinstance(v, dict):
                genes = list(v.get("pos", [])) + list(v.get("neg", []))
            else:
                genes = list(v)
            panel[str(ct)] = [str(g) for g in genes]
        return panel
    genes: list[str] = []
    if genes_arg:
        genes += split_csv(genes_arg)
    if genes_file:
        genes += [ln.strip() for ln in Path(genes_file).read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")]
    if not genes:
        raise SystemExit("ERROR: provide --markers, --genes, or --genes-file.")
    return {"panel": genes}


def cli(
    adata: str = typer.Option(..., help="path to the .h5ad to analyze"),
    cluster_key: str = typer.Option(..., help="obs column holding the clusters"),
    workdir: str = typer.Option(..., help="annotation-run working directory"),
    level: str = typer.Option("L1", help="hierarchy level tag for filenames"),
    markers: Optional[str] = typer.Option(
        None, help="marker JSON ({type:[genes]} / pos-neg / list)"),
    genes: Optional[str] = typer.Option(None, help="comma-separated gene list"),
    genes_file: Optional[str] = typer.Option(None, help="file with one gene per line"),
    lognorm_layer: Optional[str] = typer.Option(
        None, help="layer with log-normalized data; default uses adata.X"),
    dotplot: bool = typer.Option(
        False, "--dotplot/--no-dotplot", help="also render a dot plot PNG"),
) -> None:
    """Profile a CURATED marker panel across clusters — the workhorse of manual annotation."""
    wd = resolve_workdir(workdir)
    adata_obj = load_adata(adata)
    require_obs(adata_obj, cluster_key)

    panel = _load_panel(markers, genes, genes_file)
    kept, dropped = panel_gate(panel, adata_obj.var_names)
    if dropped:
        for ct, d in dropped.items():
            eprint(f"   off-panel (dropped) in '{ct}': {', '.join(d)}")
    # ordered, de-duplicated gene list across groups (first group wins for grouping)
    panel_genes, gene2group, seen = [], {}, set()
    for ct, gs in kept.items():
        for g in gs:
            if g not in seen:
                seen.add(g); panel_genes.append(g); gene2group[g] = ct
    if not panel_genes:
        raise SystemExit("ERROR: no panel genes remain after panel-gating against var_names.")

    X = resolve_expression(adata_obj, lognorm_layer)
    sub = adata_obj[:, panel_genes]
    M = sub.layers[lognorm_layer] if lognorm_layer else sub.X
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    df = pd.DataFrame(M, columns=panel_genes)
    cl = adata_obj.obs[cluster_key].astype(str).values
    df["_cl"] = cl
    cats = (list(adata_obj.obs[cluster_key].cat.categories)
            if hasattr(adata_obj.obs[cluster_key], "cat")
            else sorted(pd.unique(cl)))
    pct = df.groupby("_cl")[panel_genes].apply(lambda d: (d > 0).mean()).reindex([str(c) for c in cats])
    mean = df.groupby("_cl")[panel_genes].mean().reindex([str(c) for c in cats])
    pct.index.name = mean.index.name = cluster_key

    save_table(pct.round(4), wd, f"panel_pct_{level}.csv", index=True)
    save_table(mean.round(4), wd, f"panel_mean_{level}.csv", index=True)

    if dotplot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import scanpy as sc
            plots = wd / "plots"; plots.mkdir(parents=True, exist_ok=True)
            # group genes by candidate type so the dot plot reads as type-blocks
            var_names = {ct: [g for g in panel_genes if gene2group[g] == ct] for ct in kept
                         if any(gene2group[g] == ct for g in panel_genes)}
            adata_obj.obs["_cl_cat"] = pd.Categorical(cl, categories=[str(c) for c in cats])
            sc.settings.figdir = str(plots)
            dp = sc.pl.dotplot(
                adata_obj, var_names if len(var_names) > 1 else panel_genes, groupby="_cl_cat",
                layer=lognorm_layer, standard_scale="var", show=False, return_fig=True)
            outp = plots / f"dotplot_{level}.png"
            dp.savefig(str(outp), dpi=130, bbox_inches="tight")
            plt.close("all")
            eprint(f"   wrote {outp}")
        except Exception as e:  # noqa: BLE001 - a plotting failure shouldn't lose the tables
            eprint(f"   WARN: dot plot failed ({type(e).__name__}: {e}); tables were still written")

    eprint(">> done")
    print(f"genes_profiled={len(panel_genes)} clusters={len(cats)} "
          f"pct=tables/panel_pct_{level}.csv mean=tables/panel_mean_{level}.csv")
