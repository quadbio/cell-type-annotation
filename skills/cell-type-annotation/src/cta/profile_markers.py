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

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from cta.io import (eprint, load_adata, panel_gate, require_obs, resolve_expression,
                    resolve_workdir, save_table, split_csv)


def _load_panel(args) -> dict[str, list[str]]:
    """Return {group: [genes]} from --markers (JSON) / --genes / --genes-file.

    A marker JSON may be {type: [genes]}, {type: {"pos": [...], "neg": [...]}}, or a flat
    list. pos+neg are both profiled (you want to SEE a negative marker is off, not assume it).
    A flat list / inline genes land under a single "panel" group.
    """
    if args.markers:
        raw = json.loads(Path(args.markers).expanduser().read_text())
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
    if args.genes:
        genes += split_csv(args.genes)
    if args.genes_file:
        genes += [ln.strip() for ln in Path(args.genes_file).read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")]
    if not genes:
        raise SystemExit("ERROR: provide --markers, --genes, or --genes-file.")
    return {"panel": genes}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", required=True)
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", default="L1", help="hierarchy level tag for filenames")
    ap.add_argument("--markers", default=None, help="marker JSON ({type:[genes]} / pos-neg / list)")
    ap.add_argument("--genes", default=None, help="comma-separated gene list")
    ap.add_argument("--genes-file", default=None, help="file with one gene per line")
    ap.add_argument("--lognorm-layer", default=None,
                    help="layer with log-normalized data; default uses adata.X")
    ap.add_argument("--dotplot", action="store_true", help="also render a dot plot PNG")
    args = ap.parse_args()

    wd = resolve_workdir(args.workdir)
    adata = load_adata(args.adata)
    require_obs(adata, args.cluster_key)

    panel = _load_panel(args)
    kept, dropped = panel_gate(panel, adata.var_names)
    if dropped:
        for ct, d in dropped.items():
            eprint(f"   off-panel (dropped) in '{ct}': {', '.join(d)}")
    # ordered, de-duplicated gene list across groups (first group wins for grouping)
    genes, gene2group, seen = [], {}, set()
    for ct, gs in kept.items():
        for g in gs:
            if g not in seen:
                seen.add(g); genes.append(g); gene2group[g] = ct
    if not genes:
        raise SystemExit("ERROR: no panel genes remain after panel-gating against var_names.")

    X = resolve_expression(adata, args.lognorm_layer)
    sub = adata[:, genes]
    M = sub.layers[args.lognorm_layer] if args.lognorm_layer else sub.X
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    df = pd.DataFrame(M, columns=genes)
    cl = adata.obs[args.cluster_key].astype(str).values
    df["_cl"] = cl
    cats = (list(adata.obs[args.cluster_key].cat.categories)
            if hasattr(adata.obs[args.cluster_key], "cat")
            else sorted(pd.unique(cl)))
    pct = df.groupby("_cl")[genes].apply(lambda d: (d > 0).mean()).reindex([str(c) for c in cats])
    mean = df.groupby("_cl")[genes].mean().reindex([str(c) for c in cats])
    pct.index.name = mean.index.name = args.cluster_key

    save_table(pct.round(4), wd, f"panel_pct_{args.level}.csv", index=True)
    save_table(mean.round(4), wd, f"panel_mean_{args.level}.csv", index=True)

    if args.dotplot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import scanpy as sc
            plots = wd / "plots"; plots.mkdir(parents=True, exist_ok=True)
            # group genes by candidate type so the dot plot reads as type-blocks
            var_names = {ct: [g for g in genes if gene2group[g] == ct] for ct in kept
                         if any(gene2group[g] == ct for g in genes)}
            adata.obs["_cl_cat"] = pd.Categorical(cl, categories=[str(c) for c in cats])
            sc.settings.figdir = str(plots)
            dp = sc.pl.dotplot(
                adata, var_names if len(var_names) > 1 else genes, groupby="_cl_cat",
                layer=args.lognorm_layer, standard_scale="var", show=False, return_fig=True)
            outp = plots / f"dotplot_{args.level}.png"
            dp.savefig(str(outp), dpi=130, bbox_inches="tight")
            plt.close("all")
            eprint(f"   wrote {outp}")
        except Exception as e:  # noqa: BLE001 - a plotting failure shouldn't lose the tables
            eprint(f"   WARN: dot plot failed ({type(e).__name__}: {e}); tables were still written")

    eprint(">> done")
    print(f"genes_profiled={len(genes)} clusters={len(cats)} "
          f"pct=tables/panel_pct_{args.level}.csv mean=tables/panel_mean_{args.level}.csv")


if __name__ == "__main__":
    main()
