#!/usr/bin/env python
"""Profile a CURATED marker panel across clusters — the workhorse of manual annotation.

`compute_markers.py` finds a cluster's top genes bottom-up (unsupervised DE); this is the
top-down complement: given the markers you ALREADY believe define your candidate types,
show how strongly each cluster expresses them. The output answers the question you actually
reason from at the bench — "is marker X on in cluster Y, and where is it most specific?" —
as two clusters x genes matrices plus a dot plot.

- per-cluster **fraction expressing** (pct, 0–1) — what the dot plot's dot SIZE encodes
- per-cluster **mean log-norm expression** — what the dot plot's COLOR encodes

If the panel carries NEGATIVE markers ({type: {"pos": [...], "neg": [...]}}), their sign is
tracked PER candidate type and a `panel_neg_flags_<level>.csv` flags any negative marker
expressed in more than `NEG_MARKER_MAX_PCT` of a cluster's cells — a marker "on" where the call
says it should be off, i.e. evidence AGAINST that call. Sign is per (type, gene): a gene is
routinely a positive marker of one type and a negative (expected-absent) marker of another, and
both roles are kept. An off-panel negative marker is UNMEASURED (reported, never scored as
"confirmed absent"). A `panel_sign_<level>.csv` records the pos/neg role per (type, gene).

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

import numpy as np
import pandas as pd
import scipy.sparse as sp
import typer

from cta.constants import NEG_MARKER_MAX_PCT
from cta.io import (
    ADATA_OPT,
    CLUSTER_KEY_OPT,
    LEVEL_OPT,
    WORKDIR_OPT,
    eprint,
    load_adata,
    panel_gate,
    require_obs,
    resolve_expression,
    resolve_workdir,
    save_table,
    split_csv,
)


def _load_panel(
    markers: str | None, genes_arg: str | None, genes_file: str | None
) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Return ({group: [genes]}, {group: {negative genes}}) from --markers / --genes / --genes-file.

    A marker JSON may be {type: [genes]}, {type: {"pos": [...], "neg": [...]}}, or a flat list.
    Positive AND negative genes are both profiled (you want to SEE a negative marker is off, not
    assume it), but the SIGN is tracked so profile can flag a negative marker that is expressed
    (evidence AGAINST the call) and render pos/neg as separate blocks. A flat list / inline genes
    are all treated as positive under a single "panel" group.
    """
    if markers:
        raw = json.loads(Path(markers).expanduser().read_text())
        if isinstance(raw, list):
            return {"panel": [str(g) for g in raw]}, {"panel": set()}
        panel: dict[str, list[str]] = {}
        neg_by_group: dict[str, set[str]] = {}
        for ct, v in raw.items():
            ct = str(ct)
            if isinstance(v, dict):
                pos = [str(g) for g in v.get("pos", [])]
                neg = [str(g) for g in v.get("neg", [])]
            else:
                pos, neg = [str(g) for g in v], []
            # positives first so gene2group/gene2sign resolve a gene to its positive role when a
            # curated panel happens to list it under both signs (rare); negatives still profiled.
            panel[ct] = pos + [g for g in neg if g not in pos]
            neg_by_group[ct] = {g for g in neg if g not in pos}
        return panel, neg_by_group
    genes: list[str] = []
    if genes_arg:
        genes += split_csv(genes_arg)
    if genes_file:
        genes += [
            ln.strip()
            for ln in Path(genes_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    if not genes:
        raise SystemExit("ERROR: provide --markers, --genes, or --genes-file.")
    return {"panel": genes}, {"panel": set()}


def cli(
    adata: str = ADATA_OPT,
    cluster_key: str = CLUSTER_KEY_OPT,
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    markers: str | None = typer.Option(None, help="marker JSON ({type:[genes]} / pos-neg / list)"),
    genes: str | None = typer.Option(None, help="comma-separated gene list"),
    genes_file: str | None = typer.Option(None, help="file with one gene per line"),
    lognorm_layer: str | None = typer.Option(
        None, help="layer with log-normalized data; default uses adata.X"
    ),
    dotplot: bool = typer.Option(
        False, "--dotplot/--no-dotplot", help="also render a dot plot PNG"
    ),
) -> None:
    """Profile a CURATED marker panel across clusters — the workhorse of manual annotation."""
    wd = resolve_workdir(workdir)
    adata_obj = load_adata(adata)
    require_obs(adata_obj, cluster_key)

    panel, neg_by_group = _load_panel(markers, genes, genes_file)
    kept, dropped = panel_gate(panel, adata_obj.var_names)
    if dropped:
        for ct, d in dropped.items():
            negs = neg_by_group.get(ct, set())
            d_pos = [g for g in d if g not in negs]
            d_neg = [g for g in d if g in negs]
            if d_pos:
                eprint(f"   off-panel (dropped) in '{ct}': {', '.join(d_pos)}")
            if d_neg:
                # An off-panel negative marker is UNMEASURED, not "confirmed absent" — say so,
                # and it cannot be flagged as a violation (matters doubly on targeted panels).
                eprint(
                    f"   off-panel NEGATIVE marker(s) in '{ct}' — UNMEASURED, not "
                    f"confirmed-absent: {', '.join(d_neg)}"
                )
    # ordered, de-duplicated gene list across groups (first group wins for grouping); the sign
    # is that gene's role in the group it was assigned to (negative only if listed under "neg").
    panel_genes, gene2group, gene2sign, seen = [], {}, {}, set()
    for ct, gs in kept.items():
        for g in gs:
            if g not in seen:
                seen.add(g)
                panel_genes.append(g)
                gene2group[g] = ct
                gene2sign[g] = "neg" if g in neg_by_group.get(ct, set()) else "pos"
    if not panel_genes:
        raise SystemExit("ERROR: no panel genes remain after panel-gating against var_names.")

    # Call for its side effects: validates the layer exists and warns if it isn't log-norm.
    resolve_expression(adata_obj, lognorm_layer)
    sub = adata_obj[:, panel_genes]
    M = sub.layers[lognorm_layer] if lognorm_layer else sub.X
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    df = pd.DataFrame(M, columns=panel_genes)
    cl = adata_obj.obs[cluster_key].astype(str).values
    df["_cl"] = cl
    cats = (
        list(adata_obj.obs[cluster_key].cat.categories)
        if hasattr(adata_obj.obs[cluster_key], "cat")
        else sorted(pd.unique(cl))
    )
    pct = (
        df.groupby("_cl")[panel_genes]
        .apply(lambda d: (d > 0).mean())
        .reindex([str(c) for c in cats])
    )
    mean = df.groupby("_cl")[panel_genes].mean().reindex([str(c) for c in cats])
    pct.index.name = mean.index.name = cluster_key

    save_table(pct.round(4), wd, f"panel_pct_{level}.csv", index=True)
    save_table(mean.round(4), wd, f"panel_mean_{level}.csv", index=True)

    # Sign map — the role each on-panel gene plays PER candidate type. A gene is routinely a
    # POSITIVE marker of one type AND a NEGATIVE (expected-absent) marker of another (e.g. DCX
    # defines neurons and its presence argues against an NPC call), so this is one row per
    # (group, gene) — sign resolved per type, never collapsed to a single role per gene.
    sign_rows = [
        {"group": ct, "gene": g, "sign": "neg" if g in neg_by_group.get(ct, set()) else "pos"}
        for ct, gs in kept.items()
        for g in gs
    ]
    save_table(pd.DataFrame(sign_rows), wd, f"panel_sign_{level}.csv")

    # Negative-marker violations, evaluated PER candidate type: for each type's on-panel negative
    # markers, flag clusters expressing the marker in > NEG_MARKER_MAX_PCT of cells — "on" where a
    # call of that type says it should be off, i.e. evidence AGAINST calling that cluster that type.
    # A gene that is negative for several types is checked once per type (and may be a positive
    # marker of yet another type — the norm), so nothing is lost to a single per-gene sign.
    neg_pairs = [
        (ct, g) for ct, gs in kept.items() for g in gs if g in neg_by_group.get(ct, set())
    ]
    if neg_pairs:
        neg_rows = []
        # NB: `clu`, not `cl` — `cl` holds the per-cell cluster array the dot plot reuses below.
        for clu in pct.index:
            for ct, g in neg_pairs:
                p = float(pct.loc[clu, g])
                neg_rows.append(
                    {
                        "cluster": str(clu),
                        "gene": g,
                        "group": ct,
                        "pct": round(p, 4),
                        "violation": bool(p > NEG_MARKER_MAX_PCT),
                    }
                )
        neg_df = pd.DataFrame(neg_rows)
        save_table(neg_df, wd, f"panel_neg_flags_{level}.csv")
        n_viol = int(neg_df["violation"].sum())
        n_neg_genes = len({g for _, g in neg_pairs})
        eprint(
            f">> negative-marker check: {len(neg_pairs)} type×neg-marker pair(s) "
            f"({n_neg_genes} distinct gene(s)) × {len(pct.index)} cluster(s); "
            f"{n_viol} violation(s) (pct > {NEG_MARKER_MAX_PCT:.0%})"
        )
    else:
        eprint(">> no negative markers in panel — skipping panel_neg_flags")

    if dotplot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import scanpy as sc

            plots = wd / "plots"
            plots.mkdir(parents=True, exist_ok=True)
            # group genes by candidate type so the dot plot reads as type-blocks, and split each
            # type's negatives into their own "(−)" block so expected-absent markers are visually
            # distinct from the defining positives.
            var_names = {}
            for ct in kept:
                pos_g = [g for g in panel_genes if gene2group[g] == ct and gene2sign[g] == "pos"]
                neg_g = [g for g in panel_genes if gene2group[g] == ct and gene2sign[g] == "neg"]
                if pos_g:
                    var_names[ct] = pos_g
                if neg_g:
                    var_names[f"{ct} (−)"] = neg_g
            adata_obj.obs["_cl_cat"] = pd.Categorical(cl, categories=[str(c) for c in cats])
            sc.settings.figdir = str(plots)
            dp = sc.pl.dotplot(
                adata_obj,
                var_names if len(var_names) > 1 else panel_genes,
                groupby="_cl_cat",
                layer=lognorm_layer,
                standard_scale="var",
                show=False,
                return_fig=True,
            )
            outp = plots / f"dotplot_{level}.png"
            dp.savefig(str(outp), dpi=130, bbox_inches="tight")
            plt.close("all")
            eprint(f"   wrote {outp}")
        except Exception as e:  # noqa: BLE001 - a plotting failure shouldn't lose the tables
            eprint(f"   WARN: dot plot failed ({type(e).__name__}: {e}); tables were still written")

    eprint(">> done")
    neg_out = f" neg_flags=tables/panel_neg_flags_{level}.csv" if neg_pairs else ""
    print(
        f"genes_profiled={len(panel_genes)} n_negative={len({g for _, g in neg_pairs})} "
        f"clusters={len(cats)} pct=tables/panel_pct_{level}.csv mean=tables/panel_mean_{level}.csv "
        f"sign=tables/panel_sign_{level}.csv{neg_out}"
    )
