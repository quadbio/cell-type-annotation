#!/usr/bin/env python
"""Score curated marker signatures against clusters/cells.

Given marker sets {cell_type: [genes]} (JSON), this:
  1. panel-gates every set against adata.var_names (records dropped/off-panel genes
     and flags candidates left with too few on-panel markers to be falsifiable),
  2. computes a transparent per-CLUSTER overlap of each cluster's top-DE genes with
     each panel-gated marker set (a raw intersection count — NOT decoupler.mt.ora),
  3. computes per-CELL robust scores with decoupler.mt.aucell,
  4. computes per-CELL pyUCell scores (pos+neg markers) if pyUCell is installed,
  5. writes a z-scored AUCell table and a consolidated cross-check (each method's top
     signature per cluster + an agreement count).

Outputs to <workdir>/tables/: panel_gating, scores_overlap, scores_aucell,
scores_aucell_z (per-signature z across clusters), scores_ucell, and crosscheck_<level>.

⚠️ TARGETED/SPATIAL PANELS: ABSOLUTE AUCell across signatures is UNRELIABLE — broadly
expressed marker sets score high in every cluster and can argmax the same broad type
everywhere. Read `scores_aucell_z` (relative enrichment) and `crosscheck`, and treat
filtered-marker specificity (compute_markers.py) + reference mapping (reference_map.py) as primary.
Off-panel genes are NEVER negative evidence; a candidate with too few on-panel markers
is marked UNCALLABLE rather than allowed to score spuriously high.

Marker JSON format:
  {"T cell": ["CD3D","CD3E"], "B cell": ["MS4A1","CD79A"]}
With negative markers (for pyUCell), use the "+"/"-" suffix convention per set:
  {"T cell": {"pos": ["CD3D","CD3E"], "neg": ["MS4A1"]}}

Usage:
  python score_signatures.py --adata path.h5ad --cluster-key leiden \
      --markers markers.json --workdir /proj/ds-annotation-run --level L1 \
      [--lognorm-layer lognorm]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import typer

from cta.constants import MIN_ON_PANEL_MARKERS
from cta.io import (
    ADATA_OPT,
    CLUSTER_KEY_OPT,
    LEVEL_OPT,
    WORKDIR_OPT,
    adata_with_lognorm_X,
    eprint,
    load_adata,
    panel_gate,
    require_obs,
    resolve_workdir,
    save_table,
)


def _split_pos_neg(markers: dict):
    """Accept both flat {ct: [genes]} and {ct: {pos:[], neg:[]}} forms."""
    pos, neg = {}, {}
    for ct, v in markers.items():
        if isinstance(v, dict):
            pos[ct] = list(map(str, v.get("pos", [])))
            neg[ct] = list(map(str, v.get("neg", [])))
        else:
            pos[ct] = list(map(str, v))
            neg[ct] = []
    return pos, neg


def _net_from_sets(sets: dict) -> pd.DataFrame:
    rows = [{"source": ct, "target": g, "weight": 1.0} for ct, genes in sets.items() for g in genes]
    return pd.DataFrame(rows)


def cli(
    adata: str = ADATA_OPT,
    cluster_key: str = CLUSTER_KEY_OPT,
    markers: str = typer.Option(..., help="JSON {type:[genes]} or {type:{pos,neg}}"),
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    lognorm_layer: str | None = typer.Option(
        None, help="layer with log-normalized data; default uses adata.X"
    ),
    min_targets: int = typer.Option(
        1,
        help="decoupler tmin: drop signatures with <= this many on-panel "
        "genes. Curated marker sets are small, so the default is "
        "permissive (1); raise it for noisy/large sets.",
    ),
) -> None:
    """Score curated marker signatures against clusters/cells."""
    import json
    from pathlib import Path

    import scanpy as sc

    wd = resolve_workdir(workdir)
    adata_obj = load_adata(adata)
    require_obs(adata_obj, cluster_key, "cluster key")
    adata_obj = adata_with_lognorm_X(adata_obj, lognorm_layer)
    adata_obj.obs[cluster_key] = adata_obj.obs[cluster_key].astype("category")

    raw = json.loads(Path(markers).expanduser().read_text())
    pos, neg = _split_pos_neg(raw)

    # Panel-gate.
    pos_kept, pos_dropped = panel_gate(pos, adata_obj.var_names)
    _, neg_dropped = panel_gate(neg, adata_obj.var_names)
    dropped_rows = []
    for ct in pos:
        n_on_panel = len(pos_kept[ct])
        dropped_rows.append(
            {
                "cell_type": ct,
                "n_pos": len(pos[ct]),
                "n_pos_on_panel": n_on_panel,
                "uncallable": n_on_panel < MIN_ON_PANEL_MARKERS,
                "dropped_pos": ";".join(pos_dropped.get(ct, [])),
                "dropped_neg": ";".join(neg_dropped.get(ct, [])),
            }
        )
    save_table(pd.DataFrame(dropped_rows), wd, f"panel_gating_{level}.csv")
    uncallable = {ct for ct, g in pos_kept.items() if len(g) < MIN_ON_PANEL_MARKERS}
    if uncallable:
        eprint(
            f"   UNCALLABLE on this panel (<{MIN_ON_PANEL_MARKERS} on-panel markers): "
            f"{sorted(uncallable)} — these contribute to Unknown, not a low score."
        )
    usable = {ct: g for ct, g in pos_kept.items() if g}
    if not usable:
        raise SystemExit("ERROR: no marker genes remain after panel-gating.")

    clusters = list(adata_obj.obs[cluster_key].cat.categories)
    neg_kept, _ = panel_gate(neg, adata_obj.var_names)
    ora_df = auc_bc = uc_bc = None  # captured for the consolidated cross-check below

    # --- per-cluster marker overlap ---
    # Transparent overlap of each cluster's top DE genes with each panel-gated marker
    # set. REUSE the ranked genes compute_markers.py already wrote (markers_<level>.csv)
    # so we don't run a second full Wilcoxon pass per level; fall back to computing them
    # only if that table isn't present.
    try:
        ranked: dict[str, list[str]] = {}
        mk_path = wd / "tables" / f"markers_{level}.csv"
        if mk_path.exists():
            eprint(f"   overlap: reusing {mk_path.name} (skips a redundant Wilcoxon)")
            mk = pd.read_csv(mk_path)
            for g, sub in mk.groupby(mk["cluster"].astype(str)):
                ranked[str(g)] = sub["names"].astype(str).tolist()
        else:
            eprint("   overlap: markers table not found — computing rank_genes_groups")
            # Default-param Wilcoxon, matching `cta markers compute` (no tie_correct) so this
            # fallback can't silently disagree with the canonical markers table.
            sc.tl.rank_genes_groups(
                adata_obj, groupby=cluster_key, method="wilcoxon", use_raw=False
            )
            for g in clusters:
                ranked[str(g)] = (
                    sc.get.rank_genes_groups_df(adata_obj, group=str(g)).head(100)["names"].tolist()
                )
        ora_rows = []
        for g in clusters:
            top = ranked.get(str(g), [])
            for ct, genes in usable.items():
                inter = len(set(top) & set(genes))
                # `frac` is overlap / n_on_panel: a 1-gene set can hit frac=1.0 by
                # chance, so always read it next to n_markers and `uncallable`.
                ora_rows.append(
                    {
                        "cluster": str(g),
                        "cell_type": ct,
                        "overlap": inter,
                        "n_markers": len(genes),
                        "frac": round(inter / max(len(genes), 1), 3),
                        "uncallable": len(genes) < MIN_ON_PANEL_MARKERS,
                    }
                )
        ora_df = pd.DataFrame(ora_rows)
        save_table(ora_df, wd, f"scores_overlap_{level}.csv")
    except Exception as e:  # noqa: BLE001
        eprint(f"   overlap scoring degraded ({type(e).__name__}: {e})")

    # --- decoupler aucell (per-cell, robust rank-AUC) ---
    try:
        import decoupler as dc

        net = _net_from_sets(usable)
        eprint(f">> decoupler aucell (per-cell, tmin={min_targets})")
        dc.mt.aucell(adata_obj, net=net, tmin=min_targets)
        auc = _get_dc_result(adata_obj, "aucell")
        if auc is not None:
            by_cluster = auc.groupby(adata_obj.obs[cluster_key].astype(str).values).mean()
            by_cluster.index.name = "cluster"
            save_table(by_cluster.reset_index(), wd, f"scores_aucell_{level}.csv")
            auc_bc = by_cluster
            # z-score each signature ACROSS clusters: relative enrichment, the correct way
            # to read AUCell across clusters. On a targeted/spatial panel the ABSOLUTE
            # scores are biased toward broadly-expressed signatures (they can argmax the
            # same broad type for every cluster); the z-table is what to compare.
            std = by_cluster.std(ddof=0).replace(0, np.nan)
            z = (by_cluster - by_cluster.mean()) / std
            save_table(z.reset_index(), wd, f"scores_aucell_z_{level}.csv")
        else:
            eprint("   WARN: could not locate aucell result in adata.obsm")
    except Exception as e:  # noqa: BLE001
        eprint(f"   decoupler aucell degraded ({type(e).__name__}: {e})")

    # --- pyUCell (per-cell, pos+neg markers), best-effort ---
    # pyUCell's only edge over aucell is NEGATIVE-marker handling. With no negatives it
    # just duplicates the aucell pass, so skip it then (aucell already covers per-cell).
    has_neg = any(neg_kept.get(ct) for ct in usable)
    if not has_neg:
        eprint("   pyUCell skipped — no negative markers in any set (aucell covers per-cell)")
    try:
        if not has_neg:
            raise ImportError("skip-no-negatives")
        import pyucell

        eprint(">> pyUCell (per-cell, pos+neg)")
        signatures = {}
        for ct, g in usable.items():
            genes = [f"{x}+" for x in g] + [f"{x}-" for x in neg_kept.get(ct, [])]
            signatures[ct] = genes
        pyucell.compute_ucell_scores(adata_obj, signatures=signatures, suffix="_UCell")
        cols = [c for c in adata_obj.obs.columns if c.endswith("_UCell")]
        if cols:
            by = adata_obj.obs.groupby(adata_obj.obs[cluster_key].astype(str))[cols].mean()
            by.index.name = "cluster"
            save_table(by.reset_index(), wd, f"scores_ucell_{level}.csv")
            uc_bc = by
    except ImportError as e:
        if str(e) != "skip-no-negatives":
            eprint("   pyUCell not installed — skipping (aucell covers per-cell scoring)")
    except Exception as e:  # noqa: BLE001
        eprint(f"   pyUCell degraded ({type(e).__name__}: {e})")

    # --- consolidated cross-check (per cluster: each method's TOP signature) ---
    # Reads the transparent DE-overlap, the z-scored AUCell (relative enrichment), and
    # pyUCell side by side, with an agreement count. This is what the annotator should
    # read instead of raw absolute AUCell. Filtered-marker specificity (compute_markers.py) and
    # reference mapping (reference_map.py) remain the primary evidence.
    try:
        cons = pd.DataFrame(index=pd.Index([str(c) for c in clusters], name="cluster"))
        if ora_df is not None and len(ora_df):
            ov = ora_df[~ora_df["uncallable"]]
            if len(ov):
                ovtop = (
                    ov.sort_values(["overlap", "frac"], ascending=False)
                    .groupby("cluster")
                    .head(1)
                    .set_index("cluster")["cell_type"]
                )
                cons["overlap_top"] = ovtop
        if auc_bc is not None:
            cons["aucell_top"] = auc_bc.idxmax(axis=1)  # raw (biased — for reference)
            std = auc_bc.std(ddof=0).replace(0, np.nan)
            cons["aucell_z_top"] = ((auc_bc - auc_bc.mean()) / std).idxmax(axis=1)
        if uc_bc is not None:
            ucc = uc_bc.copy()
            ucc.columns = [c.replace("_UCell", "") for c in ucc.columns]
            cons["ucell_top"] = ucc.idxmax(axis=1)
        winners = [c for c in ["overlap_top", "aucell_z_top", "ucell_top"] if c in cons.columns]
        if winners:
            cons["agree"] = cons[winners].apply(
                lambda r: int(r.dropna().value_counts().iloc[0]) if r.notna().any() else 0, axis=1
            )
        save_table(cons.reset_index(), wd, f"crosscheck_{level}.csv")
    except Exception as e:  # noqa: BLE001
        eprint(f"   consolidated cross-check degraded ({type(e).__name__}: {e})")

    eprint(">> done")


def _get_dc_result(adata, method):
    """decoupler 2.x stores results in adata.obsm under a key containing the method
    name (e.g. 'score_aucell'). Scan for it rather than guessing the exact prefix."""
    for key in adata.obsm:
        if method in key:
            res = adata.obsm[key]
            return (
                res if isinstance(res, pd.DataFrame) else pd.DataFrame(res, index=adata.obs_names)
            )
    return None
