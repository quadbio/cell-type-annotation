"""Data-driven confidence proxy per cluster — a sanity check on hand-set confidences.

Confidence in the labels CSV is the annotator's judgement, but it should be calibrated
against objective signals. This computes, per cluster, a cheap separation proxy from the
specificity of the markers that survive `filter_rank_genes_groups` (how distinctive its top
markers are, measured as log2 fold-change) and, if present, the cross-check agreement, and
emits a suggested 0–1 confidence band. The annotator compares their own value to this and the
report flags large mismatches (a high hand-set confidence on a cluster with a weak marker
margin is exactly the thing to re-examine).

Signals per cluster (from tables/markers_filtered_<level>.csv):
  lfc_top1         — log2FC of the strongest surviving cluster-specific marker
  lfc_margin       — lfc_top1 - lfc_top2 (how much the top marker stands out)
  n_strong         — number of surviving markers with log2FC >= --strong-thresh
Plus, if tables/crosscheck_<level>.csv exists: `agree` (how many methods concur).

Writes tables/cluster_confidence_<level>.csv with those columns + `suggested_conf`.
This is advisory only — it never writes labels.
"""

from __future__ import annotations

import typer

from cta.constants import (
    CONFIDENCE_AGREEMENT_BOOST,
    CONFIDENCE_BANDS,
    CONFIDENCE_BASE,
    CONFIDENCE_CAP,
)
from cta.io import LEVEL_OPT, WORKDIR_OPT, eprint, resolve_workdir


def cli(
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    strong_thresh: float = typer.Option(
        1.0, help="log2FC above which a surviving marker counts as 'strong'"
    ),
) -> None:
    """Suggest a per-cluster confidence band from marker specificity + cross-check agreement."""
    import pandas as pd

    wd = resolve_workdir(workdir)
    markers_path = wd / "tables" / f"markers_filtered_{level}.csv"
    if not markers_path.exists():
        raise SystemExit(f"ERROR: {markers_path} not found (run compute_markers.py first)")
    markers = pd.read_csv(markers_path)
    markers["cluster"] = markers["cluster"].astype(str)

    cc = wd / "tables" / f"crosscheck_{level}.csv"
    agree = None
    if cc.exists():
        d = pd.read_csv(cc)
        if "agree" in d.columns:
            agree = dict(
                zip(
                    d["cluster"].astype(str),
                    pd.to_numeric(d["agree"], errors="coerce"),
                    strict=True,
                )
            )

    rows = []
    for cl, g in markers.groupby("cluster"):
        # Specificity is now log2FC of the surviving (cluster-specific) markers. Sort descending so
        # top1/top2 are the two strongest, regardless of the table's rank order.
        lfc = pd.to_numeric(g["logfoldchange"], errors="coerce").dropna()
        lfc = lfc.sort_values(ascending=False).tolist()
        top1 = lfc[0] if lfc else 0.0
        top2 = lfc[1] if len(lfc) > 1 else 0.0
        margin = top1 - top2
        n_strong = int(sum(s >= strong_thresh for s in lfc))
        # crude, transparent banding (thresholds in cta.constants): strong & well-separated
        # -> high; weak -> low. First matching band wins; CONFIDENCE_BASE if none.
        conf = CONFIDENCE_BASE
        for min_top1, min_strong, band_conf in CONFIDENCE_BANDS:
            if top1 >= min_top1 and n_strong >= min_strong:
                conf = band_conf
                break
        a = agree.get(str(cl)) if agree else None
        if a is not None and not pd.isna(a):
            conf = min(CONFIDENCE_CAP, conf + CONFIDENCE_AGREEMENT_BOOST * (a - 1))
        rows.append(
            {
                "cluster": str(cl),
                "lfc_top1": round(top1, 4),
                "lfc_margin": round(margin, 4),
                "n_strong": n_strong,
                "agree": ("" if a is None else a),
                "suggested_conf": round(conf, 2),
            }
        )

    # Keep a stable header even when no cluster has surviving markers (e.g. filter_rank_genes_groups
    # rejected everything), so the CSV is always readable downstream rather than headerless/empty.
    cols = ["cluster", "lfc_top1", "lfc_margin", "n_strong", "agree", "suggested_conf"]
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(wd / "tables" / f"cluster_confidence_{level}.csv", index=False)
    eprint(f">> wrote tables/cluster_confidence_{level}.csv ({len(out)} clusters)")
    print(out.to_string(index=False))
