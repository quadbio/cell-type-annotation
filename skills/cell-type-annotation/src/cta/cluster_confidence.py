"""Data-driven confidence proxy per cluster — a sanity check on hand-set confidences.

Confidence in the labels CSV is the annotator's judgement, but it should be calibrated
against objective signals. This computes, per cluster, a cheap separation proxy from the
COSG specificity scores (how distinctive its top markers are) and, if present, the
cross-check agreement, and emits a suggested 0–1 confidence band. The annotator compares
their own value to this and the report flags large mismatches (a high hand-set confidence
on a cluster with a weak marker margin is exactly the thing to re-examine).

Signals per cluster (from tables/cosg_<level>.csv):
  cosg_top1        — specificity of the single most distinctive marker
  cosg_margin      — cosg_top1 - cosg_top2 (how much the top marker stands out)
  n_strong         — number of markers with cosg_score >= --strong-thresh
Plus, if tables/crosscheck_<level>.csv exists: `agree` (how many methods concur).

Writes tables/cluster_confidence_<level>.csv with those columns + `suggested_conf`.
This is advisory only — it never writes labels.
"""
from __future__ import annotations

import typer

from cta.constants import (CONFIDENCE_AGREEMENT_BOOST, CONFIDENCE_BANDS, CONFIDENCE_BASE,
                           CONFIDENCE_CAP)
from cta.io import eprint, resolve_workdir


def cli(
    workdir: str = typer.Option(..., help="annotation-run working directory"),
    level: str = typer.Option("L1", help="hierarchy level tag for filenames"),
    strong_thresh: float = typer.Option(
        0.02, help="COSG score above which a marker counts as 'strong'"),
) -> None:
    """Suggest a per-cluster confidence band from COSG specificity + cross-check agreement."""
    import pandas as pd

    wd = resolve_workdir(workdir)
    cosg_path = wd / "tables" / f"cosg_{level}.csv"
    if not cosg_path.exists():
        raise SystemExit(f"ERROR: {cosg_path} not found (run compute_markers.py first)")
    cosg = pd.read_csv(cosg_path)
    cosg["cluster"] = cosg["cluster"].astype(str)

    cc = wd / "tables" / f"crosscheck_{level}.csv"
    agree = None
    if cc.exists():
        d = pd.read_csv(cc)
        if "agree" in d.columns:
            agree = dict(zip(d["cluster"].astype(str), pd.to_numeric(d["agree"], errors="coerce")))

    rows = []
    for cl, g in cosg.sort_values("rank").groupby("cluster"):
        scores = pd.to_numeric(g["cosg_score"], errors="coerce").dropna().tolist()
        top1 = scores[0] if scores else 0.0
        top2 = scores[1] if len(scores) > 1 else 0.0
        margin = top1 - top2
        n_strong = int(sum(s >= strong_thresh for s in scores))
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
        rows.append(dict(cluster=str(cl), cosg_top1=round(top1, 4),
                         cosg_margin=round(margin, 4), n_strong=n_strong,
                         agree=("" if a is None else a),
                         suggested_conf=round(conf, 2)))

    out = pd.DataFrame(rows)
    out.to_csv(wd / "tables" / f"cluster_confidence_{level}.csv", index=False)
    eprint(f">> wrote tables/cluster_confidence_{level}.csv ({len(out)} clusters)")
    print(out.to_string(index=False))
