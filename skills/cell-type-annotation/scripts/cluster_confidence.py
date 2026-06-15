#!/usr/bin/env python
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

Usage:
  python cluster_confidence.py --workdir DIR --level L1 [--strong-thresh 0.02]
"""
from __future__ import annotations

import argparse

import pandas as pd

from cta_io import eprint, resolve_workdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", default="L1")
    ap.add_argument("--strong-thresh", type=float, default=0.02,
                    help="COSG score above which a marker counts as 'strong'")
    args = ap.parse_args()

    wd = resolve_workdir(args.workdir)
    cosg_path = wd / "tables" / f"cosg_{args.level}.csv"
    if not cosg_path.exists():
        raise SystemExit(f"ERROR: {cosg_path} not found (run compute_markers.py first)")
    cosg = pd.read_csv(cosg_path)
    cosg["cluster"] = cosg["cluster"].astype(str)

    cc = wd / "tables" / f"crosscheck_{args.level}.csv"
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
        n_strong = int(sum(s >= args.strong_thresh for s in scores))
        # crude, transparent banding: strong & well-separated -> high; weak -> low
        conf = 0.5
        if top1 >= 0.05 and n_strong >= 3:
            conf = 0.9
        elif top1 >= 0.03 and n_strong >= 2:
            conf = 0.8
        elif top1 >= 0.02:
            conf = 0.7
        elif top1 >= 0.01:
            conf = 0.6
        else:
            conf = 0.45
        a = agree.get(str(cl)) if agree else None
        if a is not None and not pd.isna(a):
            conf = min(0.95, conf + 0.05 * (a - 1))   # +0.05 per extra concurring method
        rows.append(dict(cluster=str(cl), cosg_top1=round(top1, 4),
                         cosg_margin=round(margin, 4), n_strong=n_strong,
                         agree=("" if a is None else a),
                         suggested_conf=round(conf, 2)))

    out = pd.DataFrame(rows)
    out.to_csv(wd / "tables" / f"cluster_confidence_{args.level}.csv", index=False)
    eprint(f">> wrote tables/cluster_confidence_{args.level}.csv ({len(out)} clusters)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
