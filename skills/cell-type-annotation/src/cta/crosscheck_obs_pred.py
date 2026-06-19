#!/usr/bin/env python
"""Cross-check from an EXISTING in-object prediction column — no recompute.

Many objects already carry label-transfer / classifier predictions in `obs` (e.g. a
reference atlas mapped on previously: `CellClass_pred`, `Subregion_pred`, a CellTypist
`predicted_labels`, …). When such a column exists and fits the level, prefer it over
re-running `reference_map.py` (CellMapper): it's free and already reconciled with the
object. This script summarizes one or more prediction columns per cluster into the SAME
table format `reference_map.py` emits, so `scaffold_cards.py` / `build_report.py` pick it
up as a cross-check automatically.

For each --pred-cols column it writes tables/refmap_<level>_<col>.csv with columns:
  cluster, top1, frac1, top2, frac2, mean_conf, method
(`mean_conf` from a companion confidence column `<col><--conf-suffix>` if present.)

Usage:
  python crosscheck_obs_pred.py --adata path.h5ad --cluster-key leiden \
      --pred-cols CellClass_pred,Subregion_pred --workdir DIR --level L1 \
      [--conf-suffix _conf]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cta.io import (eprint, load_adata, require_obs, resolve_workdir,
                    safe_cluster_id, save_table, split_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--pred-cols", required=True, help="comma-separated obs prediction columns")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", default="L1")
    ap.add_argument("--conf-suffix", default="_conf",
                    help="suffix of a companion per-cell confidence column, if any")
    args = ap.parse_args()

    wd = resolve_workdir(args.workdir)
    adata = load_adata(args.adata, backed="r")
    require_obs(adata, args.cluster_key, "cluster key")
    obs = adata.obs
    clusters = obs[args.cluster_key].astype(str)

    for col in split_csv(args.pred_cols):
        if col not in obs:
            eprint(f"   WARN: prediction column '{col}' not in obs — skipping")
            continue
        # confidence column conventions vary: '<col>_conf' (e.g. CellClass_pred_conf) OR
        # the base name with the pred-suffix swapped (CellClass_pred -> CellClass_conf).
        cand = [f"{col}{args.conf_suffix}", f"{col}_confidence"]
        for ps in ("_pred", "_prediction", "_pred_label", "_predicted"):
            if col.endswith(ps):
                cand.append(col[: -len(ps)] + args.conf_suffix)
        conf_col = next((c for c in cand if c in obs), None)
        has_conf = conf_col is not None
        rows = []
        for cl, idx in clusters.groupby(clusters).groups.items():
            vc = obs.loc[idx, col].astype(str)
            vc = vc[vc != "nan"].value_counts(normalize=True)
            top1, frac1 = (vc.index[0], round(float(vc.iloc[0]), 3)) if len(vc) else ("NA", 0.0)
            top2, frac2 = (vc.index[1], round(float(vc.iloc[1]), 3)) if len(vc) > 1 else ("", 0.0)
            mean_conf = (round(float(np.nanmean(pd.to_numeric(obs.loc[idx, conf_col],
                         errors="coerce"))), 3) if has_conf else "")
            rows.append(dict(cluster=str(cl), top1=top1, frac1=frac1, top2=top2, frac2=frac2,
                             mean_conf=mean_conf, method=f"in-object:{col}"))
        save_table(pd.DataFrame(rows), wd, f"refmap_{args.level}_{safe_cluster_id(col)}.csv")
        eprint(f">> wrote tables/refmap_{args.level}_{safe_cluster_id(col)}.csv "
               f"({len(rows)} clusters, conf={'yes' if has_conf else 'no'})")


if __name__ == "__main__":
    main()
