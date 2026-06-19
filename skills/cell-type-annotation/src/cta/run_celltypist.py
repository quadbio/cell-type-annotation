#!/usr/bin/env python
"""CellTypist as a cross-check HYPOTHESIS (not a verdict), and a marker-prior source.

CellTypist only helps when a pretrained model actually matches the tissue/context. The
registry has ~60 models; you must NOT download them all to find out what fits. Instead:

  1. `list`     — fetch the remote model index (metadata only, NO downloads) and print
                  each model's name + description (+ optional keyword filter). The agent
                  reads this and decides whether any model matches the tissue.
  2. `predict`  — run ONE chosen model (auto-downloaded on demand if missing) with
                  majority_voting over the user's clusters → cluster-vs-predicted crosstab.
  3. `markers`  — extract_top_markers() from a model WITHOUT predicting (priors).

If NO model matches the tissue, OMIT CellTypist entirely and rely on the reference
mapping (reference_map.py / CellMapper) instead — do not force an ill-fitting model.

On a limited/targeted panel many model genes are missing → accuracy degrades; this is
exactly why CellTypist output is a hypothesis to reconcile, never an overwrite. See
reference/automated_tools.md.

Usage:
  python run_celltypist.py list [--contains brain] [--workdir /proj/run]
  python run_celltypist.py predict --adata path.h5ad --cluster-key leiden \
      --model Developing_Human_Brain --workdir /proj/run --level L1 [--lognorm-layer lognorm]
  python run_celltypist.py markers --model Developing_Human_Brain --workdir /proj/run [--n-genes 20]
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from cta.io import (eprint, load_adata, require_obs, resolve_expression,
                 resolve_workdir, safe_cluster_id, save_table)


def _ensure_model(model: str) -> str:
    """Download the model on demand if needed; return the resolved `.pkl` filename.

    `celltypist.annotate(model=...)` / `Model.load()` expect a LOCAL model and resolve a
    name only if it is in `get_all_models()`, which lists filenames WITH the `.pkl`
    suffix — so a bare name like 'Developing_Human_Brain' fails. We therefore return the
    `.pkl` form. Model.load also does NOT auto-fetch, so we download the one model first.
    """
    from celltypist import models
    fn = model if model.endswith(".pkl") else f"{model}.pkl"
    local = set(os.listdir(models.models_path)) if os.path.isdir(models.models_path) else set()
    if fn in local:
        return fn
    eprint(f">> model '{fn}' not local — downloading just this one")
    try:
        models.download_models(model=[fn], force_update=False)
    except Exception as e:  # noqa: BLE001
        # surface the closest matches so the caller can pick a real one
        eprint(f"   download failed ({type(e).__name__}: {e}).")
        try:
            avail = [m["filename"].replace(".pkl", "") for m in _model_index()]
            stem = model.replace(".pkl", "").lower()
            hits = [a for a in avail if stem in a.lower()]
            eprint("   did you mean: " + (", ".join(hits) if hits else
                   "(none) — run `run_celltypist.py list` to see available models."))
        finally:
            raise SystemExit(f"ERROR: could not obtain CellTypist model '{model}'.")
    return fn


def _model_index() -> list[dict]:
    """Full remote model registry (list of dicts with filename/details/No_celltypes/
    source). Uses get_models_index — `models_description(on_the_fly=True)` is unreliable
    (returns only locally-downloaded models in some celltypist versions)."""
    from celltypist import models
    return models.get_models_index(force_update=True)["models"]


def cmd_list(args):
    """Print the remote model index (metadata only — NO model downloads)."""
    ms = _model_index()
    idx = pd.DataFrame([{
        "model": m["filename"].replace(".pkl", ""),
        "n_celltypes": m.get("No_celltypes", ""),
        "details": m.get("details", ""),
        "source": m.get("source", ""),
    } for m in ms])
    if args.contains:
        kw = args.contains.lower()
        mask = idx.apply(lambda r: kw in f"{r['model']} {r['details']}".lower(), axis=1)
        idx = idx[mask]
    eprint(f">> {len(idx)} model(s)" + (f" matching '{args.contains}'" if args.contains else ""))
    # print full, untruncated, to stdout for the agent to read
    with pd.option_context("display.max_rows", None, "display.max_colwidth", None,
                           "display.width", 200):
        print(idx.to_string(index=False))
    if args.workdir:
        save_table(idx, resolve_workdir(args.workdir), "celltypist_model_index.csv")


def cmd_predict(args):
    import celltypist
    wd = resolve_workdir(args.workdir)
    model = _ensure_model(args.model)
    adata = load_adata(args.adata)
    require_obs(adata, args.cluster_key, "cluster key")
    X = resolve_expression(adata, args.lognorm_layer)
    if args.lognorm_layer is not None:
        adata = adata.copy()
        adata.X = X

    # CellTypist expects log1p of 1e4-normalized counts. resolve_expression() already
    # warned if the chosen matrix doesn't look log1p; CellTypist itself re-checks.
    # IMPORTANT: pass over_clustering=cluster_key so majority voting is computed over
    # the USER's clusters (else CellTypist runs its own internal Leiden and the
    # cluster-vs-prediction crosstab below would compare against the wrong grouping).
    eprint(f">> CellTypist predict with model '{model}' "
           f"(majority_voting over '{args.cluster_key}')")
    pred = celltypist.annotate(adata, model=model, majority_voting=True,
                               over_clustering=args.cluster_key)
    res = pred.predicted_labels.copy()
    res.insert(0, "cell", adata.obs_names)
    res.insert(1, "cluster", adata.obs[args.cluster_key].astype(str).values)
    save_table(res, wd, f"celltypist_pred_{args.level}.csv")

    # cluster -> predicted-label distribution. Summarize the RAW per-cell predictions, NOT
    # the `majority_voting` column: majority voting (over_clustering=cluster_key) forces every
    # cell in a cluster to one label, so a majority-voting crosstab is tautologically 100% per
    # cluster and says nothing about how confident/homogeneous the call is. The raw
    # `predicted_labels` distribution (top label's fraction + runners-up) is the real signal.
    ct = (pd.crosstab(res["cluster"], res["predicted_labels"])
          .apply(lambda r: r.sort_values(ascending=False).head(3).to_dict(), axis=1))
    totals = res.groupby("cluster").size()
    summary = pd.DataFrame({"cluster": ct.index, "top_predicted": ct.values,
                            "n_cells": totals.reindex(ct.index).values})
    save_table(summary, wd, f"celltypist_cluster_summary_{args.level}.csv")
    eprint(">> done")


def cmd_markers(args):
    from celltypist import models
    wd = resolve_workdir(args.workdir)
    model = _ensure_model(args.model)
    eprint(f">> extracting top markers from model '{model}' (no prediction)")
    mdl = models.Model.load(model=model)
    rows = []
    for ct in mdl.cell_types:
        for g in mdl.extract_top_markers(ct, args.n_genes):
            rows.append({"cell_type": ct, "gene": g})
    save_table(pd.DataFrame(rows), wd, f"celltypist_model_markers_{safe_cluster_id(model)}.csv")
    eprint(">> done")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    li = sub.add_parser("list", help="print the remote model index (no downloads)")
    li.add_argument("--contains", default=None, help="case-insensitive keyword filter (e.g. tissue)")
    li.add_argument("--workdir", default=None, help="if given, also save the index as a table")
    li.set_defaults(func=cmd_list)

    p = sub.add_parser("predict")
    p.add_argument("--adata", required=True)
    p.add_argument("--cluster-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--level", default="L1")
    p.add_argument("--lognorm-layer", default=None)
    p.set_defaults(func=cmd_predict)

    m = sub.add_parser("markers")
    m.add_argument("--model", required=True)
    m.add_argument("--workdir", required=True)
    m.add_argument("--n-genes", type=int, default=20)
    m.set_defaults(func=cmd_markers)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
