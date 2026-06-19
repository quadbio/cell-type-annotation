#!/usr/bin/env python
"""Map the query onto a scANVI reference model via scArches surgery — a GPU/large-memory
cross-check HYPOTHESIS (not a verdict).

scArches freezes the reference network and trains a few query-specific weights, so the
query is embedded into the reference's latent space and gets per-cell label predictions
WITH uncertainty — without retraining the reference and without needing the reference's
cells. The per-cluster summary lands in the SAME schema reference_map.py / crosscheck_obs_pred.py
emit, so scaffold_cards.py / build_report.py pick it up as one more cross-check automatically.
As with every automated method here: flag agreement/disagreement with the marker evidence,
never overwrite it (reference/automated_tools.md, reference/methodology.md §3-4).

The reference model is EITHER:
  --model-dir DIR   a scANVI model trained locally (train_reference_model.py), or
  --hf-repo  REPO   a shipped model pulled from the scvi-tools HuggingFace hub
                    (e.g. a tissue-matched scANVI atlas) — preferred when one fits, since
                    it needs no retraining.

scVI/scANVI consume RAW COUNTS. --counts-layer says where the query's counts are now;
--ref-layer says where the reference model expects them (default 'counts'; our trained
models use 'counts', many hub models do too — pass 'X' if the model used X). The script
places the query counts where the model expects them, aligns genes to the model, runs
surgery, predicts, and aggregates per cluster.

Usage:
  # onto a locally-trained model
  python map_query_scarches.py --query Q.h5ad --cluster-key leiden --workdir DIR \
      --level L1 --model-dir ./ref_scanvi_model --counts-layer counts \
      [--label-col scanvi] [--max-epochs 100] [--accelerator auto|cuda|mps|cpu]

  # onto a shipped hub model
  python map_query_scarches.py --query Q.h5ad --cluster-key leiden --workdir DIR \
      --level L1 --hf-repo scvi-tools/<some-atlas-scanvi> --counts-layer counts

Writes tables/refmap_<level>_<label-col>.csv with columns:
  cluster, top1, frac1, top2, frac2, mean_conf, method
"""
from __future__ import annotations

import argparse
import platform

import numpy as np
import scanpy as sc

from cta.io import (aggregate_predictions, eprint, require_obs, resolve_workdir, safe_cluster_id,
                    save_table)


def _resolve_accelerator(choice: str) -> str:
    if choice != "auto":
        return choice
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if (platform.system() == "Darwin" and platform.machine() == "arm64"
                and torch.backends.mps.is_available()):
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _place_counts(query, src_layer: str, ref_layer: str) -> None:
    """Put the query's raw counts where the reference model expects them."""
    counts = query.X if src_layer in (None, "X") else query.layers.get(src_layer)
    if counts is None:
        raise SystemExit(
            f"ERROR: --counts-layer '{src_layer}' not found in query.layers "
            f"{list(query.layers.keys())}. scArches needs RAW COUNTS.")
    if ref_layer in (None, "X"):
        query.X = counts
    else:
        query.layers[ref_layer] = counts




def main():
    args = _parse_args()
    if bool(args.model_dir) == bool(args.hf_repo):
        raise SystemExit("ERROR: provide exactly one of --model-dir or --hf-repo.")

    try:
        import scvi
    except ImportError:
        raise SystemExit(
            "ERROR: scvi-tools is not installed. It is in the skill's `gpu` pixi env — run "
            "`pixi install -e gpu` (on a GPU host/node). scArches surgery ships inside "
            "scvi-tools; CUDA or Apple-Silicon MPS preferred."
        )

    accel = _resolve_accelerator(args.accelerator)
    eprint(f">> accelerator: {accel}")
    try:
        scvi.settings.seed = 0
    except Exception:  # noqa: BLE001
        pass

    # Resolve the reference model source -> a path or an instantiated model object;
    # both prepare_query_anndata and load_query_data accept either.
    if args.hf_repo:
        try:
            from scvi.hub import HubModel
        except ImportError:
            raise SystemExit("ERROR: scvi.hub unavailable — upgrade scvi-tools to pull "
                             "models from the HuggingFace hub.")
        eprint(f">> pulling reference model from HuggingFace hub: {args.hf_repo}")
        hm = HubModel.pull_from_huggingface_hub(
            repo_name=args.hf_repo, cache_dir=args.cache_dir, revision="main")
        # Pass the loaded model instance — load_query_data/prepare_query_anndata accept
        # either a path or an instantiated model, and hm.model is the version-stable handle.
        reference_model = hm.model
    else:
        eprint(f">> using local reference model: {args.model_dir}")
        reference_model = args.model_dir

    eprint(f">> loading query {args.query}")
    query = sc.read_h5ad(args.query)
    require_obs(query, args.cluster_key, "cluster key")
    _place_counts(query, args.counts_layer, args.ref_layer)

    eprint(">> aligning query genes to the reference model (prepare_query_anndata)")
    scvi.model.SCANVI.prepare_query_anndata(query, reference_model)

    eprint(">> scArches surgery (load_query_data + train)")
    q_model = scvi.model.SCANVI.load_query_data(query, reference_model)
    q_model.train(max_epochs=args.max_epochs, accelerator=accel,
                  plan_kwargs={"weight_decay": 0.0})

    eprint(">> predicting labels + uncertainty")
    pred = np.asarray(q_model.predict()).astype(str)
    soft = q_model.predict(soft=True)
    conf = soft.to_numpy().max(axis=1) if hasattr(soft, "to_numpy") else np.asarray(soft).max(axis=1)

    # Persist the per-cell latent + prediction onto the query is optional; the skill
    # reasons from the per-cluster table, so we only write that (no query mutation on disk).
    clusters = query.obs[args.cluster_key].astype(str).to_numpy()
    out = aggregate_predictions(clusters, pred, conf, "scarches-scanvi")
    wd = resolve_workdir(args.workdir)
    save_table(out, wd, f"refmap_{args.level}_{safe_cluster_id(args.label_col)}.csv")
    eprint(f">> done. {len(out)} clusters; mean confidence "
           f"{out['mean_conf'].astype(float).mean():.3f} (treat as a hypothesis).")


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", "--adata", dest="query", required=True,
                    help="the query AnnData to annotate (alias: --adata)")
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", default="L1")
    ap.add_argument("--model-dir", default=None, help="local scANVI model dir")
    ap.add_argument("--hf-repo", default=None,
                    help="scvi-tools HuggingFace hub repo id to pull a shipped model from")
    ap.add_argument("--cache-dir", default=None, help="cache dir for the pulled hub model")
    ap.add_argument("--counts-layer", default="X",
                    help="query layer holding RAW COUNTS now ('X' if X is counts)")
    ap.add_argument("--ref-layer", default="counts",
                    help="layer name the reference model expects counts in ('X' or a name)")
    ap.add_argument("--label-col", default="scanvi",
                    help="name used in the output table filename (refmap_<level>_<col>.csv)")
    ap.add_argument("--max-epochs", type=int, default=100, help="surgery training epochs")
    ap.add_argument("--accelerator", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    return ap.parse_args()


if __name__ == "__main__":
    main()
