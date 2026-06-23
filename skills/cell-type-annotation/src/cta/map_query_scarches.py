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

from typing import Optional

import numpy as np
import typer

from cta.io import (aggregate_predictions, eprint, require_obs, resolve_accelerator,
                    resolve_workdir, safe_cluster_id, save_table)


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




def cli(
    query: str = typer.Option(
        ..., "--query", "--adata", help="the query AnnData to annotate (alias: --adata)"),
    cluster_key: str = typer.Option(..., help="obs column holding the clusters"),
    workdir: str = typer.Option(..., help="annotation-run working directory"),
    level: str = typer.Option("L1", help="hierarchy level tag for filenames"),
    model_dir: Optional[str] = typer.Option(None, help="local scANVI model dir"),
    hf_repo: Optional[str] = typer.Option(
        None, help="scvi-tools HuggingFace hub repo id to pull a shipped model from"),
    cache_dir: Optional[str] = typer.Option(None, help="cache dir for the pulled hub model"),
    counts_layer: str = typer.Option(
        "X", help="query layer holding RAW COUNTS now ('X' if X is counts)"),
    ref_layer: str = typer.Option(
        "counts", help="layer name the reference model expects counts in ('X' or a name)"),
    label_col: str = typer.Option(
        "scanvi", help="name used in the output table filename (refmap_<level>_<col>.csv)"),
    max_epochs: int = typer.Option(100, help="surgery training epochs"),
    accelerator: str = typer.Option("auto", help="accelerator: auto|cuda|mps|cpu"),
) -> None:
    """Map the query onto a scANVI reference model via scArches surgery."""
    import scanpy as sc
    if bool(model_dir) == bool(hf_repo):
        raise SystemExit("ERROR: provide exactly one of --model-dir or --hf-repo.")

    try:
        import scvi
    except ImportError:
        raise SystemExit(
            "ERROR: scvi-tools is not installed. It is in the skill's `gpu` pixi env — run "
            "`pixi install -e gpu` (on a GPU host/node). scArches surgery ships inside "
            "scvi-tools; CUDA or Apple-Silicon MPS preferred."
        )

    accel = resolve_accelerator(accelerator)
    eprint(f">> accelerator: {accel}")
    try:
        scvi.settings.seed = 0
    except Exception:  # noqa: BLE001
        pass

    # Resolve the reference model source -> a path or an instantiated model object;
    # both prepare_query_anndata and load_query_data accept either.
    if hf_repo:
        try:
            from scvi.hub import HubModel
        except ImportError:
            raise SystemExit("ERROR: scvi.hub unavailable — upgrade scvi-tools to pull "
                             "models from the HuggingFace hub.")
        eprint(f">> pulling reference model from HuggingFace hub: {hf_repo}")
        hm = HubModel.pull_from_huggingface_hub(
            repo_name=hf_repo, cache_dir=cache_dir, revision="main")
        # Pass the loaded model instance — load_query_data/prepare_query_anndata accept
        # either a path or an instantiated model, and hm.model is the version-stable handle.
        reference_model = hm.model
    else:
        eprint(f">> using local reference model: {model_dir}")
        reference_model = model_dir

    eprint(f">> loading query {query}")
    query_obj = sc.read_h5ad(query)
    require_obs(query_obj, cluster_key, "cluster key")
    _place_counts(query_obj, counts_layer, ref_layer)

    eprint(">> aligning query genes to the reference model (prepare_query_anndata)")
    scvi.model.SCANVI.prepare_query_anndata(query_obj, reference_model)

    eprint(">> scArches surgery (load_query_data + train)")
    q_model = scvi.model.SCANVI.load_query_data(query_obj, reference_model)
    q_model.train(max_epochs=max_epochs, accelerator=accel,
                  plan_kwargs={"weight_decay": 0.0})

    eprint(">> predicting labels + uncertainty")
    pred = np.asarray(q_model.predict()).astype(str)
    soft = q_model.predict(soft=True)
    conf = soft.to_numpy().max(axis=1) if hasattr(soft, "to_numpy") else np.asarray(soft).max(axis=1)

    # Persist the per-cell latent + prediction onto the query is optional; the skill
    # reasons from the per-cluster table, so we only write that (no query mutation on disk).
    clusters = query_obj.obs[cluster_key].astype(str).to_numpy()
    out = aggregate_predictions(clusters, pred, conf, "scarches-scanvi")
    wd = resolve_workdir(workdir)
    save_table(out, wd, f"refmap_{level}_{safe_cluster_id(label_col)}.csv")
    eprint(f">> done. {len(out)} clusters; mean confidence "
           f"{out['mean_conf'].astype(float).mean():.3f} (treat as a hypothesis).")
