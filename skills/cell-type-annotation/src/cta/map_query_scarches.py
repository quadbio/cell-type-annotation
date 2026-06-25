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

The query is the canonical AnnData from `cta data canonicalize` (reference/artifacts.md):
RAW COUNTS in layers['counts'], LOG-NORM in X. scVI/scANVI consume counts by default, so
--counts-layer defaults to 'counts' and the script reads from there. --ref-layer says where
the reference model expects its input (default 'counts'; our trained models use 'counts',
many hub models do too — pass 'X' if the model used X). The script places the query matrix
where the model expects it, aligns genes to the model, runs surgery, predicts, aggregates.

LOG-NORM models: if the reference was trained with --gene-likelihood normal (see
train_reference_model.py), it expects LOG-NORM input, not counts — pass --log-norm and
--counts-layer X (the canonical log-norm lives in X). The script verifies the flag against
the loaded model's gene_likelihood and errors on a mismatch (a wrong input scale yields garbage).

Usage:
  # onto a locally-trained model (canonical query: counts in layers['counts'])
  python map_query_scarches.py --query Q_cta.h5ad --cluster-key leiden --workdir DIR \
      --level L1 --model-dir ./ref_scanvi_model [--log-norm --counts-layer X] \
      [--label-col scanvi] [--max-epochs 100] [--accelerator auto|cuda|mps|cpu]

  # onto a shipped hub model
  python map_query_scarches.py --query Q_cta.h5ad --cluster-key leiden --workdir DIR \
      --level L1 --hf-repo scvi-tools/<some-atlas-scanvi>

Writes tables/refmap_<level>_<label-col>.csv with columns:
  cluster, top1, frac1, top2, frac2, mean_conf, method
"""

from __future__ import annotations

import numpy as np
import typer

from cta.io import (
    CLUSTER_KEY_OPT,
    DATALOADER_WORKERS_WARNING,
    LEVEL_OPT,
    WORKDIR_OPT,
    aggregate_predictions,
    apply_scvi_training_best_practices,
    eprint,
    require_obs,
    resolve_accelerator,
    resolve_workdir,
    safe_cluster_id,
    save_table,
    suppressed_warnings,
)


def _place_counts(query, src_layer: str, ref_layer: str, log_norm: bool) -> None:
    """Put the query's input matrix where the reference model expects it.

    Counts for the default likelihoods; log-norm when --log-norm (a 'normal'-likelihood
    reference). The placement is identical either way — only the expected scale differs."""
    expects = "LOG-NORM data" if log_norm else "RAW COUNTS"
    mat = query.X if src_layer in (None, "X") else query.layers.get(src_layer)
    if mat is None:
        raise SystemExit(
            f"ERROR: --counts-layer '{src_layer}' not found in query.layers "
            f"{list(query.layers.keys())}. scArches needs {expects}."
        )
    if ref_layer in (None, "X"):
        query.X = mat
    else:
        query.layers[ref_layer] = mat


def sanitize_query_against_registry(query, setup_args: dict) -> list[str]:
    """Make the query compatible with a shipped scANVI model's registered setup before
    scArches surgery (issue #10).

    Locally-trained models work because the user controls the query setup; shipped hub models
    register a `batch_key`/`labels_key` the real query atlas does not satisfy, which made
    `load_query_data` raise (missing batch column → KeyError; out-of-vocab query labels →
    "Cannot transfer setup without extend_categories"). We reconcile the query against what the
    model registered:

    - batch_key: if the model registered one and it is ABSENT from the query, inject a single
      constant 'query' batch — a brand-new category scArches surgery is designed to absorb. If
      the column is already present we PRESERVE the query's own batches (surgery handles unseen
      categories within it).
    - labels_key: set the registered label column to the model's `unlabeled_category` for ALL
      cells. We want predictions, not to transfer the query's (possibly out-of-vocab) labels,
      and an absent label column would itself break query setup.

    Mutates `query.obs` in place; returns human-readable notes describing what changed.
    """
    notes: list[str] = []
    batch_key = setup_args.get("batch_key")
    labels_key = setup_args.get("labels_key")
    unlabeled = setup_args.get("unlabeled_category")

    if batch_key:
        if batch_key in query.obs.columns:
            notes.append(f"batch_key '{batch_key}' already in query.obs — preserved")
        else:
            query.obs[batch_key] = "query"
            notes.append(f"batch_key '{batch_key}' absent in query — injected constant 'query'")

    if labels_key:
        value = unlabeled if unlabeled is not None else "Unknown"
        query.obs[labels_key] = value
        notes.append(f"labels_key '{labels_key}' set to unlabeled_category '{value}' for all cells")
    return notes


def _model_setup_args(scvi, reference_model) -> dict:
    """Return a scANVI model's registered setup_args (batch_key / labels_key /
    unlabeled_category / ...), whether `reference_model` is a path (local --model-dir) or an
    instantiated model (pulled --hf-repo). Returns {} if the registry can't be read."""
    if isinstance(reference_model, str):
        registry = scvi.model.SCANVI.load_registry(reference_model)
    else:
        registry = getattr(reference_model, "registry_", None)
        if registry is None:
            mgr = getattr(reference_model, "adata_manager", None)
            registry = getattr(mgr, "registry", None) if mgr is not None else None
    if not registry:
        return {}
    return dict(registry.get("setup_args", {}) or {})


def cli(
    query: str = typer.Option(
        ..., "--query", "--adata", help="the query AnnData to annotate (alias: --adata)"
    ),
    cluster_key: str = CLUSTER_KEY_OPT,
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_OPT,
    model_dir: str | None = typer.Option(None, help="local scANVI model dir"),
    hf_repo: str | None = typer.Option(
        None, help="scvi-tools HuggingFace hub repo id to pull a shipped model from"
    ),
    cache_dir: str | None = typer.Option(None, help="cache dir for the pulled hub model"),
    counts_layer: str = typer.Option(
        "counts",
        help="query layer holding the model's input: a canonical query keeps RAW "
        "COUNTS in layers['counts'] (default) and LOG-NORM in 'X' — so pass --counts-layer X "
        "with --log-norm",
    ),
    ref_layer: str = typer.Option(
        "counts", help="layer name the reference model expects its input in ('X' or a name)"
    ),
    log_norm: bool = typer.Option(
        False,
        "--log-norm/--no-log-norm",
        help="the reference was trained with --gene-likelihood normal and expects LOG-NORM "
        "input — pair with --counts-layer X (verified against the loaded model)",
    ),
    label_col: str = typer.Option(
        "scanvi", help="name used in the output table filename (refmap_<level>_<col>.csv)"
    ),
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
        ) from None

    accel = resolve_accelerator(accelerator)
    eprint(f">> accelerator: {accel}")
    apply_scvi_training_best_practices()  # Tensor-Core matmul precision (see io.py)
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
            raise SystemExit(
                "ERROR: scvi.hub unavailable — upgrade scvi-tools to pull "
                "models from the HuggingFace hub."
            ) from None
        eprint(f">> pulling reference model from HuggingFace hub: {hf_repo}")
        hm = HubModel.pull_from_huggingface_hub(
            repo_name=hf_repo, cache_dir=cache_dir, revision="main"
        )
        # Pass the loaded model instance — load_query_data/prepare_query_anndata accept
        # either a path or an instantiated model, and hm.model is the version-stable handle.
        reference_model = hm.model
    else:
        eprint(f">> using local reference model: {model_dir}")
        reference_model = model_dir

    eprint(f">> loading query {query}")
    query_obj = sc.read_h5ad(query)
    require_obs(query_obj, cluster_key, "cluster key")
    if log_norm:
        eprint(
            ">> --log-norm: treating the query input as LOG-NORM "
            "(reference trained with gene_likelihood=normal)"
        )
    _place_counts(query_obj, counts_layer, ref_layer, log_norm)

    # Reconcile the query with what the model registered (issue #10): shipped hub models carry a
    # batch_key/labels_key the real query atlas does not satisfy, which otherwise breaks setup.
    # Setting the registered labels_key to the unlabeled category here also means scvi no longer
    # emits a "Missing labels key … filling" warning during setup (the #14 output-noise concern):
    # we satisfy the precondition rather than suppress the message.
    setup_args = _model_setup_args(scvi, reference_model)
    for note in sanitize_query_against_registry(query_obj, setup_args):
        eprint(f">> registry-sanitize: {note}")

    eprint(">> aligning query genes to the reference model (prepare_query_anndata)")
    scvi.model.SCANVI.prepare_query_anndata(query_obj, reference_model)

    eprint(">> scArches surgery (load_query_data + train)")
    q_model = scvi.model.SCANVI.load_query_data(query_obj, reference_model)

    # The model is now loaded — verify --log-norm matches what it actually expects, since a
    # wrong input scale (counts into a Gaussian model, or vice versa) yields garbage silently.
    glik = getattr(getattr(q_model, "module", None), "gene_likelihood", None)
    if glik is not None:
        model_wants_lognorm = glik == "normal"
        if model_wants_lognorm != log_norm:
            want = (
                "LOG-NORM (pass --log-norm)"
                if model_wants_lognorm
                else "RAW COUNTS (drop --log-norm)"
            )
            raise SystemExit(
                f"ERROR: the reference model's gene_likelihood='{glik}' expects {want}, but "
                f"--log-norm={log_norm}. Re-run with the matching input scale."
            )

    # scArches best practices (scvi tutorial): weight_decay=0.0 keeps the reference latent
    # space fixed; check_val_every_n_epoch=10 tracks surgery convergence. devices=1 pins
    # single-device (our single-process design; also avoids DDP/the srun hint on multi-GPU
    # nodes). The DataLoader-workers hint is benign here (see DATALOADER_WORKERS_WARNING).
    with suppressed_warnings(DATALOADER_WORKERS_WARNING):
        q_model.train(
            max_epochs=max_epochs,
            accelerator=accel,
            devices=1,
            plan_kwargs={"weight_decay": 0.0},
            check_val_every_n_epoch=10,
        )

    eprint(">> predicting labels + uncertainty")
    pred = np.asarray(q_model.predict()).astype(str)
    soft = q_model.predict(soft=True)
    conf = (
        soft.to_numpy().max(axis=1) if hasattr(soft, "to_numpy") else np.asarray(soft).max(axis=1)
    )

    # Persist the per-cell latent + prediction onto the query is optional; the skill
    # reasons from the per-cluster table, so we only write that (no query mutation on disk).
    clusters = query_obj.obs[cluster_key].astype(str).to_numpy()
    out = aggregate_predictions(clusters, pred, conf, "scarches-scanvi")
    wd = resolve_workdir(workdir)
    save_table(out, wd, f"refmap_{level}_{safe_cluster_id(label_col)}.csv")
    eprint(
        f">> done. {len(out)} clusters; mean confidence "
        f"{out['mean_conf'].astype(float).mean():.3f} (treat as a hypothesis)."
    )
