"""Train (or refine) a scANVI reference model on a LABELED atlas — the reference side of
the train-a-model-then-map-the-query workflow (GPU / large-memory profiles).

scANVI is a semi-supervised VAE: it learns a batch-corrected latent space AND a label
classifier from the reference's annotations, so a query mapped onto it later (scArches
surgery — see map_query_scarches.py) gets per-cell label predictions WITH calibrated
uncertainty. The transferred labels remain a HYPOTHESIS to reconcile against marker
evidence, never a verdict (reference/compute_environments.md, reference/automated_tools.md).

This is heavy compute: GPU strongly preferred (CUDA, or Apple-Silicon MPS for a deeper
local option); it runs on CPU but slowly. It NEVER submits a job — on a scheduler profile
the skill arranges submission elsewhere and just invokes this script there.

scVI/scANVI consume RAW COUNTS — point --counts-layer at the reference's count layer
(or 'X' if X is counts). Optionally restrict to highly-variable genes (--n-hvg) for a
compact, transferable feature space.

Usage:
  python train_reference_model.py --reference atlas.h5ad --labels-key cell_type \
      --counts-layer counts --out-model ./ref_scanvi_model \
      [--batch-key donor] [--n-hvg 3000] [--n-latent 30] \
      [--max-epochs-scvi 200] [--max-epochs-scanvi 20] \
      [--accelerator auto|cuda|mps|cpu] [--workdir DIR]

Writes the model to --out-model (a directory) and, if --workdir is given,
tables/refmodel_summary.csv.
"""
from __future__ import annotations

from typing import Optional

import typer

from cta.io import (DATALOADER_WORKERS_WARNING, apply_scvi_training_best_practices,
                    eprint, require_obs, resolve_accelerator, resolve_workdir,
                    save_table, suppressed_warnings)


def cli(
    reference: str = typer.Option(..., help="labeled reference atlas .h5ad"),
    labels_key: str = typer.Option(..., help="reference.obs label column to learn"),
    out_model: str = typer.Option(..., help="output model DIRECTORY"),
    counts_layer: str = typer.Option(
        "X", help="layer holding RAW COUNTS ('X' if X is counts)"),
    batch_key: Optional[str] = typer.Option(
        None, help="reference.obs batch column (optional)"),
    n_hvg: Optional[int] = typer.Option(
        None, help="restrict to N highly-variable genes for a compact feature space"),
    n_latent: int = typer.Option(30, help="latent space dimensionality"),
    max_epochs_scvi: int = typer.Option(200, help="scVI training epochs"),
    max_epochs_scanvi: int = typer.Option(20, help="scANVI training epochs"),
    n_samples_per_label: int = typer.Option(100, help="scANVI samples per label"),
    unlabeled_category: str = typer.Option(
        "Unknown", help="label value treated as 'unlabeled' by scANVI"),
    accelerator: str = typer.Option(
        "auto", help="accelerator: auto|cuda|mps|cpu"),
    workdir: Optional[str] = typer.Option(
        None, help="if given, also write tables/refmodel_summary.csv"),
) -> None:
    """Train (or refine) a scANVI reference model on a LABELED atlas."""
    import scanpy as sc
    try:
        import scvi
    except ImportError:
        raise SystemExit(
            "ERROR: scvi-tools is not installed. It is in the skill's `gpu` pixi env — run "
            "`pixi install -e gpu`. (CPU works but is slow; CUDA or Apple-Silicon MPS "
            "is preferred.)"
        )

    accel = resolve_accelerator(accelerator)
    eprint(f">> accelerator: {accel}")
    apply_scvi_training_best_practices()  # Tensor-Core matmul precision (see io.py)
    try:
        scvi.settings.seed = 0  # reproducibility (GPU is still not bit-identical)
    except Exception:  # noqa: BLE001
        pass

    eprint(f">> loading reference {reference}")
    ref = sc.read_h5ad(reference)
    require_obs(ref, labels_key, "labels key")
    if batch_key:
        require_obs(ref, batch_key, "batch key")

    if counts_layer not in (None, "X"):
        if counts_layer not in (ref.layers or {}):
            raise SystemExit(
                f"ERROR: counts layer '{counts_layer}' not in reference.layers "
                f"{list(ref.layers.keys())}. scVI needs RAW COUNTS.")
        counts_layer_resolved = counts_layer
    else:
        counts_layer_resolved = None  # X holds counts

    if n_hvg:
        eprint(f">> selecting {n_hvg} highly-variable genes (flavor=seurat_v3 on counts)")
        sc.pp.highly_variable_genes(ref, n_top_genes=n_hvg, flavor="seurat_v3",
                                    layer=counts_layer_resolved, subset=True,
                                    batch_key=batch_key or None)

    eprint(">> setting up scVI (raw counts)")
    scvi.model.SCVI.setup_anndata(ref, layer=counts_layer_resolved, batch_key=batch_key or None)
    # scArches best-practice settings for query mapping (scvi-tools scArches tutorial):
    # encode_covariates=True is "essential" for mapping a new-batch query; the layer-norm /
    # batch-norm / deep-injection settings improve nuisance-variation removal during surgery.
    vae = scvi.model.SCVI(
        ref,
        n_latent=n_latent,
        encode_covariates=True,
        deeply_inject_covariates=True,
        use_layer_norm="both",
        use_batch_norm="none",
    )
    eprint(f">> training scVI ({max_epochs_scvi} epochs)")
    # devices=1: pin single-device training (our design is single-process). On a multi-GPU
    # node this also stops Lightning attempting DDP — which is what triggers its 'prepend
    # srun' hint. The DataLoader-workers hint is benign here (see DATALOADER_WORKERS_WARNING).
    with suppressed_warnings(DATALOADER_WORKERS_WARNING):
        vae.train(max_epochs=max_epochs_scvi, accelerator=accel, devices=1)

    eprint(">> initializing scANVI from the trained scVI model")
    lvae = scvi.model.SCANVI.from_scvi_model(
        vae, adata=ref, labels_key=labels_key,
        unlabeled_category=unlabeled_category)
    eprint(f">> training scANVI ({max_epochs_scanvi} epochs)")
    with suppressed_warnings(DATALOADER_WORKERS_WARNING):  # benign num_workers hint
        lvae.train(max_epochs=max_epochs_scanvi, accelerator=accel, devices=1,
                   n_samples_per_label=n_samples_per_label)

    eprint(f">> saving model to {out_model}")
    lvae.save(out_model, overwrite=True)

    n_labels = ref.obs[labels_key].astype(str).nunique()
    eprint(f">> done. reference={ref.n_obs} cells x {ref.n_vars} genes, "
           f"{n_labels} labels, latent={n_latent}, accelerator={accel}")

    if workdir:
        wd = resolve_workdir(workdir)
        import pandas as pd
        save_table(pd.DataFrame([{
            "model_path": str(out_model),
            "n_obs": ref.n_obs, "n_vars": ref.n_vars,
            "labels_key": labels_key, "n_labels": n_labels,
            "batch_key": batch_key or "", "n_latent": n_latent,
            "n_hvg": n_hvg or "", "accelerator": accel,
            "epochs_scvi": max_epochs_scvi, "epochs_scanvi": max_epochs_scanvi,
        }]), wd, "refmodel_summary.csv")

    print(out_model)
