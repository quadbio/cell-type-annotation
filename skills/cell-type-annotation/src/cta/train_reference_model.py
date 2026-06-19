#!/usr/bin/env python
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

import argparse
import platform

import scanpy as sc

from cta.io import eprint, require_obs, resolve_workdir, save_table


def _resolve_accelerator(choice: str) -> str:
    """'auto' -> cuda if available, else mps on Apple-Silicon, else cpu."""
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


def main():
    args = _parse_args()
    try:
        import scvi
    except ImportError:
        raise SystemExit(
            "ERROR: scvi-tools is not installed. It is in the skill's `gpu` pixi env — run "
            "`pixi install -e gpu`. (CPU works but is slow; CUDA or Apple-Silicon MPS "
            "is preferred.)"
        )

    accel = _resolve_accelerator(args.accelerator)
    eprint(f">> accelerator: {accel}")
    try:
        scvi.settings.seed = 0  # reproducibility (GPU is still not bit-identical)
    except Exception:  # noqa: BLE001
        pass

    eprint(f">> loading reference {args.reference}")
    ref = sc.read_h5ad(args.reference)
    require_obs(ref, args.labels_key, "labels key")
    if args.batch_key:
        require_obs(ref, args.batch_key, "batch key")

    if args.counts_layer not in (None, "X"):
        if args.counts_layer not in (ref.layers or {}):
            raise SystemExit(
                f"ERROR: counts layer '{args.counts_layer}' not in reference.layers "
                f"{list(ref.layers.keys())}. scVI needs RAW COUNTS.")
        counts_layer = args.counts_layer
    else:
        counts_layer = None  # X holds counts

    if args.n_hvg:
        eprint(f">> selecting {args.n_hvg} highly-variable genes (flavor=seurat_v3 on counts)")
        sc.pp.highly_variable_genes(ref, n_top_genes=args.n_hvg, flavor="seurat_v3",
                                    layer=counts_layer, subset=True,
                                    batch_key=args.batch_key or None)

    eprint(">> setting up scVI (raw counts)")
    scvi.model.SCVI.setup_anndata(ref, layer=counts_layer, batch_key=args.batch_key or None)
    vae = scvi.model.SCVI(ref, n_latent=args.n_latent)
    eprint(f">> training scVI ({args.max_epochs_scvi} epochs)")
    vae.train(max_epochs=args.max_epochs_scvi, accelerator=accel)

    eprint(">> initializing scANVI from the trained scVI model")
    lvae = scvi.model.SCANVI.from_scvi_model(
        vae, adata=ref, labels_key=args.labels_key,
        unlabeled_category=args.unlabeled_category)
    eprint(f">> training scANVI ({args.max_epochs_scanvi} epochs)")
    lvae.train(max_epochs=args.max_epochs_scanvi, accelerator=accel,
               n_samples_per_label=args.n_samples_per_label)

    eprint(f">> saving model to {args.out_model}")
    lvae.save(args.out_model, overwrite=True)

    n_labels = ref.obs[args.labels_key].astype(str).nunique()
    eprint(f">> done. reference={ref.n_obs} cells x {ref.n_vars} genes, "
           f"{n_labels} labels, latent={args.n_latent}, accelerator={accel}")

    if args.workdir:
        wd = resolve_workdir(args.workdir)
        import pandas as pd
        save_table(pd.DataFrame([{
            "model_path": str(args.out_model),
            "n_obs": ref.n_obs, "n_vars": ref.n_vars,
            "labels_key": args.labels_key, "n_labels": n_labels,
            "batch_key": args.batch_key or "", "n_latent": args.n_latent,
            "n_hvg": args.n_hvg or "", "accelerator": accel,
            "epochs_scvi": args.max_epochs_scvi, "epochs_scanvi": args.max_epochs_scanvi,
        }]), wd, "refmodel_summary.csv")

    print(args.out_model)


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, help="labeled reference atlas .h5ad")
    ap.add_argument("--labels-key", required=True, help="reference.obs label column to learn")
    ap.add_argument("--counts-layer", default="X",
                    help="layer holding RAW COUNTS ('X' if X is counts)")
    ap.add_argument("--out-model", required=True, help="output model DIRECTORY")
    ap.add_argument("--batch-key", default=None, help="reference.obs batch column (optional)")
    ap.add_argument("--n-hvg", type=int, default=None,
                    help="restrict to N highly-variable genes for a compact feature space")
    ap.add_argument("--n-latent", type=int, default=30)
    ap.add_argument("--max-epochs-scvi", type=int, default=200)
    ap.add_argument("--max-epochs-scanvi", type=int, default=20)
    ap.add_argument("--n-samples-per-label", type=int, default=100)
    ap.add_argument("--unlabeled-category", default="Unknown",
                    help="label value treated as 'unlabeled' by scANVI")
    ap.add_argument("--accelerator", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--workdir", default=None,
                    help="if given, also write tables/refmodel_summary.csv")
    return ap.parse_args()


if __name__ == "__main__":
    main()
