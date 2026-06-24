#!/usr/bin/env python
"""Stage 1: inspect an AnnData and verify the keys/layers the skill will rely on.

Prints a structured report (and optionally writes it as JSON) that Claude uses to
write context.md and reconcile any mismatch with the user BEFORE annotation starts.
Verifies nothing is silently wrong: cluster key, embedding key, viz embedding, and
which layer is counts vs log-normalized.

This command does NOT author `context.md` or `run_state.json` — those are written by
Claude (the annotator) from the report this prints. Pass `--workdir` to also drop the
machine-readable report at `<workdir>/inspect_report.json` so it is a durable artifact a
resumed session (or a critic subagent) can re-read.

Usage:
  python inspect_adata.py --adata path.h5ad [--workdir DIR] \
      [--cluster-key leiden] [--embedding-key X_scpoli] [--viz-key X_umap] \
      [--counts-layer counts] [--lognorm-layer lognorm] \
      [--json OUT.json]
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import typer

from cta.io import eprint, load_adata, resolve_workdir


def _nonzero_data(X, cap=200_000):
    """Flatten up to `cap` stored values (nonzeros for sparse) for distribution tests."""
    import scipy.sparse as sp
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    return data[:cap] if data.size > cap else data


def _matrix_stats(X):
    xmin, xmax = float(X.min()), float(X.max())
    looks = "scaled?" if xmin < 0 else ("counts?" if xmax > 50 else "log1p?")
    return {"min": round(xmin, 3), "max": round(xmax, 3), "looks_like": looks}


def _fingerprint(X):
    """Distinguish raw counts / size-normalized-log1p / mis-normalized data.

    The `max>50` heuristic alone passes the most dangerous case: `log1p(counts)` with
    NO size-factor normalization (modest, non-negative values) reads as 'log1p' but its
    markers track sequencing depth / cell size, not identity. We separate the cases
    with two cheap distributional probes:

    - integer-valued stored entries  -> raw counts (or CPM-as-int), never log1p
    - expm1(X).sum(1) per-cell CV     -> ~0 iff every cell was normalized to the same
      total BEFORE log1p (the signature of proper lognorm); large iff log1p was applied
      to un-normalized counts.
    """
    import scipy.sparse as sp

    data = _nonzero_data(X)
    is_int = bool(data.size and np.allclose(data, np.round(data)))

    # expm1-sum coefficient of variation across cells (recovers pre-log per-cell totals)
    Xe = X.copy()
    if sp.issparse(Xe):
        Xe.data = np.expm1(Xe.data)
        s = np.asarray(Xe.sum(axis=1)).ravel()
    else:
        s = np.expm1(np.asarray(Xe)).sum(axis=1).ravel()
    s = s[np.isfinite(s) & (s > 0)]
    cv = float(s.std() / s.mean()) if s.size else float("nan")

    xmin = float(X.min())
    if xmin < 0:
        verdict = "scaled (z-scored); NOT usable as lognorm"
    elif is_int:
        # log1p output is essentially never integer-valued; integer entries mean raw
        # (or integer-rounded) counts, regardless of the maximum value.
        verdict = "raw counts (integer); NOT lognorm"
    elif np.isfinite(cv) and cv < 0.05:
        verdict = "size-normalized + log1p (OK for markers)"
    elif np.isfinite(cv) and cv >= 0.05:
        verdict = ("log1p of UN-normalized counts? markers would track depth/cell size "
                   "(expm1-sum CV high)")
    else:
        verdict = "indeterminate"
    return {"integer_valued": is_int, "expm1_sum_cv": (round(cv, 4) if np.isfinite(cv) else None),
            "verdict": verdict}


def cli(
    adata: str = typer.Option(..., help="path to the .h5ad to inspect"),
    cluster_key: Optional[str] = typer.Option(
        None, help="obs column with the cluster labels"),
    embedding_key: Optional[str] = typer.Option(
        None, help="obsm key for the analysis embedding"),
    viz_key: Optional[str] = typer.Option(
        None, help="obsm key for a 2D viz embedding"),
    counts_layer: Optional[str] = typer.Option(
        None, help="layer expected to hold raw counts"),
    lognorm_layer: Optional[str] = typer.Option(
        None, help="layer expected to hold log-normalized data"),
    workdir: Optional[str] = typer.Option(
        None, help="annotation-run working dir; if given, also write "
                   "<workdir>/inspect_report.json (context.md/run_state.json are authored "
                   "by Claude, not here)"),
    json: Optional[str] = typer.Option(
        None, help="also write the report to this explicit JSON path"),
) -> None:
    """Inspect an AnnData and verify the keys/layers the skill will rely on."""
    import json as json_mod

    adata_obj = load_adata(adata)
    report: dict = {
        "adata_path": adata,
        "n_obs": int(adata_obj.n_obs),
        "n_vars": int(adata_obj.n_vars),
        "obs_columns": list(adata_obj.obs.columns),
        "obsm_keys": list(adata_obj.obsm.keys()),
        "layers": list(adata_obj.layers.keys()),
        "uns_keys": list(adata_obj.uns.keys()),
        "checks": {},
        "warnings": [],
        "fatal": [],
    }

    # X stats
    report["X"] = _matrix_stats(adata_obj.X)
    report["layer_stats"] = {l: _matrix_stats(adata_obj.layers[l]) for l in adata_obj.layers}

    def check(label, ok, detail=""):
        report["checks"][label] = {"ok": bool(ok), "detail": detail}
        if not ok:
            report["warnings"].append(f"{label}: {detail}")

    if cluster_key:
        ok = cluster_key in adata_obj.obs
        detail = ""
        if ok:
            vc = adata_obj.obs[cluster_key].value_counts()
            report["cluster_sizes"] = {str(k): int(v) for k, v in vc.items()}
            report["n_clusters"] = int(vc.shape[0])
            detail = f"{vc.shape[0]} clusters, sizes {int(vc.min())}..{int(vc.max())}"
        else:
            detail = f"not in obs; have {list(adata_obj.obs.columns)}"
        check(f"cluster_key '{cluster_key}'", ok, detail)

    for key, kind in [(embedding_key, "embedding"), (viz_key, "viz_embedding")]:
        if key:
            ok = key in adata_obj.obsm
            detail = (f"shape {tuple(adata_obj.obsm[key].shape)}" if ok
                      else f"not in obsm; have {list(adata_obj.obsm.keys())}")
            check(f"{kind}_key '{key}'", ok, detail)

    report.setdefault("normalization", {})
    for key, kind in [(counts_layer, "counts"), (lognorm_layer, "lognorm")]:
        if key:
            ok = key in adata_obj.layers
            if ok:
                st = report["layer_stats"][key]
                fp = _fingerprint(adata_obj.layers[key])
                report["normalization"][key] = {"declared_as": kind, **fp}
                detail = f"looks_like={st['looks_like']}; fingerprint={fp['verdict']}"
                if kind == "counts" and not fp["integer_valued"]:
                    report["warnings"].append(
                        f"counts-layer '{key}' is not integer-valued ({fp['verdict']})")
                # A declared lognorm layer that is NOT size-normalized-then-log1p is a
                # hard stop: every downstream marker would track depth, not identity.
                if kind == "lognorm" and not fp["verdict"].startswith("size-normalized"):
                    report["fatal"].append(
                        f"lognorm-layer '{key}' fails the normalization fingerprint "
                        f"({fp['verdict']}). Markers computed from it would be invalid.")
            else:
                detail = f"not in layers; have {list(adata_obj.layers.keys())}"
            check(f"{kind}_layer '{key}'", ok, detail)

    # If no explicit lognorm layer was given, fingerprint X (the default analysis matrix).
    if not lognorm_layer:
        report["normalization"]["X"] = {"declared_as": "default(X)", **_fingerprint(adata_obj.X)}

    # Surface obs columns that look like existing annotations/predictions (priors).
    hints = [c for c in adata_obj.obs.columns
             if any(t in c.lower() for t in
                    ("annot", "celltype", "cell_type", "_pred", "label", "class",
                     "subtype", "snap", "leiden", "louvain", "cluster"))]
    report["annotation_like_obs"] = hints

    # Targeted-panel flag: with few genes (imaging panels: MERSCOPE/Xenium/CosMx/Visium-HD
    # custom), marker ABSENCE is not evidence of absence and many canonical markers are
    # off-panel. When true, the annotator should (Stage 1/2) record a short panel-capability
    # note — which expected identities the panel CANNOT resolve (no canonical markers on it).
    report["targeted_panel"] = bool(adata_obj.n_vars <= 2000)
    if report["targeted_panel"]:
        report.setdefault("warnings", []).append(
            f"targeted panel: only {adata_obj.n_vars} genes — absence != negative; record a "
            "panel-capability note (which identities are uncallable on this panel).")

    # Categorical obs columns (candidate composition covariates) + whether each already
    # has a stored colour palette in uns (build_report.py reuses these). Helps the
    # annotator ask the user which covariates to profile composition over (Stage 1c).
    cat_cols = [c for c in adata_obj.obs.columns
                if str(adata_obj.obs[c].dtype) in ("category", "object", "bool")
                or adata_obj.obs[c].dtype == bool]
    report["categorical_obs"] = {
        c: {"n_categories": int(adata_obj.obs[c].nunique()),
            "has_uns_colors": f"{c}_colors" in adata_obj.uns}
        for c in cat_cols if adata_obj.obs[c].nunique() <= 200}

    print(json_mod.dumps(report, indent=2))
    if workdir:
        out = resolve_workdir(workdir) / "inspect_report.json"
        out.write_text(json_mod.dumps(report, indent=2))
        eprint(f"\n>> wrote {out}  (author context.md / run_state.json from this)")
    if json:
        with open(json, "w") as fh:
            json_mod.dump(report, fh, indent=2)
        eprint(f"\n>> wrote {json}")
    if report["warnings"]:
        eprint("\n>> WARNINGS:")
        for w in report["warnings"]:
            eprint(f"   - {w}")
    if report["fatal"]:
        eprint("\n>> FATAL (do NOT proceed to marker computation — reconcile with the user):")
        for w in report["fatal"]:
            eprint(f"   - {w}")
