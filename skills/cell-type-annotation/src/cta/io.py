"""Shared helpers for the cell-type-annotation scripts.

Keep this dependency-light (scanpy/anndata/pandas/numpy). Every script imports from
here so the artifact conventions (working dir, table output, layer resolution,
panel-gating) stay consistent and the run remains resumable and auditable.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import typer

# Shared CLI options — reuse one OptionInfo per recurring flag so help text stays uniform
# across commands instead of being re-typed (and drifting) in every module. Reuse only for
# identically-named parameters so Typer infers the right flag name from the parameter.
WORKDIR_OPT = typer.Option(..., help="annotation-run working directory")
ADATA_OPT = typer.Option(..., help="path to the .h5ad to analyze")
CLUSTER_KEY_OPT = typer.Option(..., help="obs column holding the clusters")
LEVEL_OPT = typer.Option("L1", help="hierarchy level tag, e.g. L1")
LEVEL_REQ_OPT = typer.Option(..., help="hierarchy level tag, e.g. L1")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# Exact messages of third-party warnings we have DIAGNOSED as benign-and-unfixable-here,
# so callers can silence them by the tightest possible scope (one `with` around the one
# offending call) instead of muting a whole command. Never use these to hide a warning we
# haven't traced to a specific, harmless cause.
#   - scanpy builds its rank_genes_groups results frame one column per group, so pandas
#     emits this once per cluster (scanpy/tools/_rank_genes_groups.py) — internal to
#     scanpy, no effect on the output, nothing we can fix without patching scanpy.
FRAGMENTED_FRAME_WARNING = r".*DataFrame is highly fragmented.*"
#   - PyTorch-Lightning hints to raise `num_workers` on scvi's DataLoader. We deliberately
#     keep scvi's default (dl_num_workers=0): maintainers report ~no speedup from raising
#     it, and high values can DEADLOCK with JAX (os.fork vs JAX's threads). So the hint is a
#     benign false positive here. (Observed firing on an RTX 4090, scvi 1.4.3.)
DATALOADER_WORKERS_WARNING = r".*does not have many workers.*"


@contextlib.contextmanager
def suppressed_warnings(*message_patterns: str):
    """Silence specific warning MESSAGES within this block only (and nothing else).

    Wrap the single call known to emit a diagnosed-benign third-party warning, passing its
    exact message regex (see the module-level constants). Scope it as tightly as possible so
    unrelated — and genuinely informative — warnings still reach the user.
    """
    import warnings
    with warnings.catch_warnings():
        for pat in message_patterns:
            warnings.filterwarnings("ignore", message=pat)
        yield


def apply_scvi_training_best_practices() -> None:
    """scvi-tools / Lightning training best practices for our single-process, headless runs.

    Validated live on an RTX 4090 (scvi 1.4.3). Call once after `import scvi`, before
    `train()`. This sets only *global, side-effect-light* defaults; per-call args
    (`devices=1`, `check_val_every_n_epoch`, the num_workers warning scope) stay at the
    call site.

    - **Tensor-Core matmul precision.** On Tensor-Core GPUs (RTX 40xx / A100 / …) Lightning
      warns that float32 matmuls run at full precision and leave throughput on the table;
      `set_float32_matmul_precision('high')` trades negligible precision for a real speedup.
      No-op on CPU/MPS.
    - **DataLoader workers.** We intentionally leave `scvi.settings.dl_num_workers` at its
      default of 0 (see `DATALOADER_WORKERS_WARNING`): raising it gives ~no speedup and can
      deadlock with JAX. We don't change it here — this docstring is the rationale.
    """
    try:
        import torch
        torch.set_float32_matmul_precision("high")
    except Exception:  # noqa: BLE001 — torch always present in the gpu env; stay defensive
        pass


def resolve_accelerator(choice: str) -> str:
    """Resolve a scvi-tools accelerator string.

    'auto' -> cuda if available, else mps on Apple-Silicon, else cpu; any other value is
    returned as-is. Shared by the scvi-tools commands (reference train-model / map-scarches).
    torch is imported lazily so importing `cta.io` stays light.
    """
    if choice != "auto":
        return choice
    import platform
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


def stratified_subsample_positions(n, max_n, groups=None, *, seed=0):
    """Sorted positions (into `range(n)`) for a deterministic subsample of <= `max_n`.

    Shared by the reference subsamplers (CellMapper reference cap; Census slice cap). If
    `groups` (a length-`n` array of labels) is given, the budget is allocated across groups
    proportionally with a **floor of 1** so rare groups survive, then any rounding overshoot
    is trimmed back to `max_n`; otherwise it's a plain uniform draw. Fixed `seed` → identical
    selection across runs. Returns all positions when `n <= max_n`. Callers map the returned
    positions onto their own index domain (row positions, `soma_joinid`s, …).
    """
    rng = np.random.default_rng(seed)
    if n <= max_n:
        return np.arange(n)
    if groups is not None:
        groups = np.asarray(groups)
        cats, counts = np.unique(groups, return_counts=True)
        alloc = np.maximum(1, np.round(counts / counts.sum() * max_n).astype(int))
        picks = []
        for cat, n_take in zip(cats, alloc):
            idx = np.flatnonzero(groups == cat)
            picks.append(rng.choice(idx, size=min(int(n_take), idx.size), replace=False))
        sel = np.concatenate(picks)
        if sel.size > max_n:  # rounding can overshoot the cap — trim back
            sel = rng.choice(sel, size=max_n, replace=False)
    else:
        sel = rng.choice(n, size=max_n, replace=False)
    return np.sort(sel)


def load_adata(path: str, backed: str | None = None, prefer_slim: bool = True) -> ad.AnnData:
    """Load an AnnData.

    `backed="r"` reads without materializing X (use it when you only need obs/obsm).
    `prefer_slim` (default) transparently uses a fresh `<stem>_cta.h5ad` canonical sidecar
    written by canonicalize.py — so large objects aren't re-deserialized in full by every
    script, and every script sees the same standardized shape (log-norm X, counts layer,
    symbol var_names). The sidecar is used only if it is newer-or-equal to the original;
    set `CTA_NO_SLIM=1` to disable. Scripts that MUST see the full object or write it back
    (write_back.py, canonicalize.py) pass `prefer_slim=False`.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"ERROR: AnnData not found: {p}")
    if prefer_slim and not os.environ.get("CTA_NO_SLIM"):
        slim = p.with_name(p.stem + "_cta.h5ad")
        if slim.exists() and slim != p and slim.stat().st_mtime >= p.stat().st_mtime:
            eprint(f"   using slim cache {slim.name} (set CTA_NO_SLIM=1 to disable)")
            p = slim
    return ad.read_h5ad(p, backed=backed) if backed else ad.read_h5ad(p)


def safe_cluster_id(cid) -> str:
    """Sanitize a cluster id for safe use in a filename / HTML id.

    Cluster ids can contain '/' (e.g. a merged-then-subclustered id like '24/25_0'),
    which would break file paths. Replace path-hostile characters with '_'. The real
    id is always preserved verbatim in tables and card bodies — this is for filenames
    and cross-references only. Centralized here so every script sanitizes identically.
    """
    s = str(cid)
    for ch in "/\\ :":
        s = s.replace(ch, "_")
    return s


def split_csv(s: str | None) -> list[str]:
    """Parse a comma-separated CLI argument into a clean list of tokens.

    Centralized so every script splits identically (empty tokens dropped, whitespace
    trimmed). `split_csv("a, b ,,c") -> ['a', 'b', 'c']`; `split_csv(None) -> []`.
    """
    return [t.strip() for t in (s or "").split(",") if t.strip()]


# --- label ordering + colours (shared by build_report.py and write_back.py) ----------
# Centralized here so the palette the user reviews in the HTML report is the SAME one
# written into adata.uns by write_back.py — the colour<->cell-type relationship is
# preserved end to end.

UNKNOWN_GREY = (0.7, 0.7, 0.7)


def is_unknown_label(label) -> bool:
    """Sentinel 'no real identity' labels that always render neutral grey, regardless
    of palette — Unknown, Unassigned, NA, ambiguous, the empty group."""
    s = str(label).strip().lower()
    return s.startswith("unknown") or s in {
        "", "—", "na", "nan", "none", "unassigned", "ambiguous", "doublet", "low quality"}


def similarity_order(centroids) -> list[int]:
    """Order labels so similar ones are adjacent (so they get adjacent colours/numbers).

    Hierarchical-clustering leaf order over per-label centroids; identity order on
    failure or when centroids are unavailable. Fully data-driven — no dataset-specific
    assumptions.
    """
    if centroids is None:
        return []
    n = len(centroids)
    if n <= 2:
        return list(range(n))
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist
        Z = linkage(pdist(np.asarray(centroids)), method="ward", optimal_ordering=True)
        return list(leaves_list(Z))
    except Exception:  # noqa: BLE001
        return list(range(n))


def ordered_colors(n: int) -> list:
    """n gentle, perceptually-balanced RGB colours, applied in similarity order so
    similar cell types get neighbouring colours.
      - n <= 28: scanpy's muted `zeileis_28` (no neon clash).
      - n  > 28: HSLuv (husl) hue ramp + lightness zigzag — scales to any n.
    Never uses scanpy's harsh >28 default (godsnot_102)."""
    if n <= 0:
        return []
    from matplotlib.colors import to_rgb
    if n <= 28:
        try:
            from scanpy.plotting.palettes import zeileis_28
            return [to_rgb(zeileis_28[i]) for i in range(n)]
        except Exception:  # noqa: BLE001
            from matplotlib import colormaps
            base = list(colormaps["tab20"].colors)
            return [base[i % len(base)] for i in range(n)]
    import colorsys
    try:
        import seaborn as sns
        base = list(sns.husl_palette(n, s=0.62, l=0.65))
    except Exception:  # noqa: BLE001
        base = [colorsys.hsv_to_rgb(i / n, 0.5, 0.9) for i in range(n)]
    out = []
    for i, c in enumerate(base):
        h, _, s = colorsys.rgb_to_hls(*c)
        out.append(colorsys.hls_to_rgb(h, 0.55 if i % 2 else 0.74, s))
    return out


def label_color_map(cats: list, centroids):
    """Map each category to an RGB colour, similarity-ordered.

    `cats`: category labels (list). `centroids`: per-category vectors aligned to `cats`
    (e.g. median embedding coordinate), or None for identity order. Returns
    `(order, color_of)` where `order` is the similarity-ordered list of category indices
    and `color_of[i]` is the RGB tuple for `cats[i]`. Unknown-type labels always map to
    grey and are excluded from the palette (so no hue is wasted on them)."""
    order = similarity_order(centroids) or list(range(len(cats)))
    known = [ci for ci in order if not is_unknown_label(cats[ci])]
    palette = ordered_colors(len(known))
    color_of, ki = {}, 0
    for ci in order:
        if is_unknown_label(cats[ci]):
            color_of[ci] = UNKNOWN_GREY
        else:
            color_of[ci] = palette[ki]
            ki += 1
    return order, color_of


def check_hierarchy_nesting(level_maps: list) -> list[str]:
    """Verify a label hierarchy is strictly nested: every finer-level label maps to
    exactly ONE coarser-level label. A user-requested hierarchy must nest by design —
    e.g. if L1 splits NPC vs Neuron, an L2 label cannot span both.

    `level_maps`: list of (level_name, {cluster_id: label}) in order broad -> fine.
    Returns a list of human-readable violation strings (empty if consistent).
    """
    violations = []
    for (pl, pmap), (cl, cmap) in zip(level_maps, level_maps[1:]):
        by_child: dict[str, set] = {}
        for clu, child in cmap.items():
            parent = pmap.get(str(clu), pmap.get(clu))
            if parent is None:
                continue
            by_child.setdefault(str(child), set()).add(str(parent))
        for child, parents in by_child.items():
            if len(parents) > 1:
                violations.append(
                    f"{cl} label '{child}' has multiple {pl} parents: {sorted(parents)} "
                    f"(a finer label must refine exactly one coarser label)")
    return violations


def resolve_workdir(workdir: str) -> Path:
    """Working directory lives in the user's project, never in the plugin repo."""
    wd = Path(workdir).expanduser().resolve()
    for sub in ("tables", "evidence", "plots", "reports", "labels"):
        (wd / sub).mkdir(parents=True, exist_ok=True)
    return wd


def save_table(df: pd.DataFrame, workdir: Path, name: str, index: bool = False) -> Path:
    out = resolve_workdir(str(workdir)) / "tables" / name
    df.to_csv(out, index=index)
    eprint(f"   wrote {out}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    return out


def require_obs(adata: ad.AnnData, key: str, what: str = "column") -> None:
    if key not in adata.obs:
        raise SystemExit(
            f"ERROR: {what} '{key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )


def resolve_expression(adata: ad.AnnData, layer: str | None):
    """Return the matrix to use for log-normalized analyses.

    `layer=None` uses adata.X. Marker/scoring jobs (Job A) must run on
    log-normalized data — verify with the user which layer that is. We sanity-check
    that the values look log-scaled (max is modest, non-negative) and WARN otherwise
    rather than guessing.
    """
    if layer is not None and layer not in (adata.layers or {}):
        raise SystemExit(
            f"ERROR: layer '{layer}' not found. Available layers: "
            f"{list(adata.layers.keys())}"
        )
    X = adata.layers[layer] if layer is not None else adata.X
    src = f"layers['{layer}']" if layer is not None else "X"
    xmax = float(X.max())
    xmin = float(X.min())
    if xmin < 0:
        eprint(f"   WARN: {src} has negative values (max={xmax:.2f}); looks scaled, "
               "not log-normalized. Marker stats expect log1p data.")
    elif xmax > 50:
        eprint(f"   WARN: {src} max={xmax:.1f} looks like raw counts, not log1p. "
               "Point at the log-normalized layer.")
    else:
        eprint(f"   using {src} for expression (min={xmin:.2f}, max={xmax:.2f})")
    return X


def adata_with_lognorm_X(adata: ad.AnnData, layer: str | None) -> ad.AnnData:
    """Return an AnnData whose `.X` is the log-normalized matrix for marker/scoring jobs.

    Wraps `resolve_expression`: if `layer` is given, returns a COPY with `.X` set to that
    layer (so the caller's downstream scanpy calls read the right matrix without mutating
    the original); if `layer` is None, `.X` is already the target, so the object is returned
    unchanged. Centralizes the copy-only-when-needed pattern shared by the marker tools.
    """
    X = resolve_expression(adata, layer)
    if layer is None:
        return adata
    out = adata.copy()
    out.X = X
    return out


# The canonical layer holding log1p of CP10k (1e4 counts/cell) — what scale-sensitive
# classifiers (CellTypist) expect. `cta data canonicalize` precomputes it so downstream
# steps read it directly instead of re-normalizing ad hoc; X stays the user's own log-norm.
LOGNORM_CP10K_LAYER = "lognorm_cp10k"


def cp10k_log1p_matrix(*, counts=None, lognorm=None):
    """Build a ``log1p(CP10k)`` matrix (1e4 counts/cell) — the input scale-sensitive
    classifiers like CellTypist expect, regardless of how `adata.X` is normalized.

    Pass exactly one source (counts wins if both given):
      - ``counts``: raw counts → ``normalize_total(1e4)`` → ``log1p`` (the exact path).
      - ``lognorm``: an already size-normalized-log1p matrix on some other target →
        ``expm1`` → ``normalize_total(1e4)`` → ``log1p`` (re-targets to 1e4; valid because
        the caller verified it is genuine size-normalized log1p).

    Operates on a copy; returns ``(matrix, note)`` where `note` describes the source. Shared
    by `cta data canonicalize` (writes it to ``layers['lognorm_cp10k']``) and the CellTypist
    cross-check (fallback when that layer is absent).
    """
    import anndata as ad_
    import scanpy as sc
    import scipy.sparse as sp

    if counts is not None:
        base = counts.copy()
        note = "from raw counts"
    elif lognorm is not None:
        base = lognorm.copy()
        if sp.issparse(base):
            base.data = np.expm1(base.data)
        else:
            base = np.expm1(np.asarray(base))
        note = "re-targeted from the log-norm matrix"
    else:
        raise ValueError("cp10k_log1p_matrix: pass counts= or lognorm=")
    tmp = ad_.AnnData(base)  # fresh uns → no stale log1p marker
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    return tmp.X, note


def panel_gate(markers: dict[str, list[str]], var_names) -> tuple[dict, dict]:
    """Intersect each marker set with the genes present (the panel).

    Returns (kept, dropped). NEVER treat a dropped (off-panel) gene as negative
    evidence — it is unmeasured, not absent.
    """
    present = set(map(str, var_names))
    kept, dropped = {}, {}
    for ct, genes in markers.items():
        g = [str(x) for x in genes]
        k = [x for x in g if x in present]
        d = [x for x in g if x not in present]
        kept[ct] = k
        if d:
            dropped[ct] = d
    # A total miss when var_names look like Ensembl is the classic ID-convention mismatch
    # (markers are symbols): point at canonicalization rather than failing opaquely later.
    if not any(kept.values()) and _looks_like_ensembl(var_names):
        eprint("   WARN: 0 markers matched and var_names look like Ensembl IDs — run "
               "`cta data canonicalize` so var_names become gene symbols, then retry.")
    return kept, dropped


def _looks_like_ensembl(var_names) -> bool:
    """True if most var_names match the Ensembl gene-ID pattern (ENSG…, ENSMUSG…)."""
    import re
    names = [str(n) for n in var_names]
    if not names:
        return False
    pat = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")
    return sum(1 for n in names if pat.match(n)) / len(names) > 0.5


def read_markers_json(path: str) -> dict[str, list[str]]:
    """Marker sets as {cell_type: [genes]} from a JSON file."""
    data = json.loads(Path(path).expanduser().read_text())
    if not isinstance(data, dict):
        raise SystemExit("ERROR: markers JSON must be an object {type: [genes]}")
    return {str(k): list(map(str, v)) for k, v in data.items()}


def aggregate_predictions(clusters, labels, conf, method: str) -> pd.DataFrame:
    """Per-cluster summary of transferred per-cell labels — the shared `refmap_*` schema.

    Used by every label-transfer cross-check (reference_map / map_query_scarches /
    run_scimilarity) so they emit an identical table:
    ``cluster, top1, frac1, top2, frac2, mean_conf, method``. `mean_conf` is NaN-safe
    (blank when no finite confidences exist, e.g. a method with no per-cell score).
    """
    df = pd.DataFrame({"cluster": clusters, "label": labels, "conf": conf})
    rows = []
    for cl, g in df.groupby("cluster", sort=True):
        vc = g["label"].value_counts()
        n = vc.sum()
        top1, frac1 = vc.index[0], round(float(vc.iloc[0] / n), 4)
        top2, frac2 = (vc.index[1], round(float(vc.iloc[1] / n), 4)) if len(vc) > 1 else ("", 0.0)
        c = np.asarray(g["conf"], dtype=float)
        mean_conf = round(float(np.nanmean(c)), 4) if np.isfinite(c).any() else ""
        rows.append({"cluster": cl, "top1": top1, "frac1": frac1, "top2": top2,
                     "frac2": frac2, "mean_conf": mean_conf, "method": method})
    return pd.DataFrame(rows)
