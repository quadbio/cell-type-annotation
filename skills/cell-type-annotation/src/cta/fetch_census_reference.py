"""Fetch a small, capped reference AnnData slice from CZ CELLxGENE Census.

The output is a valid ``--reference`` for reference_map.py: raw counts in X plus a
cell-type label obs column. reference_map.py subsets to shared genes and applies its
own normalize_total(1e4)+log1p, so leaving X as Census raw counts is exactly right.

Laptop-scale / remote model:
  Opening the Census does NOT download the atlas. The Census is a remote TileDB-SOMA
  store on S3; ``open_soma`` just attaches to it. We read the matching obs in two
  phases so a broad filter (e.g. tissue_general=='blood', millions of cells) never
  pulls the heavy label column for the full match set:

    1. a CHEAP probe reading only ``soma_joinid`` (one int column) to count matches;
    2. the label/stratify column read ONLY for a bounded random candidate pool of
       those ids, capped at POOL_CAP, then a fixed-seed stratified subsample of the
       pool down to --max-cells.

  Only then do we fetch the expression matrix for that handful of cells. Nothing
  unbounded is ever pulled. This requires network access and fails cleanly when
  offline (the TileDB-SOMA open raises).

Reproducibility:
  ``--census-version`` defaults to 'stable' (the friendly LTS alias), which advances every
  ~6 months. We resolve it to the dated build (e.g. '2025-11-08') and stamp it into
  ``adata.uns['census_provenance']`` alongside the organism/filter/counts, so the slice is
  reproducible. Pass that date back as ``--census-version`` to refetch the exact release.

Caveat on the labels:
  Census ``cell_type`` is CL-backed but AUTHOR-SUPPLIED and heterogeneous across the
  many datasets that make up a tissue slice — granularity and naming differ between
  contributors. A label transferred from a Census slice is therefore a coarser/noisier
  hypothesis than one from a single curated, internally-consistent atlas. Treat the
  reference_map.py cross-check accordingly: corroborating signal, not ground truth.

Usage:
  # 1) tissue_general free-text filter (simplest)
  python fetch_census_reference.py --organism homo_sapiens --tissue brain \
      --out brain_ref.h5ad [--max-cells 50000] [--workdir /proj/run]

  # 2) advanced: raw Census obs_value_filter with an ontology term (overrides --tissue)
  python fetch_census_reference.py --organism homo_sapiens \
      --obs-value-filter "tissue_ontology_term_id=='UBERON:0000955' and is_primary_data==True" \
      --out brain_ref.h5ad

Then feed the result to reference_map.py:
  python reference_map.py --query Q.h5ad --reference brain_ref.h5ad \
      --cluster-key leiden --label-cols cell_type --workdir /proj/run --ref-counts
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import typer

from cta.io import eprint, resolve_workdir, save_table, stratified_subsample_positions

# soma_joinid is the obs primary key; obs_coords in get_anndata indexes by it.
JOINID = "soma_joinid"

# Fixed seed for the deterministic pool downsample (matches the io subsampler default).
SEED = 0
# Candidate-pool cap for the heavy label read: max(max_cells * FACTOR, MIN). Reading the
# string label column for a broad filter scales with the match set; we instead read it for
# at most this many cells (a uniform random pre-sample, which preserves group proportions
# for the stratified draw). Above this, the read is capped and stratification is approximate.
POOL_FACTOR = 10
POOL_MIN = 200_000


def _import_census():
    """Guarded import: cellxgene_census is an optional extra, not a core dependency.

    Absence must be a clean, actionable SystemExit (never a traceback) — it ships in
    the skill's pixi env (`pixi install`).
    """
    try:
        import cellxgene_census  # noqa: F401
    except ImportError:
        raise SystemExit(
            "ERROR: cellxgene_census is not installed. It ships in the skill's pixi env — run "
            "`pixi install` (from the skill dir) to build it. It pulls a remote "
            "TileDB-SOMA store over S3, so a working network connection is also required."
        )
    return cellxgene_census


def _resolve_census_build(cellxgene_census, census_version: str) -> str:
    """Resolve a moving alias ('stable'/'latest') to the dated LTS build (YYYY-MM-DD).

    Census best practice is to pin `census_version` to a dated release for reproducibility
    (the alias 'stable' advances every ~6 months). We keep 'stable' as the friendly default
    but resolve and RECORD the concrete build so the written slice is reproducible. Best-effort:
    fall back to the raw string if the version directory can't be read (e.g. transient network).
    """
    try:
        entry = cellxgene_census.get_census_version_directory().get(census_version)
        if entry and entry.get("release_build"):
            return entry["release_build"]
    except Exception:  # noqa: BLE001 — provenance must never break the fetch
        pass
    return census_version


def _normalize_organism(organism: str) -> str:
    """Census `organism` accepts the friendly name (e.g. 'Homo sapiens').

    Accept the snake_case form used by the experiment name too ('homo_sapiens') and
    normalize it to the spaced, capitalized form the API expects.
    """
    o = organism.strip()
    if "_" in o:
        o = o.replace("_", " ")
    return o[:1].upper() + o[1:] if o else o


def _build_obs_value_filter(obs_value_filter: Optional[str], tissue: Optional[str]) -> str:
    """Construct the Census obs_value_filter (SOMA value_filter syntax).

    If --obs-value-filter is given, use it VERBATIM (the advanced escape hatch — e.g. a
    tissue_ontology_term_id constraint). Otherwise build one from --tissue against the
    `tissue_general` column. In both cases we AND-in `is_primary_data == True` to drop
    duplicated cells, UNLESS the raw filter already mentions is_primary_data.
    """
    if obs_value_filter:
        raw = obs_value_filter.strip()
        if "is_primary_data" in raw:
            return raw
        return f"({raw}) and is_primary_data == True"
    if not tissue:
        raise SystemExit(
            "ERROR: provide either --tissue (a tissue_general value, e.g. 'brain') or "
            "--obs-value-filter (a raw Census obs_value_filter string)."
        )
    tissue = tissue.replace("'", "\\'")
    return f"tissue_general == '{tissue}' and is_primary_data == True"


def _choose_joinids(obs: pd.DataFrame, max_cells: int, stratify_by: str) -> np.ndarray:
    """Deterministically pick <= max_cells soma_joinids, stratified by `stratify_by`.

    Fixed-seed so repeated runs return the same slice; takes all when the matching set
    already fits. Delegates the proportional, rare-group-preserving allocation to
    cta.io.stratified_subsample_positions and maps the chosen positions onto soma_joinids.
    """
    n = len(obs)
    ids = obs[JOINID].to_numpy()
    groups = None
    if stratify_by and stratify_by in obs.columns:
        groups = obs[stratify_by].astype(str).to_numpy()
    elif stratify_by:
        eprint(f"   WARN: stratify col '{stratify_by}' absent from obs; random subsample")
    sel = np.sort(ids[stratified_subsample_positions(n, max_cells, groups)])
    if n > max_cells:
        kind = (f"stratified subsample by '{stratify_by}'" if groups is not None
                else "random subsample")
        eprint(f"   {kind}: {n} -> {sel.size} cells")
    return sel


def cli(
    organism: str = typer.Option(
        ..., help="e.g. 'homo_sapiens'/'Homo sapiens' or 'mus_musculus'/'Mus musculus'"),
    out: str = typer.Option(..., help="output .h5ad path"),
    tissue: Optional[str] = typer.Option(
        None, help="tissue_general value to filter on, e.g. 'brain'. For an "
                   "ontology term instead, use --obs-value-filter."),
    obs_value_filter: Optional[str] = typer.Option(
        None, help="advanced: raw Census obs_value_filter string that OVERRIDES the "
                   "tissue filter, e.g. "
                   "\"tissue_ontology_term_id=='UBERON:0000955' and is_primary_data==True\""),
    cell_type_col: str = typer.Option(
        "cell_type", help="Census obs column to carry as the reference label (default cell_type)"),
    max_cells: int = typer.Option(
        50000, help="HARD cap on fetched cells; the slice is subsampled down to this"),
    stratify_by: Optional[str] = typer.Option(
        None, help="obs col to stratify the subsample by; default is --cell-type-col"),
    census_version: str = typer.Option(
        "stable", help="Census version tag (default 'stable')"),
    workdir: Optional[str] = typer.Option(
        None, help="if given, also write tables/census_reference_summary.csv"),
) -> None:
    """Fetch a small, capped reference AnnData slice from CZ CELLxGENE Census."""
    cellxgene_census = _import_census()

    organism = _normalize_organism(organism)
    obs_filter = _build_obs_value_filter(obs_value_filter, tissue)
    stratify_by = stratify_by or cell_type_col

    eprint(f">> opening CELLxGENE Census (version='{census_version}')")
    eprint("   (remote TileDB-SOMA over S3; the atlas is NOT downloaded)")
    with cellxgene_census.open_soma(census_version=census_version) as census:
        census_build = _resolve_census_build(cellxgene_census, census_version)
        if census_build != census_version:
            eprint(f"   resolved '{census_version}' -> build {census_build} "
                   f"(pin --census-version {census_build} to reproduce this slice)")
        eprint(f">> probing matching obs (organism='{organism}')")
        eprint(f"   obs_value_filter: {obs_filter}")
        obs_cols = list(dict.fromkeys([JOINID, cell_type_col, stratify_by]))
        # Phase 1: CHEAP probe — read only soma_joinid (one int column) to count matches.
        # A broad filter can match millions of cells; we must not pull the label column
        # for all of them (that read is what hangs on a login node).
        ids = cellxgene_census.get_obs(
            census,
            organism=organism,
            value_filter=obs_filter,
            column_names=[JOINID],
        )[JOINID].to_numpy()
        n_match = ids.size
        if n_match == 0:
            raise SystemExit(
                "ERROR: no cells matched the filter. Check the --tissue spelling against "
                "Census `tissue_general` values, or pass --obs-value-filter with an "
                "ontology term, e.g. \"tissue_ontology_term_id=='UBERON:0000955'\"."
            )
        eprint(f"   {n_match} cells match before capping")

        if n_match <= max_cells:
            # Whole match set fits under the cap: take all; no label read needed here
            # (get_anndata reads the labels for these few cells below).
            joinids = np.sort(ids)
        else:
            # Phase 2: read the label/stratify column ONLY for a bounded candidate pool,
            # so the heavy read never scales with the full match set.
            pool_cap = max(max_cells * POOL_FACTOR, POOL_MIN)
            if n_match > pool_cap:
                eprint(
                    f"   WARN: large match set ({n_match} cells); capping the obs read to a "
                    f"random pool of {pool_cap}. Stratification is approximate over this pool "
                    "— narrow --tissue/--obs-value-filter for a more representative slice."
                )
                pool_ids = np.sort(np.random.default_rng(SEED).choice(ids, size=pool_cap, replace=False))
            else:
                pool_ids = np.sort(ids)
            obs = cellxgene_census.get_obs(
                census,
                organism=organism,
                coords=pool_ids.tolist(),  # restrict the label read to the pool
                column_names=obs_cols,
            )
            joinids = _choose_joinids(obs, max_cells, stratify_by)

        eprint(f">> fetching AnnData for {joinids.size} selected cells")
        adata_obj = cellxgene_census.get_anndata(
            census,
            organism=organism,
            obs_value_filter=obs_filter,
            obs_coords=joinids.tolist(),  # restrict to our capped set of joinids
            column_names={"obs": obs_cols},
        )

    if adata_obj.n_obs == 0:
        raise SystemExit("ERROR: fetched AnnData is empty after applying obs_coords.")

    # Stamp provenance so the slice is self-describing and reproducible (Census best
    # practice: record the dated build, not just the moving 'stable' alias).
    adata_obj.uns["census_provenance"] = {
        "census_version": census_build,
        "census_version_requested": census_version,
        "organism": organism,
        "obs_value_filter": obs_filter,
        "n_matched": int(n_match),
        "n_fetched": int(adata_obj.n_obs),
    }

    # --- summary ---
    eprint(">> done")
    eprint(f"   reference : n_obs={adata_obj.n_obs}  n_vars={adata_obj.n_vars}  (X = raw counts)")
    eprint(f"   organism  : {organism}")
    eprint(f"   census    : build {census_build}")
    eprint(f"   filter    : {obs_filter}")
    counts = adata_obj.obs[cell_type_col].astype(str).value_counts()
    eprint(f"   '{cell_type_col}' value_counts ({counts.size} types):")
    for label, c in counts.items():
        eprint(f"      {c:>7d}  {label}")

    if workdir:
        wd = resolve_workdir(workdir)
        summary = counts.rename_axis(cell_type_col).reset_index(name="n_cells")
        save_table(summary, wd, "census_reference_summary.csv")

    eprint(f">> writing {out}")
    adata_obj.write_h5ad(out)
    print(out)
