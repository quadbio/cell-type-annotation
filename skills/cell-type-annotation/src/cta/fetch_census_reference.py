"""Fetch a small, capped reference AnnData slice from CZ CELLxGENE Census.

The output is a valid ``--reference`` for reference_map.py: raw counts in X plus a
cell-type label obs column. reference_map.py subsets to shared genes and applies its
own normalize_total(1e4)+log1p, so leaving X as Census raw counts is exactly right.

Laptop-scale / remote model:
  Opening the Census does NOT download the atlas. The Census is a remote TileDB-SOMA
  store on S3; ``open_soma`` just attaches to it. We FIRST read only the obs metadata
  for the matching slice (to collect soma_joinids), pick a capped, stratified, fixed-
  seed subset of those ids, and only THEN fetch the expression matrix for that handful
  of cells. Nothing unbounded is ever pulled. This requires network access and fails
  cleanly when offline (the TileDB-SOMA open raises).

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

from cta.io import eprint, resolve_workdir, save_table

# soma_joinid is the obs primary key; obs_coords in get_anndata indexes by it.
JOINID = "soma_joinid"


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

    Fixed-seed (np.random.default_rng(0)) so repeated runs return the same slice. When
    the matching set already fits under the cap, take all of it. Stratifying by the
    cell-type column allocates the budget proportionally with a floor of 1 per group, so
    rare types survive the subsample.
    """
    n = len(obs)
    if n <= max_cells:
        return np.sort(obs[JOINID].to_numpy())

    rng = np.random.default_rng(0)
    if stratify_by and stratify_by in obs.columns:
        groups = obs[stratify_by].astype(str).to_numpy()
        cats, counts = np.unique(groups, return_counts=True)
        alloc = np.maximum(1, np.round(counts / counts.sum() * max_cells).astype(int))
        picks = []
        for cat, n_take in zip(cats, alloc):
            ids = obs.loc[groups == cat, JOINID].to_numpy()
            picks.append(rng.choice(ids, size=min(int(n_take), ids.size), replace=False))
        sel = np.concatenate(picks)
        if sel.size > max_cells:  # rounding can overshoot — trim back to the cap
            sel = rng.choice(sel, size=max_cells, replace=False)
        eprint(f"   stratified subsample by '{stratify_by}': {n} -> {sel.size} cells")
    else:
        if stratify_by:
            eprint(f"   WARN: stratify col '{stratify_by}' absent from obs; random subsample")
        all_ids = obs[JOINID].to_numpy()
        sel = rng.choice(all_ids, size=max_cells, replace=False)
        eprint(f"   random subsample: {n} -> {sel.size} cells")
    return np.sort(sel)


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
        eprint(f">> reading matching obs metadata (organism='{organism}')")
        eprint(f"   obs_value_filter: {obs_filter}")
        # Read ONLY the obs columns we need to choose joinids — no expression yet.
        obs_cols = list(dict.fromkeys([JOINID, cell_type_col, stratify_by]))
        obs = cellxgene_census.get_obs(
            census,
            organism=organism,
            value_filter=obs_filter,
            column_names=obs_cols,
        )
        if obs is None or len(obs) == 0:
            raise SystemExit(
                "ERROR: no cells matched the filter. Check the --tissue spelling against "
                "Census `tissue_general` values, or pass --obs-value-filter with an "
                "ontology term, e.g. \"tissue_ontology_term_id=='UBERON:0000955'\"."
            )
        eprint(f"   {len(obs)} cells match before capping")

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

    # --- summary ---
    eprint(">> done")
    eprint(f"   reference : n_obs={adata_obj.n_obs}  n_vars={adata_obj.n_vars}  (X = raw counts)")
    eprint(f"   organism  : {organism}")
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
