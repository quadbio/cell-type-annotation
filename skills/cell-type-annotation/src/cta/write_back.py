#!/usr/bin/env python
"""Final step: write per-level cluster->label annotations back into adata.obs IN-PLACE.

Reads the per-level label tables produced during annotation
(`<workdir>/labels/annotation_<level>.csv`) and maps each cluster to its label,
adding one obs column per level (`<col-prefix>_<level>`, e.g. cta_L1). Optionally
mirrors the `state` (cell-state axis), CL ontology id, and confidence columns into
companion obs columns (`<col-prefix>_<level>_state`, …) when present in the CSV.

Each label column is written as an ORDERED pandas Categorical, with categories in the
same similarity order the HTML reports use, and the matching palette is written to
`adata.uns['<col>_colors']`. So when the user later does `sc.pl.umap(adata, color=...)`,
the colour<->cell-type relationship matches the reports they signed off on — scanpy
does not invent fresh arbitrary colours. Ordering/colours need the anchor embedding
(`--embedding-key`, falling back to `--viz-key`); without one, categories fall back to
a stable order (Unknown/Unassigned last) and colours are still written.

Before writing, it enforces two gates (in both --dry-run and real runs): the hierarchy must
be strictly nested (check_hierarchy_nesting), and every non-blank `cl_id` must be a
well-formed ontology accession (CL/PCL:0000000). Both block by default with an explicit
override (--allow-non-nested / --allow-bad-cl-id). The cl_id gate is SYNTAX only; whether a
term exists / is current / matches its label is checked in-chat via the knowledgebase MCP.

This MUTATES the user's AnnData file. The skill gates this behind an explicit
user confirmation in chat; the script additionally supports --dry-run so the
mapping can be previewed without touching the file. Pass --output to write a COPY
instead of mutating the original in place.

Usage:
  python write_back.py --adata path.h5ad --workdir <dir> --cluster-key leiden \
      --embedding-key X_scVI [--levels L1,L2,L3,L4] [--col-prefix cta] [--dry-run]
  # write a copy instead of mutating in place:
  python write_back.py --adata path.h5ad --workdir <dir> --cluster-key leiden \
      --embedding-key X_scVI --output path.annotated.h5ad

If --levels is omitted, levels are auto-discovered from the filenames matching
`<workdir>/labels/annotation_*.csv` (sorted).
"""

from __future__ import annotations

import os
import re
import tempfile

import pandas as pd
import typer

from cta.constants import CL_ID_PATTERN
from cta.io import (
    CLUSTER_KEY_OPT,
    WORKDIR_OPT,
    check_hierarchy_nesting,
    eprint,
    label_color_map,
    load_adata,
    require_obs,
    resolve_workdir,
    split_csv,
)

# cl_id cells that mean "no ontology id" — skipped by the syntax check (cl_id is optional).
_BLANK_CL_ID = {"", "nan", "na", "none"}

# CSV columns that, when present, get mirrored into companion obs columns. `state` is the
# orthogonal cell-STATE axis (normal / low-quality / cycling / doublet …) — written next to,
# not folded into, the cell-TYPE label, and exempt from the CL-ID + nesting machinery (only
# `label` feeds the hierarchy-nesting check).
EXTRA_COLS = ("state", "cl_id", "confidence")
UNASSIGNED = "Unassigned"


def discover_levels(workdir) -> list[str]:
    """Auto-discover levels from labels/annotation_<level>.csv filenames."""
    labels_dir = workdir / "labels"
    pat = re.compile(r"^annotation_(.+)\.csv$")
    levels = []
    for f in sorted(labels_dir.glob("annotation_*.csv")):
        m = pat.match(f.name)
        if m:
            levels.append(m.group(1))
    return levels


def load_level_table(workdir, level: str) -> pd.DataFrame:
    path = workdir / "labels" / f"annotation_{level}.csv"
    if not path.exists():
        raise SystemExit(f"ERROR: label table not found for level '{level}': {path}")
    df = pd.read_csv(path)
    for req in ("cluster", "label"):
        if req not in df.columns:
            raise SystemExit(f"ERROR: {path} must have a '{req}' column; got {list(df.columns)}")
    # Cluster ids are matched as strings against the cluster_key categories.
    # Strip whitespace so a stray trailing space in a hand-authored CSV doesn't
    # silently drop a cluster to Unassigned.
    df["cluster"] = df["cluster"].astype(str).str.strip()
    return df


def build_mapping(df: pd.DataFrame, value_col: str) -> dict[str, str]:
    """cluster(str) -> value, dropping rows with a missing value."""
    out: dict[str, str] = {}
    for cl, val in zip(df["cluster"], df[value_col], strict=True):
        if pd.isna(val):
            continue
        out[str(cl)] = str(val)
    return out


def validate_cl_ids(records: list[tuple[str, str, str, str]]) -> list[str]:
    """SYNTACTICALLY validate cl_id accessions before mutating the user's AnnData.

    `records`: (level, cluster, label, cl_id). Returns human-readable violation strings for any
    NON-BLANK cl_id that is not a well-formed ontology accession (CL/PCL:0000000, per
    `constants.CL_ID_PATTERN`). Blank cl_ids are fine — cl_id is optional, and Unknown /
    low-quality clusters routinely carry none (so state-only rows are exempt automatically).

    This is a cheap, offline SYNTAX check only. It deliberately does NOT confirm the term
    exists, is not obsolete, or matches its label — that SEMANTIC check needs an ontology
    lookup and is done in-chat via the knowledgebase MCP (`bc_get_term_details`), because `cta`
    commands have no MCP access. See SKILL.md Stage 3.6 and reference/knowledge_sources.md.
    """
    violations = []
    for level, cluster, label, cl_id in records:
        cid = str(cl_id).strip()
        if cid.lower() in _BLANK_CL_ID:
            continue
        if not CL_ID_PATTERN.match(cid):
            violations.append(
                f"{level} cluster '{cluster}' (label '{label}'): cl_id '{cid}' is not a "
                f"well-formed ontology accession (expected e.g. CL:0000679 or PCL:0000123)"
            )
    return violations


def order_label_column(series: pd.Series, adata, cluster_key: str, embedding_key: str | None):
    """Return (ordered categorical Series, hex colour list aligned to its categories).

    Categories are put in similarity order using per-label centroids in the anchor
    embedding (the SAME ordering the HTML reports use); Unknown/Unassigned labels render
    grey and sort to the end. Without a usable embedding, categories fall back to a
    stable order (known labels as first seen, Unknown last) and colours are still
    assigned. The hex list is aligned to the returned categories for adata.uns.
    """
    from matplotlib.colors import to_hex

    from cta.io import is_unknown_label

    cats = list(pd.Categorical(series).categories)
    centroids = None
    if embedding_key and embedding_key in (adata.obsm or {}):
        import numpy as np

        emb = np.asarray(adata.obsm[embedding_key])
        vals = series.to_numpy()
        rows = []
        for c in cats:
            mask = vals == c
            rows.append(np.median(emb[mask], axis=0) if mask.any() else np.zeros(emb.shape[1]))
        centroids = np.vstack(rows) if rows else None
    else:
        if embedding_key:
            eprint(
                f"   note: embedding '{embedding_key}' not in obsm — colours use a "
                f"stable (non-similarity) order"
            )

    order, color_of = label_color_map(cats, centroids)
    # Keep Unknown-type labels at the end regardless of where similarity placed them.
    order = [ci for ci in order if not is_unknown_label(cats[ci])] + [
        ci for ci in order if is_unknown_label(cats[ci])
    ]
    ordered_cats = [cats[ci] for ci in order]
    colors = [to_hex(color_of[ci]) for ci in order]
    out = series.astype("category").cat.set_categories(ordered_cats, ordered=True)
    return out, colors


def map_level(
    adata, cluster_key: str, df: pd.DataFrame, prefix: str, level: str, embedding_key: str | None
):
    """Return ({obs_col: mapped Series}, {col: colour list}, unmapped, orphans) for one level.

    `unmapped` = cluster ids in the data with no label row (-> Unassigned).
    `orphans`  = cluster ids in the CSV that match no cluster in the data (likely an id
    formatting mismatch — surfaced as a warning).
    """
    clusters = adata.obs[cluster_key].astype(str).str.strip()
    present = list(dict.fromkeys(clusters.tolist()))  # cluster ids actually in the data

    label_map = build_mapping(df, "label")
    unmapped = [c for c in present if c not in label_map]
    orphans = [c for c in label_map if c not in set(present)]

    cols: dict[str, pd.Series] = {}
    colors: dict[str, list] = {}
    main_col = f"{prefix}_{level}"
    mapped = clusters.map(label_map).fillna(UNASSIGNED)
    cols[main_col], colors[main_col] = order_label_column(mapped, adata, cluster_key, embedding_key)

    for extra in EXTRA_COLS:
        if extra in df.columns:
            emap = build_mapping(df, extra)
            if emap:
                col = f"{prefix}_{level}_{extra}"
                cols[col] = clusters.map(emap)  # leave NaN where absent (no Unassigned)
    return cols, colors, unmapped, orphans


def cli(
    adata: str = typer.Option(..., help="path to the .h5ad to annotate"),
    workdir: str = WORKDIR_OPT,
    cluster_key: str = CLUSTER_KEY_OPT,
    embedding_key: str | None = typer.Option(
        None,
        help="obsm key of the anchor embedding, used to order labels by "
        "similarity and write a matching uns palette (so saved colours "
        "match the reports). Falls back to --viz-key.",
    ),
    viz_key: str | None = typer.Option(None, help="fallback obsm key if --embedding-key absent"),
    levels: str | None = typer.Option(
        None, help="comma-separated, e.g. L1,L2,L3,L4 (default: auto-discover)"
    ),
    col_prefix: str = typer.Option("cta", help="prefix for the written obs columns"),
    output: str | None = typer.Option(
        None, help="write the annotated object to this path instead of mutating --adata in place"
    ),
    dry_run: bool = typer.Option(False, help="preview the mapping without touching the file"),
    allow_non_nested: bool = typer.Option(
        False,
        help="override the nested-hierarchy check (NOT recommended — a requested "
        "hierarchy should nest; a finer label spanning two parents is a bug)",
    ),
    allow_bad_cl_id: bool = typer.Option(
        False,
        help="override the cl_id syntax check (NOT recommended — a malformed ontology "
        "accession should be fixed, not written into obs)",
    ),
) -> None:
    """Write per-level cluster->label annotations back into adata.obs in-place."""
    workdir = resolve_workdir(workdir)

    levels = split_csv(levels) or discover_levels(workdir)
    if not levels:
        raise SystemExit(
            f"ERROR: no levels to write. Provide --levels or place "
            f"annotation_<level>.csv files in {workdir / 'labels'}."
        )
    eprint(f">> levels: {', '.join(levels)}")

    adata_obj = load_adata(adata, prefer_slim=False)  # must write back the FULL original
    require_obs(adata_obj, cluster_key, "cluster key")

    embedding_key = embedding_key or viz_key
    if not embedding_key:
        eprint(
            "   note: no --embedding-key/--viz-key given — label order/colours will use "
            "a stable (non-similarity) order; pass the anchor embedding to match reports"
        )

    all_cols: dict[str, pd.Series] = {}
    all_colors: dict[str, list] = {}
    level_maps: list = []
    cl_id_records: list[tuple[str, str, str, str]] = []
    for level in levels:
        df = load_level_table(workdir, level)
        level_maps.append(
            (level, dict(zip(df["cluster"].astype(str), df["label"].astype(str), strict=True)))
        )
        if "cl_id" in df.columns:
            for _, r in df.iterrows():
                cl_id_records.append(
                    (level, str(r["cluster"]), str(r.get("label", "")), r.get("cl_id", ""))
                )
        cols, colors, unmapped, orphans = map_level(
            adata_obj, cluster_key, df, col_prefix, level, embedding_key
        )
        main_col = f"{col_prefix}_{level}"

        eprint(f"\n>> {level} -> obs['{main_col}']")
        vc = cols[main_col].value_counts(dropna=False)
        for label, n in vc.items():
            eprint(f"   {label}: {n}")
        for extra_col in cols:
            if extra_col != main_col:
                eprint(f"   (+ companion column '{extra_col}')")
        if main_col in adata_obj.obs:
            eprint(f"   WARN: obs['{main_col}'] already exists — it will be overwritten")
        if unmapped:
            eprint(
                f"   WARN: {len(unmapped)} cluster(s) had no label row -> '{UNASSIGNED}': "
                f"{unmapped}"
            )
        if orphans:
            eprint(
                f"   WARN: {len(orphans)} label row(s) match no cluster in obs"
                f"['{cluster_key}'] (id formatting mismatch?): {orphans}"
            )
        all_cols.update(cols)
        all_colors.update(colors)

    # Enforce a nested hierarchy: every finer label must refine exactly one coarser label.
    violations = check_hierarchy_nesting(level_maps)
    if violations:
        msg = (
            "hierarchy is NOT nested — a level must strictly refine the level above:\n   "
            + "\n   ".join(violations)
            + "\n   Fix the labels so each finer label has one parent (e.g. qualify by "
            "compartment: 'Ventral telencephalic NPC' vs '… neuron')."
        )
        if allow_non_nested:
            eprint(f"\n>> WARN: {msg}\n>> --allow-non-nested set: writing anyway.")
        else:
            raise SystemExit(f"\nERROR: {msg}\n   (override with --allow-non-nested)")
    else:
        eprint(f"\n>> hierarchy nesting OK: {' ⊂ '.join(lvl for lvl, _ in reversed(level_maps))}")

    # Syntactically validate any cl_id before it lands in the user's AnnData (semantic
    # existence/obsolescence is checked in-chat via the knowledgebase MCP — see SKILL.md 3.6).
    cl_id_violations = validate_cl_ids(cl_id_records)
    if cl_id_violations:
        msg = (
            "cl_id column has malformed ontology accession(s):\n   "
            + "\n   ".join(cl_id_violations)
            + "\n   Fix each (or clear the cell) — a cl_id must look like CL:0000679. "
            "Existence/obsolescence and label-match are verified separately in chat via the "
            "knowledgebase MCP."
        )
        if allow_bad_cl_id:
            eprint(f"\n>> WARN: {msg}\n>> --allow-bad-cl-id set: writing anyway.")
        else:
            raise SystemExit(f"\nERROR: {msg}\n   (override with --allow-bad-cl-id)")
    else:
        n_checked = sum(
            1 for _, _, _, c in cl_id_records if str(c).strip().lower() not in _BLANK_CL_ID
        )
        if n_checked:
            eprint(f">> cl_id syntax OK: {n_checked} accession(s) well-formed")

    if dry_run:
        eprint("\n>> DRY RUN: no file written. Columns that WOULD be added:")
        for c in all_cols:
            extra = f"  (+ uns['{c}_colors'])" if c in all_colors else ""
            eprint(f"   - obs['{c}']{extra}")
        return

    for c, series in all_cols.items():
        adata_obj.obs[c] = series
        if c in all_colors and all_colors[c]:
            # scanpy contract: uns['<col>_colors'][i] <-> obs['<col>'].cat.categories[i]
            adata_obj.uns[f"{c}_colors"] = list(all_colors[c])

    # Write to a temp file in the same dir, then atomically replace, so a failure
    # mid-write never corrupts the destination AnnData. With --output, the original
    # --adata is left untouched.
    dest = os.path.abspath(os.path.expanduser(output or adata))
    dest_dir = os.path.dirname(dest) or "."
    fd, tmp = tempfile.mkstemp(suffix=".h5ad", dir=dest_dir)
    os.close(fd)
    try:
        adata_obj.write_h5ad(tmp)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    where = dest if output else f"{dest} (IN-PLACE)"
    eprint(f"\n>> wrote {len(all_cols)} obs column(s) + {len(all_colors)} palette(s) to {where}:")
    for c in all_cols:
        extra = f"  (+ uns['{c}_colors'])" if c in all_colors else ""
        eprint(f"   - obs['{c}']{extra}")
