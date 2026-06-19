#!/usr/bin/env python
"""Final step: write per-level cluster->label annotations back into adata.obs IN-PLACE.

Reads the per-level label tables produced during annotation
(`<workdir>/labels/annotation_<level>.csv`) and maps each cluster to its label,
adding one obs column per level (`<col-prefix>_<level>`, e.g. cta_L1). Optionally
mirrors the CL ontology id and confidence columns when present in the CSV.

Each label column is written as an ORDERED pandas Categorical, with categories in the
same similarity order the HTML reports use, and the matching palette is written to
`adata.uns['<col>_colors']`. So when the user later does `sc.pl.umap(adata, color=...)`,
the colour<->cell-type relationship matches the reports they signed off on — scanpy
does not invent fresh arbitrary colours. Ordering/colours need the anchor embedding
(`--embedding-key`, falling back to `--viz-key`); without one, categories fall back to
a stable order (Unknown/Unassigned last) and colours are still written.

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

import argparse
import os
import re
import tempfile

import pandas as pd

from cta.io import (check_hierarchy_nesting, eprint, label_color_map, load_adata,
                    require_obs, resolve_workdir, split_csv)

# CSV columns that, when present, get mirrored into companion obs columns.
EXTRA_COLS = ("cl_id", "confidence")
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
            raise SystemExit(
                f"ERROR: {path} must have a '{req}' column; got {list(df.columns)}"
            )
    # Cluster ids are matched as strings against the cluster_key categories.
    # Strip whitespace so a stray trailing space in a hand-authored CSV doesn't
    # silently drop a cluster to Unassigned.
    df["cluster"] = df["cluster"].astype(str).str.strip()
    return df


def build_mapping(df: pd.DataFrame, value_col: str) -> dict[str, str]:
    """cluster(str) -> value, dropping rows with a missing value."""
    out: dict[str, str] = {}
    for cl, val in zip(df["cluster"], df[value_col]):
        if pd.isna(val):
            continue
        out[str(cl)] = str(val)
    return out


def order_label_column(series: pd.Series, adata, cluster_key: str, embedding_key: str | None):
    """Return (ordered categorical Series, hex colour list aligned to its categories).

    Categories are put in similarity order using per-label centroids in the anchor
    embedding (the SAME ordering the HTML reports use); Unknown/Unassigned labels render
    grey and sort to the end. Without a usable embedding, categories fall back to a
    stable order (known labels as first seen, Unknown last) and colours are still
    assigned. The hex list is aligned to the returned categories for adata.uns.
    """
    from cta.io import is_unknown_label
    from matplotlib.colors import to_hex

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
            eprint(f"   note: embedding '{embedding_key}' not in obsm — colours use a "
                   f"stable (non-similarity) order")

    order, color_of = label_color_map(cats, centroids)
    # Keep Unknown-type labels at the end regardless of where similarity placed them.
    order = ([ci for ci in order if not is_unknown_label(cats[ci])]
             + [ci for ci in order if is_unknown_label(cats[ci])])
    ordered_cats = [cats[ci] for ci in order]
    colors = [to_hex(color_of[ci]) for ci in order]
    out = series.astype("category").cat.set_categories(ordered_cats, ordered=True)
    return out, colors


def map_level(adata, cluster_key: str, df: pd.DataFrame, prefix: str, level: str,
              embedding_key: str | None):
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
    cols[main_col], colors[main_col] = order_label_column(
        mapped, adata, cluster_key, embedding_key)

    for extra in EXTRA_COLS:
        if extra in df.columns:
            emap = build_mapping(df, extra)
            if emap:
                col = f"{prefix}_{level}_{extra}"
                cols[col] = clusters.map(emap)  # leave NaN where absent (no Unassigned)
    return cols, colors, unmapped, orphans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--cluster-key", required=True)
    ap.add_argument("--embedding-key", default=None,
                    help="obsm key of the anchor embedding, used to order labels by "
                         "similarity and write a matching uns palette (so saved colours "
                         "match the reports). Falls back to --viz-key.")
    ap.add_argument("--viz-key", default=None, help="fallback obsm key if --embedding-key absent")
    ap.add_argument("--levels", help="comma-separated, e.g. L1,L2,L3,L4 (default: auto-discover)")
    ap.add_argument("--col-prefix", default="cta")
    ap.add_argument("--output", default=None,
                    help="write the annotated object to this path instead of mutating "
                         "--adata in place")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-non-nested", action="store_true",
                    help="override the nested-hierarchy check (NOT recommended — a requested "
                         "hierarchy should nest; a finer label spanning two parents is a bug)")
    args = ap.parse_args()

    workdir = resolve_workdir(args.workdir)

    levels = split_csv(args.levels) or discover_levels(workdir)
    if not levels:
        raise SystemExit(
            f"ERROR: no levels to write. Provide --levels or place "
            f"annotation_<level>.csv files in {workdir / 'labels'}."
        )
    eprint(f">> levels: {', '.join(levels)}")

    adata = load_adata(args.adata, prefer_slim=False)  # must write back the FULL original
    require_obs(adata, args.cluster_key, "cluster key")

    embedding_key = args.embedding_key or args.viz_key
    if not embedding_key:
        eprint("   note: no --embedding-key/--viz-key given — label order/colours will use "
               "a stable (non-similarity) order; pass the anchor embedding to match reports")

    all_cols: dict[str, pd.Series] = {}
    all_colors: dict[str, list] = {}
    level_maps: list = []
    for level in levels:
        df = load_level_table(workdir, level)
        level_maps.append((level, dict(zip(df["cluster"].astype(str), df["label"].astype(str)))))
        cols, colors, unmapped, orphans = map_level(
            adata, args.cluster_key, df, args.col_prefix, level, embedding_key)
        main_col = f"{args.col_prefix}_{level}"

        eprint(f"\n>> {level} -> obs['{main_col}']")
        vc = cols[main_col].value_counts(dropna=False)
        for label, n in vc.items():
            eprint(f"   {label}: {n}")
        for extra_col in cols:
            if extra_col != main_col:
                eprint(f"   (+ companion column '{extra_col}')")
        if main_col in adata.obs:
            eprint(f"   WARN: obs['{main_col}'] already exists — it will be overwritten")
        if unmapped:
            eprint(
                f"   WARN: {len(unmapped)} cluster(s) had no label row -> '{UNASSIGNED}': "
                f"{unmapped}"
            )
        if orphans:
            eprint(
                f"   WARN: {len(orphans)} label row(s) match no cluster in obs"
                f"['{args.cluster_key}'] (id formatting mismatch?): {orphans}"
            )
        all_cols.update(cols)
        all_colors.update(colors)

    # Enforce a nested hierarchy: every finer label must refine exactly one coarser label.
    violations = check_hierarchy_nesting(level_maps)
    if violations:
        msg = ("hierarchy is NOT nested — a level must strictly refine the level above:\n   "
               + "\n   ".join(violations)
               + "\n   Fix the labels so each finer label has one parent (e.g. qualify by "
                 "compartment: 'Ventral telencephalic NPC' vs '… neuron').")
        if args.allow_non_nested:
            eprint(f"\n>> WARN: {msg}\n>> --allow-non-nested set: writing anyway.")
        else:
            raise SystemExit(f"\nERROR: {msg}\n   (override with --allow-non-nested)")
    else:
        eprint(f"\n>> hierarchy nesting OK: {' ⊂ '.join(l for l, _ in reversed(level_maps))}")

    if args.dry_run:
        eprint("\n>> DRY RUN: no file written. Columns that WOULD be added:")
        for c in all_cols:
            extra = f"  (+ uns['{c}_colors'])" if c in all_colors else ""
            eprint(f"   - obs['{c}']{extra}")
        return

    for c, series in all_cols.items():
        adata.obs[c] = series
        if c in all_colors and all_colors[c]:
            # scanpy contract: uns['<col>_colors'][i] <-> obs['<col>'].cat.categories[i]
            adata.uns[f"{c}_colors"] = list(all_colors[c])

    # Write to a temp file in the same dir, then atomically replace, so a failure
    # mid-write never corrupts the destination AnnData. With --output, the original
    # --adata is left untouched.
    dest = os.path.abspath(os.path.expanduser(args.output or args.adata))
    dest_dir = os.path.dirname(dest) or "."
    fd, tmp = tempfile.mkstemp(suffix=".h5ad", dir=dest_dir)
    os.close(fd)
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    where = dest if args.output else f"{dest} (IN-PLACE)"
    eprint(f"\n>> wrote {len(all_cols)} obs column(s) + {len(all_colors)} palette(s) to {where}:")
    for c in all_cols:
        extra = f"  (+ uns['{c}_colors'])" if c in all_colors else ""
        eprint(f"   - obs['{c}']{extra}")


if __name__ == "__main__":
    main()
