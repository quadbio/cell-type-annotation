#!/usr/bin/env python
"""Scaffold one evidence-card STUB per cluster from the exported tables.

The annotator (Claude) reasons cluster-by-cluster and the *reasoning* lives in
the evidence cards. This script removes the hand-rolled, error-prone card-writing
step: it pre-fills the mechanical parts (top COSG / Wilcoxon markers, any
automated cross-check rows) and leaves the judgement sections as TODO for Claude.

Cards are written to <workdir>/evidence/L<level>_cluster<safe-id>.md, matching the
filename convention build_report.py consumes (it extracts the id via
stem.split("cluster", 1)[-1]). Cluster ids can contain '/' (e.g. "24/25_0"), which
would break file paths, so the id is sanitized for the FILENAME only — the real id
is preserved verbatim in the card body.

Existing cards are NOT overwritten unless --overwrite is passed: a card may already
hold human/Claude reasoning that must not be clobbered on a re-run.

Usage:
  python scaffold_cards.py --workdir /proj/ds-annotation-run --level L1 [--overwrite]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cta_io import eprint, resolve_workdir, safe_cluster_id, split_csv

TOP_N = 8


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001 - a malformed optional table shouldn't abort
        eprint(f"   WARN: could not read {path.name} ({type(e).__name__}: {e})")
        return None
    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(str)
    return df


def _rows_for(df: pd.DataFrame | None, cid: str) -> pd.DataFrame:
    if df is None or "cluster" not in df.columns:
        return pd.DataFrame()
    return df[df["cluster"] == cid]


def _read_evidence(path: Path) -> pd.DataFrame | None:
    """Read an evidence table, coercing its cluster id to a 'cluster' column. Tolerates
    tables saved with the cluster as the (possibly unnamed) index/first column."""
    df = _read_csv(path)
    if df is None:
        return None
    if "cluster" not in df.columns:
        df = df.rename(columns={df.columns[0]: "cluster"})
        df["cluster"] = df["cluster"].astype(str)
    return df


def _evidence_section(sources: list[tuple[str, pd.DataFrame]], cid: str, top: int = 16) -> str:
    """The actual per-cluster numbers behind the call — rows from the evidence tables the
    annotator scored on (e.g. curated marker-fraction tables). Wide tables are summarized
    as the cluster's strongest values (descending) so the salient markers surface; the full
    row stays in the table on disk."""
    blocks = []
    for label, df in sources:
        rows = _rows_for(df, cid)
        if rows.empty:
            continue
        r = rows.iloc[0]
        _meta = {"cluster", "n", "n_cells", "cells", "size", "count", "total", "index"}
        nums = {}
        for c in rows.columns:
            if str(c).lower() in _meta:
                continue
            try:
                nums[c] = float(r[c])
            except (TypeError, ValueError):
                pass
        if nums:
            items = sorted(nums.items(), key=lambda kv: -kv[1])[:top]
            body = " · ".join(f"`{k}` {v:.2f}" for k, v in items)
            blocks.append(f"**{label}** — strongest {len(items)} of {len(nums)} values:\n\n{body}")
        else:
            body = " · ".join(f"`{c}` {r[c]}" for c in rows.columns if c != "cluster")
            blocks.append(f"**{label}**:\n\n{body}")
    return "\n\n".join(blocks) if blocks else ""


def _cosg_section(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "- (no COSG markers)"
    rows = rows.sort_values("rank") if "rank" in rows.columns else rows
    out = []
    for _, r in rows.head(TOP_N).iterrows():
        gene = r.get("gene", "?")
        score = r.get("cosg_score")
        if pd.notna(score):
            out.append(f"- **{gene}** (cosg {float(score):.3f})")
        else:
            out.append(f"- **{gene}**")
    return "\n".join(out)


def _wilcoxon_section(rows: pd.DataFrame) -> str:
    if rows.empty:
        return "- (no Wilcoxon markers)"
    # markers_<level>.csv is already ranked per cluster; preserve that order.
    has_lfc = "logfoldchanges" in rows.columns
    has_pct = "pct_nz_group" in rows.columns
    out = []
    for _, r in rows.head(TOP_N).iterrows():
        gene = r.get("names", "?")
        bits = []
        if has_lfc and pd.notna(r.get("logfoldchanges")):
            bits.append(f"logFC {float(r['logfoldchanges']):.2f}")
        if has_pct and pd.notna(r.get("pct_nz_group")):
            bits.append(f"pct {float(r['pct_nz_group']):.2f}")
        suffix = f" ({', '.join(bits)})" if bits else ""
        out.append(f"- **{gene}**{suffix}")
    return "\n".join(out)


# One concise line per source type (these tables repeat in every card, so keep terse).
_CC_LEGEND = {
    "crosscheck": "_Cluster scored against the candidate marker sets (`markers_<level>.json`) "
                  "by 3 independent methods: `overlap_top` = set with most top-DE genes in it, "
                  "`aucell_top` = top AUCell, `aucell_z_top` = top cross-cluster z-scored AUCell; "
                  "`agree` = how many of the 3 concur. On targeted panels, weight overlap/z over raw AUCell._",
    "panel_gating": "_Per candidate set: on-panel vs dropped (off-panel) markers; "
                    "`uncallable` = too few on-panel markers to score._",
    "refmap": "_kNN label transfer from the reference dataset (CellMapper): the cluster's "
              "majority reference label and its fraction._",
}


def _crosscheck_section(sources: list[tuple[str, pd.DataFrame]], cid: str) -> str:
    blocks = []
    for label, df in sources:
        rows = _rows_for(df, cid)
        if rows.empty:
            continue
        cols = [c for c in rows.columns if c != "cluster"]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = "\n".join(
            "| " + " | ".join(str(r[c]) for c in cols) + " |"
            for _, r in rows.iterrows()
        )
        legend = next((v for k, v in _CC_LEGEND.items() if label.startswith(k)), "")
        caption = f"\n\n{legend}" if legend else ""
        blocks.append(f"**{label}**\n\n{header}\n{sep}\n{body}{caption}")
    if not blocks:
        return "(none yet)"
    return "\n\n".join(blocks)


def _sizes_map(workdir: Path, level: str) -> dict[str, int]:
    """Best-effort n-cells per cluster from a sizes table, if one exists."""
    for name in (f"sizes_{level}.csv", f"cluster_sizes_{level}.csv", "sizes.csv"):
        df = _read_csv(workdir / "tables" / name)
        if df is None or "cluster" not in df.columns:
            continue
        count_col = next(
            (c for c in ("n_cells", "n", "size", "count", "cells") if c in df.columns),
            None,
        )
        if count_col is None:
            continue
        try:
            return {str(r["cluster"]): int(r[count_col]) for _, r in df.iterrows()}
        except Exception:  # noqa: BLE001
            continue
    return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--level", required=True, help="hierarchy level tag, e.g. L1")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing cards (default: skip, preserve reasoning)")
    ap.add_argument("--evidence", default=None,
                    help="comma-separated evidence table(s) to embed per cluster — the actual "
                         "numbers behind the call (e.g. the curated marker-fraction tables you "
                         "scored on). Names are resolved under <workdir>/tables/ (a .csv "
                         "suffix is optional) or given as absolute paths. Each cluster's row is "
                         "shown, wide tables summarized as the strongest values.")
    args = ap.parse_args()

    wd = resolve_workdir(args.workdir)
    level = args.level
    tables = wd / "tables"
    evidence = wd / "evidence"

    cosg = _read_csv(tables / f"cosg_{level}.csv")
    if cosg is None or "cluster" not in (cosg.columns if cosg is not None else []):
        raise SystemExit(
            f"ERROR: required table not found or malformed: "
            f"{tables / f'cosg_{level}.csv'} (need a 'cluster' column)"
        )
    markers = _read_csv(tables / f"markers_{level}.csv")

    # Optional automated cross-check tables.
    crosscheck_sources: list[tuple[str, pd.DataFrame]] = []
    for refmap in sorted(tables.glob(f"refmap_{level}_*.csv")):
        df = _read_csv(refmap)
        if df is not None:
            crosscheck_sources.append((refmap.stem, df))
    cc = _read_csv(tables / f"crosscheck_{level}.csv")
    if cc is not None:
        crosscheck_sources.append((f"crosscheck_{level}", cc))
    pg = _read_csv(tables / f"panel_gating_{level}.csv")
    if pg is not None:
        crosscheck_sources.append((f"panel_gating_{level}", pg))

    # User-designated evidence tables — the actual numbers behind the call.
    evidence_sources: list[tuple[str, pd.DataFrame]] = []
    for name in split_csv(args.evidence):
        path = Path(name) if Path(name).is_absolute() else tables / name
        if path.suffix != ".csv":
            path = path.with_suffix(".csv")
        df = _read_evidence(path)
        if df is not None:
            evidence_sources.append((path.stem, df))
        else:
            eprint(f"   WARN: --evidence table not found: {path}")

    sizes = _sizes_map(wd, level)

    # Preserve first-seen order of clusters as they appear in cosg.
    clusters = list(dict.fromkeys(cosg["cluster"].tolist()))

    written, skipped = 0, 0
    for cid in clusters:
        safe = safe_cluster_id(cid)
        out = evidence / f"L{level.lstrip('L')}_cluster{safe}.md"
        # ^ tolerate level given as "L1" or "1": canonicalize the filename prefix.
        if out.exists() and not args.overwrite:
            eprint(f"   skip (exists): {out.name}")
            skipped += 1
            continue

        n_cells = sizes.get(str(cid))
        size_str = f" — n={n_cells} cells" if n_cells is not None else ""

        ev = _evidence_section(evidence_sources, str(cid))
        ev_block = f"## Evidence used\n\n{ev}\n\n" if ev else ""
        card = (
            f"# Cluster {cid}{size_str}\n\n"
            f"_Level {level} evidence card (stub — auto-generated)._\n\n"
            f"## Marker evidence\n\n"
            f"_`cosg` = specificity score, 0–1 (higher = more cluster-specific); "
            f"`logFC` = log2 fold-change of this cluster vs all others; "
            f"`pct` = fraction of cells in the cluster expressing the gene._\n\n"
            f"**Top COSG (specificity):**\n\n"
            f"{_cosg_section(_rows_for(cosg, str(cid)))}\n\n"
            f"**Top Wilcoxon (rank_genes_groups):**\n\n"
            f"{_wilcoxon_section(_rows_for(markers, str(cid)))}\n\n"
            f"{ev_block}"
            f"## Automated cross-check\n\n"
            f"{_crosscheck_section(crosscheck_sources, str(cid))}\n\n"
            f"## Rationale\n\n"
            f"_TODO: Claude fills this in._\n\n"
            f"## Rejected / alternatives\n\n"
            f"_TODO._\n"
        )
        out.write_text(card)
        written += 1

    eprint(f">> scaffolded {written} card(s), skipped {skipped} existing "
           f"(of {len(clusters)} clusters) in {evidence}")
    print(f"written={written} skipped={skipped} total={len(clusters)}")


if __name__ == "__main__":
    main()
