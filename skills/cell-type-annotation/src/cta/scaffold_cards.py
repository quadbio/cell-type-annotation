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

from pathlib import Path

import pandas as pd
import typer

from cta.io import LEVEL_REQ_OPT, WORKDIR_OPT, eprint, resolve_workdir, safe_cluster_id, split_csv

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


# Human-readable gloss per known evidence source (keyed by a substring of the table name,
# lower-cased). `is_pct` = render values as percentages. Keeps the "Evidence used" block
# legible — a short labelled line, not a bare column dump or a bulky table.
_EVIDENCE_GLOSS = {
    "panel_pct": ("curated marker panel, % of cells expressing", True),
    "panel_mean": ("curated marker panel, mean log-norm expression", False),
    "crosscheck": ("cross-method agreement (overlap/AUCell/UCell picking the same type)", False),
    "scores_aucell": ("AUCell signature enrichment, z-scored across clusters", False),
}
_DROP_COLS = {"rank", "index", "level_0"}  # carry no interpretive value in a per-cluster card


def _gloss_for(label: str) -> tuple[str, bool]:
    key = str(label).lower()
    for sub, (expl, is_pct) in _EVIDENCE_GLOSS.items():
        if sub in key:
            return expl, is_pct
    return "", False


def _celltypist_phrase(val, total: int | None = None) -> str | None:
    """Turn a CellTypist per-cell vote distribution (e.g. {'123 DMH Nkx2-4 Glut': 292,
    '095 DMH Prdm13 Gaba': 61, ...}) into one human sentence with the TOP label's real
    fraction of the cluster + the runners-up. `total` is the cluster's cell count; the dict
    holds only the top-3, so we must divide by `total`, not by the dict's sum (which would
    inflate the %). Falls back to the dict sum if no total is supplied."""
    import ast

    try:
        d = ast.literal_eval(val) if isinstance(val, str) else val
    except (ValueError, SyntaxError):
        return None
    if not isinstance(d, dict) or not d:
        return None
    items = sorted(d.items(), key=lambda kv: -int(kv[1]))
    denom = int(total) if total else sum(int(v) for _, v in items)
    top_lab, top_n = items[0][0], int(items[0][1])
    pct = f", {top_n / denom * 100:.0f}% of cells" if denom else ""
    runners = [f"{lab} ({int(n)})" for lab, n in items[1:] if int(n) > 0][:2]
    nxt = f"; next: {', '.join(runners)}" if runners else ""
    return f"top per-cell prediction **{top_lab}** ({top_n}/{denom} cells{pct}){nxt}"


def _evidence_section(sources: list[tuple[str, pd.DataFrame]], cid: str, top: int = 8) -> str:
    """Compact, human-readable view of the numbers behind the call — ONE short labelled line
    per source, not a bulky table. Fractions render as %, the CellTypist prediction renders as
    a sentence (not a raw dict), and sources already shown elsewhere (COSG) are skipped."""
    blocks = []
    for label, df in sources:
        rows = _rows_for(df, cid)
        if rows.empty:
            continue
        r = rows.iloc[0]
        key = str(label).lower()
        if "celltypist" in key:
            # cluster total (for the real %) comes from an n_cells-style column if present
            total = None
            for c in rows.columns:
                if str(c).lower() in ("n_cells", "n", "total", "cells"):
                    try:
                        total = int(float(r[c]))
                    except (TypeError, ValueError):
                        pass
            phrase = next(
                (
                    _celltypist_phrase(r[c], total)
                    for c in rows.columns
                    if str(c).lower() not in ("cluster", "n_cells", "n", "total", "cells")
                    and _celltypist_phrase(r[c], total) is not None
                ),
                None,
            )
            if phrase:
                blocks.append(
                    f"**CellTypist cross-check** _(model hypothesis, not ground truth)_: {phrase}"
                )
            continue
        if key.startswith("cosg"):
            continue  # COSG specificity is already in the Marker-evidence section above
        expl, is_pct = _gloss_for(label)
        _meta = {"cluster", "n", "n_cells", "cells", "size", "count", "total", "index"}
        nums = {}
        for c in rows.columns:
            if str(c).lower() in _meta or str(c).lower() in _DROP_COLS:
                continue
            try:
                nums[c] = float(r[c])
            except (TypeError, ValueError):
                pass
        head = f"**{label}**" + (f" — _{expl}_" if expl else "")
        if nums:
            items = sorted(nums.items(), key=lambda kv: -kv[1])[:top]
            fmt = (lambda k, v: f"{k} {v * 100:.0f}%") if is_pct else (lambda k, v: f"{k} {v:.2f}")
            body = " · ".join(fmt(k, v) for k, v in items)
            more = f" _(top {len(items)} of {len(nums)})_" if len(nums) > len(items) else ""
            blocks.append(f"{head}: {body}{more}")
        else:
            body = " · ".join(f"`{c}` = {r[c]}" for c in rows.columns if c != "cluster")
            blocks.append(f"{head}: {body}")
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
    "crosscheck": "_Cluster scored against the candidate marker sets (`markers_<level>.json`) by "
    "independent methods: `overlap_top` = set with most top-DE genes in it, "
    "`aucell_top` = top raw AUCell (biased — shown for reference only), "
    "`aucell_z_top` = top cross-cluster z-scored AUCell, `ucell_top` = top pyUCell. "
    "`agree` = how many of {`overlap_top`, `aucell_z_top`, `ucell_top`} concur "
    "(raw `aucell_top` is excluded as biased). On targeted panels, weight overlap/z "
    "over raw AUCell._",
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
            "| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in rows.iterrows()
        )
        legend = next((v for k, v in _CC_LEGEND.items() if label.startswith(k)), "")
        caption = f"\n\n{legend}" if legend else ""
        blocks.append(f"**{label}**\n\n{header}\n{sep}\n{body}{caption}")
    if not blocks:
        return "(none yet)"
    return "\n\n".join(blocks)


def _labels_map(workdir: Path, level: str, explicit: str | None) -> dict[str, dict]:
    """Per-cluster label metadata from the labels CSV — the SINGLE SOURCE OF TRUTH for the
    call and its reasoning. Carrying `rationale`/`rejected`/`confidence`/`label`/`cl_id` in
    the CSV (not hand-edited into N markdown files) is what makes the reasoning durable: a
    re-scaffold re-renders the cards FROM the CSV instead of wiping a TODO the annotator
    already filled. Returns {} if no labels CSV exists yet (cards fall back to TODO stubs)."""
    path = Path(explicit) if explicit else workdir / "labels" / f"annotation_{level}.csv"
    df = _read_csv(path)
    if df is None or "cluster" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        cid = str(r["cluster"]).strip()
        rec = {}
        for col in ("label", "state", "cl_id", "confidence", "rationale", "rejected"):
            if col in df.columns and pd.notna(r.get(col)):
                rec[col] = r[col]
        out[cid] = rec
    return out


def _section_or_todo(rec: dict, key: str, placeholder: str) -> str:
    """Render a reasoning section from the labels CSV, or a TODO stub if absent/blank."""
    val = str(rec.get(key, "")).strip()
    return val if val else placeholder


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


def cli(
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_REQ_OPT,
    overwrite: bool = typer.Option(
        False, help="overwrite existing cards (default: skip, preserve reasoning)"
    ),
    evidence: str | None = typer.Option(
        None,
        help="comma-separated evidence table(s) to embed per cluster — the actual "
        "numbers behind the call (e.g. the curated marker-fraction tables you "
        "scored on). Names are resolved under <workdir>/tables/ (a .csv "
        "suffix is optional) or given as absolute paths. Each cluster's row is "
        "shown, wide tables summarized as the strongest values.",
    ),
    labels: str | None = typer.Option(
        None,
        help="labels CSV to source the per-cluster call + reasoning from "
        "(default: <workdir>/labels/annotation_<level>.csv). Columns "
        "`label`,`state`,`confidence`,`cl_id`,`rationale`,`rejected` are rendered into "
        "each card; this is the single source of truth, so re-scaffolding "
        "after you fill the CSV refreshes the cards without losing reasoning.",
    ),
) -> None:
    """Scaffold one evidence-card stub per cluster from the exported tables."""
    wd = resolve_workdir(workdir)
    tables = wd / "tables"
    evidence_dir = wd / "evidence"

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
    for name in split_csv(evidence):
        path = Path(name) if Path(name).is_absolute() else tables / name
        if path.suffix != ".csv":
            path = path.with_suffix(".csv")
        df = _read_evidence(path)
        if df is not None:
            evidence_sources.append((path.stem, df))
        else:
            eprint(f"   WARN: --evidence table not found: {path}")

    sizes = _sizes_map(wd, level)
    labels_map = _labels_map(wd, level, labels)

    # Preserve first-seen order of clusters as they appear in cosg.
    clusters = list(dict.fromkeys(cosg["cluster"].tolist()))

    written, skipped, missing_rationale = 0, 0, []
    for cid in clusters:
        safe = safe_cluster_id(cid)
        out = evidence_dir / f"L{level.lstrip('L')}_cluster{safe}.md"
        # ^ tolerate level given as "L1" or "1": canonicalize the filename prefix.
        if out.exists() and not overwrite:
            eprint(f"   skip (exists): {out.name}")
            skipped += 1
            continue

        n_cells = sizes.get(str(cid))
        size_str = f" — n={n_cells} cells" if n_cells is not None else ""

        rec = labels_map.get(str(cid), {})
        call_bits = []
        if rec.get("label"):
            call_bits.append(f"**Call:** {rec['label']}")
        # `state` is the orthogonal cell-STATE axis (normal / low-quality / cycling / doublet …),
        # independent of the cell-TYPE label — surfaced here so a cycling or low-quality cluster
        # is flagged even when it still carries a confident type call.
        if str(rec.get("state", "")).strip():
            call_bits.append(f"state: {str(rec['state']).strip()}")
        if rec.get("cl_id"):
            call_bits.append(f"`{rec['cl_id']}`")
        if rec.get("confidence") not in (None, ""):
            try:
                call_bits.append(f"confidence {float(rec['confidence']):.2f}")
            except (TypeError, ValueError):
                pass
        call_line = (" · ".join(call_bits) + "\n\n") if call_bits else ""
        rationale = _section_or_todo(rec, "rationale", "_TODO: Claude fills this in._")
        rejected = _section_or_todo(rec, "rejected", "_TODO._")
        if rationale.startswith("_TODO"):
            missing_rationale.append(str(cid))

        ev = _evidence_section(evidence_sources, str(cid))
        ev_block = f"## Evidence used\n\n{ev}\n\n" if ev else ""
        card = (
            f"# Cluster {cid}{size_str}\n\n"
            f"{call_line}"
            f"_Level {level} evidence card._\n\n"
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
            f"{rationale}\n\n"
            f"## Rejected / alternatives\n\n"
            f"{rejected}\n"
        )
        out.write_text(card)
        written += 1

    eprint(
        f">> scaffolded {written} card(s), skipped {skipped} existing "
        f"(of {len(clusters)} clusters) in {evidence_dir}"
    )
    if labels_map and missing_rationale:
        eprint(
            f"   NOTE: {len(missing_rationale)} card(s) still have a TODO rationale "
            f"(no `rationale` in the labels CSV): {', '.join(missing_rationale[:12])}"
            + (" …" if len(missing_rationale) > 12 else "")
            + "\n   Fill the `rationale`/`rejected` columns in the labels CSV and re-run; "
            "build_report.py --strict will block descent until they're filled."
        )
    elif not labels_map:
        eprint(
            "   NOTE: no labels CSV found — cards carry TODO rationale stubs. Write "
            f"labels/annotation_{level}.csv (with `rationale`/`rejected` columns) and "
            "re-scaffold to render the reasoning into the cards."
        )
    print(
        f"written={written} skipped={skipped} total={len(clusters)} "
        f"missing_rationale={len(missing_rationale)}"
    )
