#!/usr/bin/env python
"""Assemble a self-contained HTML report for one hierarchy level.

Reads the evidence cards (markdown in <workdir>/evidence/L<level>_*.md) and the
labels table (<workdir>/labels/annotation_L<level>.csv), and renders a single HTML
file to <workdir>/reports/report_L<level>.html for the user to review at the
checkpoint. Flagged clusters (Unknown / low confidence) float to the top.

If --adata (+ keys) is given, embeds plots (base64 PNG, self-contained):
  - UMAP on the viz embedding, colored by the assigned label;
  - dotplot of the top markers per assigned label;
  - OPTIONAL generic composition panel: label proportions split by a user-named
    categorical obs (--split-by). NEVER hard-wired to any dataset's metadata.
The summary table also shows the top marker genes (from COSG) behind each cluster.

The evidence cards are authored by Claude (the reasoning lives there); this script
just packages them for human review — no annotation logic here.

Labels CSV columns (flexible; missing ones are tolerated):
  cluster, label, cl_id, confidence, status, note

Usage:
  python build_report.py --workdir /proj/run --level L1 [--title "Level 1 — broad class"]
  # with plots:
  python build_report.py --workdir /proj/run --level L1 --adata A.h5ad \
      --cluster-key leiden --viz-key X_umap [--lognorm-layer lognorm] [--split-by sample]
"""
from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from cta.constants import REPORT_LOW_CONF_THRESHOLD
from cta.io import (
    LEVEL_REQ_OPT, WORKDIR_OPT, eprint, is_unknown_label, resolve_workdir,
    safe_cluster_id)


def md_to_html(s: str) -> str:
    try:
        import markdown as _md
        return _md.markdown(s, extensions=["tables", "fenced_code"])
    except Exception:  # markdown is optional; fall back to <pre>
        return f"<pre>{html.escape(s)}</pre>"


TEMPLATE = """<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;
   margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #ddd;padding-bottom:.3rem}}
 h2{{margin-top:2rem;border-bottom:1px solid #eee;padding-bottom:.2rem}}
 .card{{border:1px solid #e0e0e0;border-radius:8px;padding:.5rem 1rem;margin:.5rem 0;
   background:#fafafa}}
 .flagged{{border-color:#e0a800;background:#fffbe6}}
 .badge{{display:inline-block;padding:.1rem .5rem;border-radius:4px;font-size:.78rem;
   font-weight:600;margin-left:.5rem}}
 .unknown{{background:#ffe0e0;color:#a00}} .ok{{background:#e0f0e0;color:#070}}
 .lowconf{{background:#fff0d0;color:#a60}}
 .meta{{color:#666;font-size:.9rem}} table{{border-collapse:collapse;font-size:.9rem;width:100%}}
 td,th{{border:1px solid #ddd;padding:.3rem .6rem;vertical-align:top;text-align:left}}
 th{{background:#f4f4f4;position:sticky;top:0}}
 tr:nth-child(even){{background:#fafafa}}
 .group{{border:1px solid #e6e6e6;border-radius:8px;margin:.6rem 0;background:#fff}}
 .group>summary{{padding:.6rem 1rem;font-size:1.02rem;background:#f7f7f9;border-radius:8px}}
 .group[open]>summary{{border-bottom:1px solid #eee;border-radius:8px 8px 0 0}}
 .grpbody{{padding:.3rem 1rem .8rem}}
 .gflag>summary{{background:#fffbe6}}
 summary{{cursor:pointer;font-weight:600}}
 .conf{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
 .legend td{{border:none;padding:.05rem .4rem}} .sw{{display:inline-block;width:.9rem;height:.9rem;border-radius:2px;vertical-align:middle}}
 .nowrap{{white-space:nowrap}} code{{font-size:.85em}}
 .plot{{margin:1.5rem 0}} .plot img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
 .plot h3{{margin:0 0 .3rem}}
 .figcap{{font-size:13px;color:#666;margin:.45rem 0 0;max-width:760px;line-height:1.5}}
 details.plotbox>summary{{font-size:1.05rem;margin:.5rem 0}}
</style></head><body>
<h1>{title}</h1>
<p class="meta">{n} clusters · {n_regions} distinct labels · {n_flagged} flagged for review · level <code>{level}</code></p>
<details class="plotbox"><summary>How to read this report</summary>
<table class="legend">
<tr><td><span class="badge ok">assigned</span></td><td>A confident label is assigned (confidence ≥ 0.6).</td></tr>
<tr><td><span class="badge lowconf">LOW CONF</span></td><td>Label assigned but confidence &lt; 0.6 — the evidence is weak; review the card.</td></tr>
<tr><td><span class="badge lowconf">NEEDS_SUBCLUSTER</span></td><td>The cluster mixes identities; it should be split and re-annotated rather than given one label.</td></tr>
<tr><td><span class="badge lowconf">REFUTED</span></td><td>A critic subagent overturned the proposed label; it was revised or downgraded.</td></tr>
<tr><td><span class="badge unknown">UNKNOWN</span></td><td>Evidence is irreconcilable — an honest <code>Unknown</code> (a first-class outcome, not a failure).</td></tr>
</table>
<p class="figcap">Clusters are <b>flagged for review</b> when they are Unknown, refuted, need sub-clustering, or have
confidence &lt; 0.6; flagged clusters float to the top and their evidence cards open by default.
<b>Confidence</b> is a 0–1 value <b>set by the annotator's judgement</b> — subjective, not a computed
quantity. (The advisory <code>cluster_confidence.py</code> can compare it to a COSG-margin proxy, but that
comparison is not shown in this report.) Each label is backed by a per-cluster <b>evidence card</b> below
showing the exact markers and cross-checks behind the call.</p>
</details>
{rollup}
{plots}
{summary_table}
<h2>Evidence cards <span class="meta">(grouped by label; flagged groups open)</span></h2>
<details class="plotbox"><summary>How to read an evidence card</summary>
<ul>
<li><b>Marker evidence</b> — the cluster's own top markers. <b>COSG</b> = specificity (0–1, higher = more exclusive to this cluster); <b>Wilcoxon</b> ranks differential genes (<b>logFC</b> = log2 fold-change vs all other cells, <b>pct</b> = fraction of the cluster expressing it).</li>
<li><b>Evidence used</b> — the numbers the call rests on, one short labelled line per source (e.g. which curated markers are on, as % of cells). The CellTypist line is the model's <b>top per-cell prediction</b> — the most common per-cell label with its share of the cluster's cells (often well under 50%, since fine reference models split votes across related subtypes) — a cross-check hypothesis, not ground truth.</li>
<li><b>Automated cross-check</b> — an independent method's prediction (per-cluster scoring agreement / reference mapping), used as a hypothesis to agree or disagree with — never as ground truth.</li>
<li><b>Rationale</b> — why this label was chosen from the evidence above. <b>Rejected / alternatives</b> — other identities considered and why they were ruled out.</li>
<li><b>Confidence</b> — the badge; low-confidence or flagged clusters float to the top and open by default.</li>
</ul>
</details>
{cards}
</body></html>"""


def _status_badge(status: str, conf):
    status = (status or "").lower()
    if status in ("unknown",):
        return '<span class="badge unknown">UNKNOWN</span>'
    try:
        if conf is not None and float(conf) < REPORT_LOW_CONF_THRESHOLD:
            return '<span class="badge lowconf">LOW CONF</span>'
    except (TypeError, ValueError):
        pass
    if status in ("refuted", "needs_subcluster"):
        return f'<span class="badge lowconf">{status.upper()}</span>'
    return '<span class="badge ok">assigned</span>'


def _top_markers_by_cluster(wd: Path, level: str, n: int = 5) -> dict[str, str]:
    """cluster -> 'g1, g2, ...' from the COSG specificity table, if present."""
    p = wd / "tables" / f"cosg_{level}.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out = {}
    for cl, g in df.sort_values("rank").groupby(df["cluster"].astype(str)):
        out[str(cl)] = ", ".join(g.head(n)["gene"].astype(str))
    return out


def _cluster_sizes(adata_path, cluster_key) -> dict[str, int]:
    """cluster id -> n cells (backed read of obs only; empty on any failure)."""
    if not adata_path or not cluster_key:
        return {}
    try:
        import anndata as ad
        obs = ad.read_h5ad(adata_path, backed="r").obs
        if cluster_key not in obs:
            return {}
        vc = obs[cluster_key].astype(str).value_counts()
        return {str(k): int(v) for k, v in vc.items()}
    except Exception:  # noqa: BLE001
        return {}


def _b64_img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def _find_card_files(wd: Path, level: str) -> list[Path]:
    """Evidence card files for a level. scaffold_cards.py writes 'L<level>_cluster<id>.md';
    fall back to cards written without the leading 'L'."""
    cards_dir = wd / "evidence"
    cards = sorted(cards_dir.glob(f"L{level}_*.md"))
    if not cards:
        cards = sorted(cards_dir.glob(f"{level}_*.md"))
    return cards


def _print_gate(problems: list[str], warnings: list[str]) -> None:
    """Render the level-completeness verdict to stderr (shared by render and --check-only)."""
    if problems or warnings:
        eprint("\n>> level-completeness check:")
        for p in problems:
            eprint(f"   ✗ {p}")
        for w in warnings:
            eprint(f"   ⚠ {w}")
    else:
        eprint("\n>> level-completeness check: ✓ report + labels + filled cards + signed gate")


def cli(
    workdir: str = WORKDIR_OPT,
    level: str = LEVEL_REQ_OPT,
    title: Optional[str] = typer.Option(None, help="report title"),
    adata: Optional[str] = typer.Option(
        None, help="enable plots (UMAP/dotplot/composition)"),
    cluster_key: Optional[str] = typer.Option(None, help="obs column holding the clusters"),
    viz_key: Optional[str] = typer.Option(
        None, help="obsm key for the UMAP/tSNE viz embedding"),
    embedding_key: Optional[str] = typer.Option(
        None,
        help="obsm key of the anchor embedding (used to ORDER labels by "
             "similarity so similar types get adjacent colors; defaults to --viz-key)"),
    lognorm_layer: Optional[str] = typer.Option(None, help="layer with log-normalized data"),
    split_by: Optional[str] = typer.Option(
        None,
        help="OPTIONAL categorical obs to split composition by — comma-separated "
             "for multiple panels (e.g. 'condition,week'); reuses uns colors. "
             "Generic; never assume a specific column exists."),
    strict: bool = typer.Option(
        False,
        help="treat an incomplete level as an error (exit 1): every cluster must "
             "have a non-blank label AND a filled (non-TODO) rationale. This is the "
             "gate to run before descending — it catches the silent-skip failure "
             "mode where a level is 'done' but its reasoning was never written."),
    check_only: bool = typer.Option(
        False,
        help="re-evaluate the completeness gate WITHOUT rendering or writing the "
             "report (read-only). Use this to re-check after recording a sign-off "
             "— never re-run the renderer just to flip the gate. With --strict it "
             "exits 1 when the level is incomplete; it writes nothing."),
) -> None:
    """Assemble a self-contained HTML report for one hierarchy level."""
    from cta.report_plots import discover_plots, make_plots

    wd = resolve_workdir(workdir)
    title = title or f"Cell-type annotation — level {level}"

    labels_path = wd / "labels" / f"annotation_{level}.csv"
    labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()

    # attach top markers (from COSG) per cluster to the summary
    top_markers = _top_markers_by_cluster(wd, level)
    if not labels.empty and "cluster" in labels.columns and top_markers:
        labels = labels.copy()
        labels["top_markers"] = labels["cluster"].astype(str).map(top_markers)

    # Read-only gate: re-evaluate completeness without rendering or overwriting the report.
    # This is how you re-check after recording a sign-off — never re-run the renderer just to
    # flip the gate (omitting --adata on such a re-run is what silently dropped the figures).
    if check_only:
        problems, warnings = _level_completeness(
            wd, level, labels, _find_card_files(wd, level))
        _print_gate(problems, warnings)
        raise SystemExit(1 if (problems and strict) else 0)

    # Figures: (re)generate PNGs only when an AnnData is supplied, but ALWAYS embed whatever is
    # already on disk in plots/ (via discover_plots). So rebuilding without --adata reuses the
    # existing figures instead of silently dropping them.
    if adata:
        make_plots(
            adata_path=adata, cluster_key=cluster_key, viz_key=viz_key,
            embedding_key=embedding_key, lognorm_layer=lognorm_layer,
            split_by=split_by, level=level, labels=labels, wd=wd)
    plot_items = discover_plots(wd, level)
    plots_inner = ""
    for heading, caption, p in plot_items:
        try:
            cap = f'<p class="figcap">{html.escape(caption)}</p>' if caption else ""
            plots_inner += (f'<figure class="plot"><h3>{html.escape(heading)}</h3>'
                            f'<img src="{_b64_img(p)}" alt="{html.escape(heading)}">{cap}</figure>')
        except Exception:  # noqa: BLE001
            pass
    plots_html = (f'<details class="plotbox" open><summary>Figures (cell map / markers / '
                  f'composition)</summary>{plots_inner}</details>') if plots_inner else ""

    # evidence cards (scaffold_cards.py writes "L<level>_cluster<id>.md")
    card_files = _find_card_files(wd, level)

    def is_flagged(row) -> bool:
        # An explicit status (if the labels CSV carries one) OR an Unknown label always flags —
        # don't rely on the optional `status` column alone, or an Unknown/refuted call with
        # confidence >= 0.6 would silently pass review.
        st = str(row.get("status", "")).lower()
        if st in ("unknown", "refuted", "needs_subcluster"):
            return True
        if is_unknown_label(str(row.get("label", ""))):
            return True
        try:
            return float(row.get("confidence", 1)) < REPORT_LOW_CONF_THRESHOLD
        except (TypeError, ValueError):
            return False

    flagged_clusters = set()
    cl2label, cl2conf = {}, {}
    rollup_html = summary_rows = ""
    n_regions = 0
    if not labels.empty:
        for _, r in labels.iterrows():
            if is_flagged(r):
                flagged_clusters.add(str(r.get("cluster")))
            cl2label[str(r.get("cluster"))] = str(r.get("label", "—"))
            cl2conf[str(r.get("cluster"))] = r.get("confidence")
        n_regions = labels["label"].nunique() if "label" in labels.columns else 0

        # --- rollup: one row per label (cells, n clusters, flagged), most cells first ---
        if "label" in labels.columns:
            sizes = _cluster_sizes(adata, cluster_key)
            total_cells = sum(sizes.values()) or 0
            rr = ""
            rollup = []
            for lab in labels["label"].drop_duplicates():
                sub = labels[labels["label"] == lab]
                cls_ids = [str(c) for c in sub["cluster"]]
                ncell = sum(sizes.get(c, 0) for c in cls_ids)
                nflag = sum(c in flagged_clusters for c in cls_ids)
                confs = pd.to_numeric(sub.get("confidence"), errors="coerce") if "confidence" in sub else None
                cavg = f"{confs.mean():.2f}" if confs is not None and confs.notna().any() else "—"
                rollup.append((lab, ncell, len(cls_ids), cavg, nflag, cls_ids))
            # order by cells desc when known, else by #clusters desc
            rollup.sort(key=lambda r: (-(r[1]), -r[2]))
            has_cells = total_cells > 0
            for lab, ncell, ncl, cavg, nflag, cls_ids in rollup:
                cell_td = (f"<td class='conf'>{ncell:,}</td><td class='conf'>"
                           f"{100*ncell/total_cells:.1f}%</td>") if has_cells else ""
                flag_txt = f'<span class="badge lowconf">{nflag} flagged</span>' if nflag else ""
                rr += (f"<tr><td>{html.escape(str(lab))}</td>{cell_td}"
                       f"<td class='conf'>{ncl}</td><td class='conf'>{cavg}</td>"
                       f"<td><code>{html.escape(', '.join(cls_ids))}</code> {flag_txt}</td></tr>")
            cell_hdr = "<th>cells</th><th>%</th>" if has_cells else ""
            rollup_html = (f"<h2>Labels overview</h2><table><tr><th>label</th>{cell_hdr}"
                           f"<th>#clusters</th><th>avg conf</th><th>clusters</th></tr>{rr}</table>")

        # --- compact per-cluster summary (key cols only; full note lives in the card) ---
        cols = [c for c in ["cluster", "label", "cl_id", "confidence", "top_markers"] if c in labels.columns]
        sl = labels.sort_values(
            by=[c for c in ["label", "confidence"] if c in labels.columns]) if cols else labels
        head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        body = ""
        for _, r in sl.iterrows():
            badge = _status_badge(str(r.get("status", "")), r.get("confidence"))
            cells = ""
            for c in cols:
                v = html.escape(str(r[c]))
                cells += f'<td class="conf">{v}</td>' if c == "confidence" else f"<td>{v}</td>"
            body += f"<tr>{cells}<td>{badge}</td></tr>"
        summary_rows = ('<details><summary><h2 style="display:inline">Per-cluster summary</h2>'
                        f"</summary><table><tr>{head}<th>flag</th></tr>{body}</table></details>")

    # --- evidence cards grouped by label (region), flagged groups open ---
    def card_cluster(p: Path) -> str:
        stem = p.stem
        return stem.split("cluster", 1)[-1] if "cluster" in stem else stem

    # card filenames sanitize '/' -> '_' (e.g. 24/25_0 -> 24_25_0); match labels via sanitized id
    cl2label_safe = {safe_cluster_id(c): lab for c, lab in cl2label.items()}
    flagged_safe = {safe_cluster_id(c) for c in flagged_clusters}
    groups: dict[str, list[Path]] = {}
    for p in card_files:
        groups.setdefault(cl2label_safe.get(card_cluster(p), "—"), []).append(p)
    # order groups: flagged groups first, then by size desc, then name
    def grp_key(item):
        lab, ps = item
        has_flag = any(card_cluster(p) in flagged_safe for p in ps)
        return (not has_flag, -len(ps), lab)
    cards_html = ""
    for lab, ps in sorted(groups.items(), key=grp_key):
        ps.sort(key=lambda p: (card_cluster(p) not in flagged_safe, p.name))
        gflag = any(card_cluster(p) in flagged_safe for p in ps)
        inner = ""
        for p in ps:
            cid = card_cluster(p)
            flagged = cid in flagged_safe
            cardbody = md_to_html(p.read_text())
            cls = "card flagged" if flagged else "card"
            inner += (f'<div class="{cls}"><details {"open" if flagged else ""}>'
                      f'<summary>cluster {html.escape(cid)}'
                      f'{" — FLAGGED" if flagged else ""}</summary>{cardbody}</details></div>')
        flagtag = f' · {sum(card_cluster(p) in flagged_safe for p in ps)} flagged' if gflag else ""
        cards_html += (f'<details class="group{" gflag" if gflag else ""}"{" open" if gflag else ""}>'
                       f'<summary>{html.escape(lab)} <span class="meta">({len(ps)} cluster'
                       f'{"s" if len(ps)!=1 else ""}{flagtag})</span></summary>'
                       f'<div class="grpbody">{inner}</div></details>')

    out = wd / "reports" / f"report_{level}.html"
    out.write_text(TEMPLATE.format(
        title=html.escape(title), level=html.escape(level),
        n=len(card_files) or len(labels),
        n_regions=n_regions,
        n_flagged=len(flagged_clusters),
        rollup=rollup_html,
        plots=plots_html, summary_table=summary_rows,
        cards=cards_html or "<p>No evidence cards yet.</p>",
    ))
    eprint(f">> wrote {out}")
    print(str(out))

    # --- Level-completeness gate ----------------------------------------------------------
    # A level is "complete" only when (1) the report exists [just written], (2) every cluster
    # carries a non-blank label, (3) every card's rationale is filled (not a TODO stub), and
    # (4) the level's assumptions gate is signed off. Skipping any of these is the usual way a
    # run looks finished while its reasoning was never recorded — so we surface it loudly, and
    # under --strict refuse to pass (exit 1) so the descent to the next level is blocked.
    problems, warnings = _level_completeness(wd, level, labels, card_files)
    _print_gate(problems, warnings)
    if problems and strict:
        raise SystemExit(
            f"ERROR: level {level} is incomplete ({len(problems)} blocking issue(s)); "
            "fix the labels CSV / fill rationale, then re-run. Refusing to pass --strict.")


def _level_completeness(wd, level, labels, card_files):
    """Return (problems, warnings). `problems` block --strict; `warnings` are advisory."""
    import json as _json
    problems, warnings = [], []
    # (2) every labelled cluster has a non-blank label
    if labels is None or getattr(labels, "empty", True):
        problems.append(f"no labels table at labels/annotation_{level}.csv")
    else:
        blank = [str(r["cluster"]) for _, r in labels.iterrows()
                 if "label" not in labels.columns or str(r.get("label", "")).strip() == ""]
        if blank:
            problems.append(f"{len(blank)} cluster(s) with a blank label: {', '.join(blank[:12])}")
        # (3) rationale present in the CSV for every cluster (single source of truth)
        if "rationale" in labels.columns:
            no_rat = [str(r["cluster"]) for _, r in labels.iterrows()
                      if str(r.get("rationale", "")).strip() == ""]
            if no_rat:
                problems.append(f"{len(no_rat)} cluster(s) missing a `rationale`: "
                                f"{', '.join(no_rat[:12])}")
        else:
            problems.append("labels CSV has no `rationale` column — add it (single source of "
                            "truth for card reasoning) so cards don't carry TODO stubs")
    # (3b) belt-and-suspenders: scan rendered cards for leftover TODO stubs
    todo_cards = [p.name for p in (card_files or []) if "_TODO" in p.read_text()]
    if todo_cards:
        problems.append(f"{len(todo_cards)} card(s) still contain a TODO rationale: "
                        f"{', '.join(todo_cards[:8])}")
    # (3c) staleness: cards are snapshots of the tables. If a table was regenerated AFTER the
    # cards were last scaffolded (e.g. a fixed cross-check table), the cards silently show the
    # old numbers — exactly the failure where a corrected CellTypist table left stale cards.
    # Warn (don't block) so the annotator re-scaffolds; heuristic, so advisory only.
    try:
        cards = list(card_files or [])
        tdir = wd / "tables"
        tbls = list(tdir.glob("*.csv")) if tdir.exists() else []
        if tbls:
            newest_tbl = max(t.stat().st_mtime for t in tbls)
            stale = [p.name for p in cards if p.stat().st_mtime < newest_tbl - 1]
            if stale:
                warnings.append(f"{len(stale)} card(s) are OLDER than the newest table in "
                                f"tables/ — they may show stale numbers; re-run "
                                f"scaffold_cards.py --overwrite (e.g. {', '.join(stale[:5])})")
            # same check for the embedded figures: since the report reuses on-disk PNGs, a
            # rebuild without --adata after the tables changed can show stale plots. Advisory.
            pdir = wd / "plots"
            figs = (list(pdir.glob(f"umap_{level}.png")) + list(pdir.glob(f"dotplot_{level}.png"))
                    + list(pdir.glob(f"composition_{level}_*.png"))) if pdir.exists() else []
            stale_f = [p.name for p in figs if p.stat().st_mtime < newest_tbl - 1]
            if stale_f:
                warnings.append(f"{len(stale_f)} figure(s) in plots/ are OLDER than the newest "
                                f"table — re-run build_report.py with --adata to refresh them "
                                f"(e.g. {', '.join(stale_f[:4])})")
    except OSError:
        pass
    # (4) gate signed for this level — BLOCKS --strict: labels must not be treated as final
    # before the Gate-A assumptions sign-off is recorded (this is what Gate A is for).
    rs = wd / "run_state.json"
    if rs.exists():
        try:
            gates = _json.loads(rs.read_text()).get("gates", {}).get("signed_off_levels", [])
            if level not in gates and f"cta_{level}" not in gates:
                problems.append(f"assumptions gate for {level} not recorded in "
                                "run_state.json gates.signed_off_levels (sign off first)")
        except Exception:  # noqa: BLE001
            problems.append("could not read run_state.json to verify the assumptions gate")
    else:
        problems.append("no run_state.json — can't verify the assumptions gate is signed")
    return problems, warnings
