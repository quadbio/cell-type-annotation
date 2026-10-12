"""The labels table of a level, its completeness check, and its Markdown report.

``results/labels_<level>.csv`` is the single source of truth for a level: the call and the
reasoning behind it, one row per cluster. The report renders from it and from the evidence
tables, so re-rendering never loses reasoning, and a level is not finished while
:func:`check_level` reports problems.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from cell_type_annotation.checks import validate_cl_ids

#: Columns of ``results/labels_<level>.csv``. ``state`` is the orthogonal cell-state axis (normal,
#: low-quality, cycling, doublet, ...), exempt from CL ids and from nesting.
LABEL_COLUMNS = ("cluster", "label", "state", "cl_id", "confidence", "status", "rationale", "rejected")
STATUSES = ("assigned", "unknown", "needs_subcluster", "refuted")
#: Calls below this confidence are flagged for review.
LOW_CONFIDENCE = 0.6


def _blank(value) -> bool:
    return pd.isna(value) or not str(value).strip()


def check_level(labels: pd.DataFrame) -> list[str]:
    """What still stops a level from being complete.

    A level is complete when every cluster has a label, a status, a rationale, a confidence in
    [0, 1] and a well-formed (or blank) ``cl_id``. Descend only on an empty list, and only after
    the level's assumptions are signed off.

    Parameters
    ----------
    labels
        The level's labels table.

    Returns
    -------
    One message per problem.
    """
    missing = [c for c in LABEL_COLUMNS if c not in labels.columns]
    if missing:
        return [f"labels table is missing columns {missing}"]
    problems = []
    for column in ("label", "status", "rationale"):
        blank = labels.loc[labels[column].map(_blank), "cluster"].astype(str).tolist()
        if blank:
            problems.append(f"{len(blank)} cluster(s) without a {column}: {', '.join(blank)}")
    bad_status = labels.loc[~labels["status"].map(_blank) & ~labels["status"].isin(STATUSES), "cluster"]
    if len(bad_status):
        problems.append(f"status must be one of {list(STATUSES)}: cluster(s) {', '.join(map(str, bad_status))}")
    confidence = pd.to_numeric(labels["confidence"], errors="coerce")
    bad_conf = labels.loc[confidence.isna() | ~confidence.between(0, 1), "cluster"].astype(str).tolist()
    if bad_conf:
        problems.append(f"confidence must be a number in [0, 1]: cluster(s) {', '.join(bad_conf)}")
    problems += validate_cl_ids(dict(zip(labels["cluster"].astype(str), labels["cl_id"], strict=True)))
    return problems


def prediction_summary(clusters, labels, confidence=None) -> pd.DataFrame:
    """Summarize a cross-check's per-cell predictions per cluster, the same way for every method.

    Parameters
    ----------
    clusters
        Cluster per cell.
    labels
        Predicted label per cell.
    confidence
        Optional per-cell confidence. Methods differ in what it means: compare it within a method,
        never across methods.

    Returns
    -------
    One row per cluster: ``top1``, ``frac1``, ``top2``, ``frac2``, ``mean_conf``.
    """
    cells = pd.DataFrame(
        {
            "cluster": np.asarray(clusters).astype(str),
            "label": np.asarray(labels).astype(str),
            "conf": np.nan if confidence is None else np.asarray(confidence, dtype=float),
        }
    )
    rows = []
    for cluster, group in cells.groupby("cluster", sort=False):
        share = group["label"].value_counts(normalize=True)
        rows.append(
            {
                "cluster": cluster,
                "top1": share.index[0],
                "frac1": round(float(share.iloc[0]), 3),
                "top2": share.index[1] if len(share) > 1 else "",
                "frac2": round(float(share.iloc[1]), 3) if len(share) > 1 else 0.0,
                "mean_conf": round(float(group["conf"].mean()), 3),
            }
        )
    return pd.DataFrame(rows, columns=["cluster", "top1", "frac1", "top2", "frac2", "mean_conf"])


def _flagged(row) -> bool:
    if str(row["status"]).strip() in ("unknown", "refuted", "needs_subcluster"):
        return True
    if str(row["label"]).strip().lower().startswith(("unknown", "unassigned")):
        return True
    confidence = pd.to_numeric(row["confidence"], errors="coerce")
    return bool(pd.isna(confidence) or confidence < LOW_CONFIDENCE)


def _cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        value = f"{value:.3g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(_cell, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def render_level(
    level: str,
    labels: pd.DataFrame,
    *,
    evidence: Mapping[str, pd.DataFrame] | None = None,
    sizes: Mapping[str, int] | None = None,
    figures: Sequence[str] = (),
    max_rows: int = 8,
) -> str:
    """Render a level's report: an overview, the open problems, then one evidence card per cluster.

    Parameters
    ----------
    level
        Level name, e.g. ``"L2 within Neuron"``.
    labels
        The level's labels table.
    evidence
        Evidence tables by title, each tidy with a ``cluster`` column, already sorted and filtered
        to what a reader should see (e.g. top filtered markers, the curated panel's detection,
        negative-marker violations, the cross-check's top labels, QC deviation, the confidence proxy).
    sizes
        Cells per cluster.
    figures
        Paths of the level's figures, relative to the report.
    max_rows
        Rows shown per evidence table per cluster.

    Returns
    -------
    The report as Markdown, flagged clusters first.
    """
    evidence = dict(evidence or {})
    sizes = {str(k): int(v) for k, v in (sizes or {}).items()}
    labels = labels.assign(cluster=labels["cluster"].astype(str))
    flagged = labels.apply(_flagged, axis=1)
    problems = check_level(labels)

    lines = [f"# Level {level}", ""]
    n_cells = sum(sizes.get(c, 0) for c in labels["cluster"])
    lines.append(
        f"{len(labels)} clusters, {labels['label'].nunique()} labels"
        + (f", {n_cells:,} cells" if n_cells else "")
        + f", {int(flagged.sum())} flagged for review."
    )
    lines += ["", "## Completeness", ""]
    lines += (
        [f"- {p}" for p in problems] if problems else ["Complete: every cluster has a label, status and rationale."]
    )

    overview = (
        labels.assign(
            cells=labels["cluster"].map(sizes).fillna(0).astype(int),
            confidence=pd.to_numeric(labels["confidence"], errors="coerce"),
            flagged=flagged,
        )
        .groupby("label", sort=False)
        .agg(
            clusters=("cluster", ", ".join),
            cells=("cells", "sum"),
            confidence=("confidence", "mean"),
            flagged=("flagged", "sum"),
        )
        .sort_values(["cells", "label"], ascending=[False, True])
        .reset_index()
    )
    if not n_cells:
        overview = overview.drop(columns="cells")
    lines += ["", "## Labels", "", _table(overview.round({"confidence": 2}))]

    if figures:
        lines += ["", "## Figures", "", *(f"- [{f}]({f})" for f in figures)]

    lines += ["", "## Evidence cards"]
    for _, row in labels.assign(_flag=flagged).sort_values("_flag", ascending=False, kind="stable").iterrows():
        cluster = row["cluster"]
        call = [f"**{row['label']}**"]
        call += [f"`{row['cl_id']}`"] if not _blank(row["cl_id"]) else []
        call += [f"state {row['state']}"] if not _blank(row["state"]) else []
        call += [f"confidence {_cell(pd.to_numeric(row['confidence'], errors='coerce'))}", f"status {row['status']}"]
        call += [f"{sizes[cluster]:,} cells"] if cluster in sizes else []
        lines += ["", f"### Cluster {cluster}{' (flagged)' if row['_flag'] else ''}", "", " · ".join(call)]
        lines += ["", f"**Rationale.** {_cell(row['rationale']) or '_missing_'}"]
        if not _blank(row["rejected"]):
            lines += ["", f"**Rejected.** {_cell(row['rejected'])}"]
        for title, table in evidence.items():
            rows = table[table["cluster"].astype(str) == cluster].drop(columns="cluster").head(max_rows)
            if not rows.empty:
                lines += ["", f"**{title}**", "", _table(rows)]
    return "\n".join(lines) + "\n"
