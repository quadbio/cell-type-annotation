"""The signed, sourced marker table, and the rule for which markers a figure shows.

One table, ``results/markers.csv``, holds every curated marker of the run: one row per
``(level, node, gene)`` with its ``sign`` (``pos``, or ``neg`` for a gene whose presence argues
against the type) and its ``source`` and ``pmid``. The assumptions ledger signs it off level by
level, and it is the whitelist from which :func:`display_markers` picks what a figure shows.
"""

from collections.abc import Iterable, Mapping
from os import PathLike

import numpy as np
import pandas as pd

from cell_type_annotation.hierarchy import LabelHierarchy

MARKER_COLUMNS = ("level", "node", "gene", "sign", "source", "pmid", "note")
REQUIRED_COLUMNS = ("level", "node", "gene", "sign")

#: Labels that mean "no identity", drawn with the ∅ flag rather than given markers.
UNKNOWN_LABELS = frozenset({"unknown", "unassigned"})

#: What each flag symbol means, for the figure's key.
FLAGS = {
    "†": "canonical markers off-panel",
    "−": "no canonical marker separates it from its siblings",
    "∅": "unassigned",
}


def read_markers(source: str | PathLike | pd.DataFrame) -> pd.DataFrame:
    """Read and validate the signed marker table.

    Parameters
    ----------
    source
        Path to the CSV, or the frame itself.

    Returns
    -------
    The table with all of :data:`MARKER_COLUMNS` (missing optional ones filled with ``""``),
    ``sign`` lower-cased, plus any extra columns.

    Raises
    ------
    ValueError
        If a required column is missing, a sign is not ``pos``/``neg``, or a gene is both a
        positive and a negative marker of the same node.
    """
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Marker table is missing columns {missing}; expected {list(MARKER_COLUMNS)}")
    for column in MARKER_COLUMNS:
        df[column] = df[column].fillna("").astype(str).str.strip() if column in df.columns else ""
    df["sign"] = df["sign"].str.lower()
    bad = sorted(set(df["sign"]) - {"pos", "neg"})
    if bad:
        raise ValueError(f"`sign` must be 'pos' or 'neg', got {bad}")
    both = df.groupby(["level", "node", "gene"])["sign"].nunique()
    both = both[both > 1]
    if not both.empty:
        raise ValueError(f"Genes listed as both positive and negative for one node: {list(both.index)}")
    return df[[*MARKER_COLUMNS, *(c for c in df.columns if c not in MARKER_COLUMNS)]]


def _is_unknown(label: str) -> bool:
    return label.strip().lower() in UNKNOWN_LABELS or label.strip().lower().startswith("unknown")


def display_markers(
    scores: pd.DataFrame,
    hierarchy: LabelHierarchy,
    markers: pd.DataFrame,
    var_names: Iterable[str],
    *,
    level_keys: Mapping[str, str] | None = None,
    min_margin: float = 10.0,
    n: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick the canonical markers a hierarchy figure shows, and flag the nodes that have none.

    A node shows a gene when it is one of the node's curated positive markers, its detection
    margin over the pooled siblings is at least ``min_margin`` points, and no single sibling
    detects it more widely. The best ``n`` by margin are shown. A node where none passes is
    flagged, never given a data-driven substitute:

    - ``∅``: an unknown or unassigned node;
    - ``†``: its canonical markers are all off the panel;
    - ``−``: no measured canonical marker separates it from its siblings (or none was curated).

    A ``−`` flag on a confident call is evidence for the critics: the markers the call was signed
    off on do not separate it from its siblings in this data.

    Parameters
    ----------
    scores
        :func:`~cell_type_annotation.hierarchy.hierarchy_markers` output, run with
        ``include_genes`` set to every measured positive marker, so each one has a row on every node.
    hierarchy
        The hierarchy the scores were computed on.
    markers
        Signed marker table (:func:`read_markers`).
    var_names
        The genes measured.
    level_keys
        Maps the table's ``level`` values to the hierarchy's ``obs`` columns, e.g.
        ``{"L1": "annot_L1_v1"}``. Defaults to identity.
    min_margin
        Minimum ``det_in - det_out``, in percentage points.
    n
        Markers shown per node.

    Returns
    -------
    ``(shown, flags)``. ``shown`` has one row per displayed gene (``level, node, gene, score,
    det_in, det_out, det_best_sibling, rank, source, pmid``) and is what
    :func:`~cell_type_annotation.pl.annotation_hierarchy` takes as ``markers``. ``flags`` has one
    row per flagged node: ``level, node, symbol, reason``.
    """
    level_keys = dict(level_keys or {})
    present = set(map(str, var_names))
    positives = markers[markers["sign"] == "pos"].assign(level=lambda d: d["level"].map(lambda v: level_keys.get(v, v)))
    det_in = scores.set_index(["level", "node", "gene"])["det_in"]

    shown, flags = [], []
    for level, node in hierarchy.iter_nodes(levels=hierarchy.levels):
        siblings = hierarchy.siblings_of(level, node)
        if not siblings:
            continue
        canonical = positives[(positives["level"] == level) & (positives["node"] == node)]
        measured = canonical[canonical["gene"].isin(present)]
        if _is_unknown(node):
            flags.append((level, node, "∅", "unassigned"))
            continue
        if canonical.empty:
            flags.append((level, node, "−", "no canonical marker curated"))
            continue
        if measured.empty:
            flags.append((level, node, "†", f"off-panel: {', '.join(canonical['gene'])}"))
            continue

        candidates = []
        for gene, source, pmid in measured[["gene", "source", "pmid"]].itertuples(index=False):
            hit = scores[(scores["level"] == level) & (scores["node"] == node) & (scores["gene"] == gene)]
            if hit.empty:
                raise ValueError(
                    f"No score for {gene!r} on {level} / {node}; pass it to hierarchy_markers' include_genes"
                )
            best_sibling = max(det_in.get((level, s, gene), np.nan) for s in siblings)
            row = hit.iloc[0]
            if row["score"] >= min_margin and row["det_in"] > best_sibling:
                candidates.append((row["score"], gene, row["det_in"], row["det_out"], best_sibling, source, pmid))
        if not candidates:
            flags.append((level, node, "−", f"measured but not separating: {', '.join(measured['gene'])}"))
            continue
        for rank, (score, gene, d_in, d_out, best, source, pmid) in enumerate(sorted(candidates, reverse=True)[:n], 1):
            shown.append((level, node, gene, score, d_in, d_out, best, rank, source, pmid))

    shown_cols = ["level", "node", "gene", "score", "det_in", "det_out", "det_best_sibling", "rank", "source", "pmid"]
    return pd.DataFrame(shown, columns=shown_cols), pd.DataFrame(flags, columns=["level", "node", "symbol", "reason"])
