"""Checks on the data and the evidence that no scanpy or decoupler function makes."""

import io
import re
from collections.abc import Iterable, Mapping
from importlib.resources import files

import numpy as np
import pandas as pd
import scipy.sparse as sp

#: Below this many on-panel positive markers a candidate type is uncallable on the panel: it
#: contributes to ``Unknown``, it does not score zero.
MIN_ON_PANEL_MARKERS = 2

#: A negative marker detected in more of a cluster's cells than this argues against the call.
#: Calibrated on five atlases (immune, GTEx, mouse pancreas, HypoMap, HNOCA): genuine positives
#: reach 50-68% of their type's cells, the background of a truly absent marker 10-20% in the
#: noisiest data. 0.20 flags no genuine positive and few true negatives; raise toward 0.25 for
#: continuous developmental data.
NEG_MARKER_MAX_PCT = 0.20

#: Cell Ontology (CL) and Provisional Cell Ontology (PCL) accessions.
CL_ID_PATTERN = re.compile(r"^(?:CL|PCL):\d{7}$")

#: ``(min top-1 log2FC, min markers with log2FC >= strong, suggested confidence)``, first match wins.
CONFIDENCE_BANDS = ((3.0, 5, 0.9), (2.0, 3, 0.8), (1.0, 1, 0.7), (0.5, 0, 0.6))
CONFIDENCE_BASE = 0.45


def normalization_fingerprint(X, *, n_cells: int = 2000, seed: int = 0) -> dict:
    """Tell raw counts, proper log-normalization and log1p of un-normalized counts apart.

    A ``max > 50`` heuristic passes the most dangerous case: ``log1p`` of counts that were never
    size-normalized reads as log1p, but its markers track depth and cell size, not identity. Two
    cheap probes on a sample of cells separate the cases:

    - integer-valued entries mean counts; log1p output is essentially never integer;
    - ``expm1(X).sum(1)`` recovers each cell's pre-log total, whose coefficient of variation is
      ~0 when every cell was normalized to the same total before ``log1p``.

    The CV test recognizes ``normalize_total``-style data. Size factors (scran), sctransform, or an
    ``X`` subset to highly variable genes after normalization read as un-normalized: confirm those
    with the user rather than stopping.

    Parameters
    ----------
    X
        Dense or sparse cells x genes matrix, as declared log-normalized.
    n_cells
        Cells to sample.
    seed
        Seed for the sample.

    Returns
    -------
    ``{"verdict", "ok", "integer_valued", "expm1_sum_cv", "min", "max"}``. ``ok`` is true only for
    size-normalized log1p data.
    """
    rows = np.random.default_rng(seed).choice(X.shape[0], size=min(n_cells, X.shape[0]), replace=False)
    sample = X[np.sort(rows)]
    if sp.issparse(sample):  # stays sparse: densifying 2,000 cells x 30,000 genes costs ~0.5 GB
        sample = sp.csr_matrix(sample, dtype=float)
        stored = sample.data
        exp = sample.copy()
        exp.data = np.expm1(exp.data)
        totals = np.asarray(exp.sum(axis=1)).ravel()
    else:
        sample = np.asarray(sample, dtype=float)
        stored = sample.ravel()
        totals = np.expm1(sample).sum(axis=1)
    finite = bool(np.isfinite(stored).all())
    nonzero = stored[(stored != 0) & np.isfinite(stored)]
    xmin, xmax = (float(np.nanmin(stored)), float(np.nanmax(stored))) if stored.size else (0.0, 0.0)
    if sp.issparse(sample) and sample.nnz < np.prod(sample.shape):
        xmin, xmax = min(xmin, 0.0), max(xmax, 0.0)

    integer_valued = bool(nonzero.size and np.allclose(nonzero, np.round(nonzero)))
    totals = totals[np.isfinite(totals) & (totals > 0)]
    cv = float(totals.std() / totals.mean()) if totals.size else float("nan")

    if not finite:
        verdict = "non-finite values (NaN or inf, e.g. genes missing from part of a joined object)"
    elif xmin < 0:
        verdict = "scaled (z-scored): not usable as log-normalized data"
    elif integer_valued:
        verdict = "raw counts (integer-valued): not log-normalized"
    elif np.isfinite(cv) and cv < 0.05:
        verdict = "size-normalized + log1p"
    elif np.isfinite(cv):
        verdict = "log1p of un-normalized counts? per-cell totals vary, so markers would track depth"
    else:
        verdict = "indeterminate"
    return {
        "verdict": verdict,
        "ok": verdict == "size-normalized + log1p",
        "integer_valued": integer_valued,
        "expm1_sum_cv": round(cv, 4) if np.isfinite(cv) else None,
        "min": round(xmin, 3),
        "max": round(xmax, 3),
    }


def panel_gate(
    markers: pd.DataFrame, var_names: Iterable[str], *, min_on_panel: int = MIN_ON_PANEL_MARKERS
) -> pd.DataFrame:
    """Split each candidate's markers into measured and off-panel.

    An off-panel gene is unmeasured, never evidence of absence. A candidate with fewer than
    ``min_on_panel`` measured positive markers is uncallable on this panel.

    Parameters
    ----------
    markers
        Signed marker table as returned by :func:`~cell_type_annotation.markers.read_markers`.
    var_names
        The genes measured.
    min_on_panel
        Positive markers a candidate needs on the panel to be callable.

    Returns
    -------
    One row per ``(level, node)``: ``n_pos``, ``n_pos_on_panel``, ``uncallable``, ``off_panel_pos``,
    ``off_panel_neg`` (semicolon-joined).
    """
    present = set(map(str, var_names))
    rows = []
    for (level, node), group in markers.groupby(["level", "node"], sort=False):
        pos = group.loc[group["sign"] == "pos", "gene"]
        neg = group.loc[group["sign"] == "neg", "gene"]
        n_on = int(pos.isin(present).sum())
        rows.append(
            {
                "level": level,
                "node": node,
                "n_pos": len(pos),
                "n_pos_on_panel": n_on,
                "uncallable": n_on < min_on_panel,
                "off_panel_pos": ";".join(pos[~pos.isin(present)]),
                "off_panel_neg": ";".join(neg[~neg.isin(present)]),
            }
        )
    columns = ["level", "node", "n_pos", "n_pos_on_panel", "uncallable", "off_panel_pos", "off_panel_neg"]
    return pd.DataFrame(rows, columns=columns)


def negative_marker_flags(
    pct: pd.DataFrame, markers: pd.DataFrame, *, max_pct: float = NEG_MARKER_MAX_PCT
) -> pd.DataFrame:
    """Flag clusters that express a candidate's negative markers.

    A gene is routinely a positive marker of one type and a negative marker of another, so the
    check runs per ``(candidate, gene)`` pair. Off-panel negatives are skipped: unmeasured is not
    confirmed absent.

    Parameters
    ----------
    pct
        Fraction of cells detecting each gene, clusters x genes, e.g. from
        ``sc.get.aggregate(adata, by=cluster_key, func="count_nonzero")`` divided by cluster sizes.
    markers
        Signed marker table.
    max_pct
        Detection above which a negative marker counts as expressed.

    Returns
    -------
    One row per ``(cluster, level, node, gene)``: ``pct`` and ``violation``. ``level`` keeps a
    label reused at two levels apart.
    """
    negatives = markers.loc[(markers["sign"] == "neg") & markers["gene"].isin(pct.columns), ["level", "node", "gene"]]
    if pct[negatives["gene"].unique()].isna().any().any():
        raise ValueError("`pct` has missing detection values: every NaN would read as 'not expressed'")
    rows = [
        {
            "cluster": str(cluster),
            "level": level,
            "node": node,
            "gene": gene,
            "pct": round(float(pct.at[cluster, gene]), 4),
            "violation": bool(pct.at[cluster, gene] > max_pct),
        }
        for cluster in pct.index
        for level, node, gene in negatives.itertuples(index=False)
    ]
    return pd.DataFrame(rows, columns=["cluster", "level", "node", "gene", "pct", "violation"])


def validate_cl_ids(cl_ids: Mapping[str, object] | pd.Series) -> list[str]:
    """Check ontology ids for syntax: ``CL:0000000`` or ``PCL:0000000``; blanks are allowed.

    Syntax only. Whether the term exists, is current and matches its label is checked through the
    knowledgebase MCP (``bc_get_term_details``).

    Parameters
    ----------
    cl_ids
        Ontology id per cluster.

    Returns
    -------
    One message per malformed id.
    """
    problems = []
    for cluster, value in dict(cl_ids).items():
        cl_id = "" if pd.isna(value) else str(value).strip()
        if cl_id and not CL_ID_PATTERN.match(cl_id):
            problems.append(f"cluster {cluster!r}: cl_id {cl_id!r} is not of the form CL:0000000 or PCL:0000000")
    return problems


def qc_deviation(medians: pd.DataFrame) -> pd.DataFrame:
    """Robust z of each cluster's QC medians against the spread across clusters.

    ``(median - median over clusters) / (1.4826 * MAD)``, per column, so a cluster that is off
    the population on depth, mitochondrial fraction or a cell-cycle score stands out whatever the
    metric. Where most clusters share one value the MAD is 0, and the scale falls back to
    ``1.2533 * mean absolute deviation`` so a lone outlier still shows. Columns without any spread
    give 0; a missing median stays missing. Meaningful from three clusters on.

    Parameters
    ----------
    medians
        Clusters x numeric QC metrics, e.g. ``obs.groupby(cluster_key)[metrics].median()``.

    Returns
    -------
    Clusters x metrics robust z, rounded to 2 decimals.
    """
    center = medians.median()
    deviation = (medians - center).abs()
    scale = (deviation.median() * 1.4826).where(lambda s: s > 0, deviation.mean() * 1.2533)
    spread = scale > 0
    z = (medians - center).divide(scale.where(spread))
    z.loc[:, ~spread] = 0.0
    return z.mask(medians.isna()).round(2)  # missing evidence stays missing, never "normal"


def confidence_proxy(
    filtered_markers: pd.DataFrame,
    clusters: Iterable[str],
    *,
    agree: Mapping[str, float] | None = None,
    strong: float = 1.0,
) -> pd.DataFrame:
    """A per-cluster confidence band from marker specificity, to calibrate hand-set confidences.

    Advisory: a high confidence on a cluster with weak markers is the thing to re-examine.

    Parameters
    ----------
    filtered_markers
        Markers surviving ``sc.tl.filter_rank_genes_groups``, e.g. from
        ``sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_groups_filtered")``, with
        columns ``group`` (or ``cluster``) and ``logfoldchanges``.
    clusters
        All clusters, so a cluster without a surviving marker gets the lowest band rather than no row.
    agree
        Optional number of cross-check methods agreeing per cluster; each above one adds 0.05.
    strong
        log2 fold-change at which a marker counts as strong.

    Returns
    -------
    One row per cluster: ``lfc_top1``, ``lfc_margin``, ``n_strong``, ``suggested_conf``.
    """
    agree = {str(k): v for k, v in dict(agree if agree is not None else {}).items()}
    key = "group" if "group" in filtered_markers.columns else "cluster"
    # filter_rank_genes_groups blanks the names of rejected genes rather than dropping their rows
    kept = filtered_markers.dropna(subset=["names"]) if "names" in filtered_markers.columns else filtered_markers
    lfc_by_cluster = {
        str(cluster): sorted(pd.to_numeric(g["logfoldchanges"], errors="coerce").dropna(), reverse=True)
        for cluster, g in kept.groupby(key, observed=True)
    }
    rows = []
    for cluster in map(str, clusters):
        lfc = lfc_by_cluster.get(cluster, [])
        top1 = lfc[0] if lfc else 0.0
        top2 = lfc[1] if len(lfc) > 1 else 0.0
        n_strong = sum(v >= strong for v in lfc)
        conf = next((c for t, n, c in CONFIDENCE_BANDS if lfc and top1 >= t and n_strong >= n), CONFIDENCE_BASE)
        if pd.notna(agree.get(cluster)):
            conf = min(0.95, conf + 0.05 * (agree[cluster] - 1))
        rows.append(
            {
                "cluster": cluster,
                "lfc_top1": round(top1, 3),
                "lfc_margin": round(top1 - top2, 3),
                "n_strong": n_strong,
                "suggested_conf": round(conf, 2),
            }
        )
    return pd.DataFrame(rows)


def cell_cycle_genes(var_names: Iterable[str] | None = None) -> tuple[list[str], list[str]]:
    """The Tirosh et al. (2016) S and G2/M gene lists, for ``sc.tl.score_genes_cell_cycle``.

    Human symbols, matched case-insensitively so the same lists resolve mouse genes. A few legacy
    symbols (MLF1IP, FAM64A, HN1) go unmatched on objects using their current names.

    Parameters
    ----------
    var_names
        If given, return only the genes present, spelled as in ``var_names``.

    Returns
    -------
    ``(s_genes, g2m_genes)``.
    """
    phases = pd.read_csv(
        io.StringIO(files("cell_type_annotation").joinpath("data/cell_cycle_tirosh.tsv").read_text()), sep="\t"
    )
    s, g2m = (phases.loc[phases["phase"] == p, "gene"].tolist() for p in ("S", "G2M"))
    if var_names is None:
        return s, g2m
    lookup = {str(v).upper(): str(v) for v in var_names}
    return [lookup[g] for g in s if g in lookup], [lookup[g] for g in g2m if g in lookup]
