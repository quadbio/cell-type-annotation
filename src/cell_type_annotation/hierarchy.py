"""A strictly nested label hierarchy, and marker scoring of each node against its siblings.

A hierarchy is an ordered list of ``adata.obs`` columns, coarse to fine, that nest strictly: every
label at level ``i+1`` occurs under exactly one label at level ``i``. :func:`build_hierarchy`
enforces that per cell, so it doubles as the nesting check before write-back.

:func:`hierarchy_markers` asks, at every node, which genes mark it **against its siblings within
its parent**. That is the question of a top-down annotation: ``FOXG1`` is a poor telencephalic
marker against every other class (telencephalic progenitors carry it too), but an excellent one
at the branch point where the question is telencephalic versus non-telencephalic.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

logger = logging.getLogger(__name__)

ChildOrder = Literal["size", "name", "category"]
Aggregate = Literal["leaf_min", "pooled"]


@dataclass(frozen=True)
class LabelHierarchy:
    """A strictly nested hierarchy of categorical labels, derived from ``adata.obs`` columns.

    Nodes are addressed by a ``(level, node)`` pair, where ``level`` is the name of the ``obs``
    column the node lives in, so the object is agnostic about level naming.

    Attributes
    ----------
    levels
        The ``obs`` column names that make up the hierarchy, coarse to fine.
    leaf_level
        The ``obs`` column whose categories define the leaves. Either the last entry of ``levels``
        or a further, finer column that is not itself drawn as a level.
    leaf_order
        Leaf labels in nested order: root classes in ``root_order``, then children ordered within
        each parent. This is the top-to-bottom row order of the figure.
    """

    levels: tuple[str, ...]
    leaf_level: str
    leaf_order: tuple[str, ...]
    _nodes: dict[str, tuple[str, ...]] = field(repr=False)
    _parent: dict[tuple[str, str], str | None] = field(repr=False)
    _children: dict[tuple[str, str], tuple[str, ...]] = field(repr=False)
    _leaves: dict[tuple[str, str], tuple[str, ...]] = field(repr=False)
    _counts: dict[tuple[str, str], int] = field(repr=False)
    n_cells_total: int = 0

    @property
    def all_levels(self) -> tuple[str, ...]:
        """All columns spanned, including ``leaf_level`` when it lies beyond ``levels``."""
        if self.leaf_level in self.levels:
            return self.levels
        return (*self.levels, self.leaf_level)

    def level_index(self, level: str) -> int:
        """Return the position of ``level`` within :attr:`all_levels`."""
        try:
            return self.all_levels.index(level)
        except ValueError:
            raise KeyError(f"'{level}' is not a level of this hierarchy: {list(self.all_levels)}") from None

    def nodes(self, level: str) -> list[str]:
        """Return the labels at ``level``, in nested order."""
        self.level_index(level)
        return list(self._nodes[level])

    def parent_of(self, level: str, node: str) -> str | None:
        """Return the parent label of ``node``, or ``None`` if ``level`` is the root level."""
        return self._parent[(level, node)]

    def children_of(self, level: str, node: str) -> list[str]:
        """Return the labels of ``node``'s children, in nested order. Empty at the finest level."""
        return list(self._children.get((level, node), ()))

    def leaves_of(self, level: str, node: str) -> list[str]:
        """Return the ``leaf_level`` labels descending from ``node``, in nested order."""
        return list(self._leaves[(level, node)])

    def siblings_of(self, level: str, node: str) -> list[str]:
        """Return the other labels sharing ``node``'s parent.

        At the root level every other root label is a sibling, since they all share the implicit
        root of the tree.
        """
        parent = self.parent_of(level, node)
        if parent is None:
            return [n for n in self._nodes[level] if n != node]
        parent_level = self.all_levels[self.level_index(level) - 1]
        return [n for n in self._children[(parent_level, parent)] if n != node]

    def n_cells(self, level: str, node: str) -> int:
        """Return the number of cells carrying ``node`` at ``level``."""
        return self._counts[(level, node)]

    def iter_nodes(self, *, levels: Sequence[str] | None = None):
        """Yield ``(level, node)`` pairs in level-then-nested order."""
        for level in levels if levels is not None else self.all_levels:
            for node in self._nodes[level]:
                yield level, node


def _ordered_categories(
    series: pd.Series, order: Literal["size", "name", "category"], keys: Sequence[str] | None = None
) -> list[str]:
    """Order the observed labels of ``series`` by size, name, or declared categorical order.

    When ``keys`` is given, labels are first ranked by the position of the first key they
    contain, and ``order`` only breaks ties. That is what makes an ordering follow an axis
    rather than a family: one key list orders ``Telencephalic NPC`` and ``Telencephalic Neuron``
    identically, however different their cell counts are.
    """
    counts = series.value_counts()
    counts = counts[counts > 0]  # a categorical also lists its unobserved categories
    if order == "size":
        base = list(counts.index)
    elif order == "name":
        base = sorted(counts.index)
    elif order == "category":
        base = (
            [c for c in series.cat.categories if c in set(counts.index)]
            if isinstance(series.dtype, pd.CategoricalDtype)
            else sorted(counts.index)
        )
    else:
        raise ValueError(f"Unknown order '{order}', expected one of 'size', 'name', 'category'")

    if not keys:
        return base

    def rank(label: str) -> int:
        return next((i for i, key in enumerate(keys) if key in label), len(keys))

    return sorted(base, key=rank)  # stable, so `base` breaks ties within a key


def build_hierarchy(
    adata: AnnData,
    levels: Sequence[str],
    *,
    leaf_level: str | None = None,
    root_order: Sequence[str] | None = None,
    child_order: ChildOrder = "size",
    level_order: Mapping[str, Sequence[str]] | None = None,
    dropna: bool = True,
) -> LabelHierarchy:
    """Derive a strictly nested label hierarchy from categorical ``obs`` columns.

    Parameters
    ----------
    adata
        AnnData object whose ``obs`` holds the label columns
    levels
        Names of the ``obs`` columns forming the hierarchy, coarse to fine
    leaf_level
        Name of the ``obs`` column defining the leaves. Defaults to the last entry of ``levels``.
        Pass a finer column to aggregate against leaves that are not themselves drawn as a level
    root_order
        Explicit ordering of the labels at ``levels[0]``. Defaults to the declared categorical
        order if that column is categorical, otherwise to descending cell count
    child_order
        How to order the children within each parent: ``'size'`` (descending cell count),
        ``'name'`` (alphabetical), or ``'category'`` (declared categorical order)
    level_order
        Per-level ordering keys, ``{level: [key, ...]}``, applied within *every* parent at that
        level so the ordering follows an axis rather than each family's own cell counts. A label
        is ranked by the position of the first key it contains, labels matching no key sort last,
        and ``child_order`` breaks ties. ``root_order`` still wins at the root level when given
    dropna
        Whether to drop cells with a missing label in any level. If ``False``, a missing label
        raises instead

    Returns
    -------
    LabelHierarchy
        The derived hierarchy.

    Raises
    ------
    ValueError
        If ``levels`` is empty, a column is missing from ``obs``, the labels are not strictly
        nested, or ``dropna`` is ``False`` and labels are missing.

    Examples
    --------
    >>> hierarchy = build_hierarchy(adata, levels=["annot_L1_v1", "annot_L2_v1", "annot_L3_v1"])
    >>> hierarchy.siblings_of("annot_L2_v1", "Telencephalic NPC")
    ['Non-Telen. NPC']
    """
    levels = list(levels)
    if not levels:
        raise ValueError("`levels` must name at least one obs column")

    leaf_level = levels[-1] if leaf_level is None else leaf_level
    all_levels = levels if leaf_level in levels else [*levels, leaf_level]

    missing = [c for c in all_levels if c not in adata.obs]
    if missing:
        raise ValueError(f"Columns not found in `adata.obs`: {missing}")

    # Labels as strings, keeping a categorical's declared order for `child_order="category"`.
    # `.astype(str)` turns NaN into the literal 'nan', so find the missing values beforehand.
    obs = pd.DataFrame(
        {
            level: adata.obs[level].cat.rename_categories(str)
            if isinstance(adata.obs[level].dtype, pd.CategoricalDtype)
            else adata.obs[level].astype(str)
            for level in all_levels
        },
        index=adata.obs_names,
    )
    na = adata.obs[all_levels].isna().any(axis=1)
    if na.any():
        if not dropna:
            raise ValueError(f"{int(na.sum())} cells have a missing label; pass `dropna=True` to drop them")
        logger.info("Dropping %s cells with a missing label in %s", f"{int(na.sum()):,}", all_levels)
        obs = obs.loc[~na]

    if obs.empty:
        raise ValueError("No cells left after dropping missing labels")

    # --- strict nesting: every child label must have exactly one parent ---
    for parent_col, child_col in zip(all_levels[:-1], all_levels[1:], strict=False):
        n_parents = obs.groupby(child_col, observed=True)[parent_col].nunique()
        violations = n_parents[n_parents > 1]
        if not violations.empty:
            detail = ", ".join(f"{label!r} ({n} parents)" for label, n in violations.items())
            raise ValueError(
                f"Labels in '{child_col}' do not nest strictly within '{parent_col}': {detail}. "
                "Every child label must occur under exactly one parent label."
            )

    counts = {level: obs[level].value_counts() for level in all_levels}

    parent: dict[tuple[str, str], str | None] = {}
    for parent_col, child_col in zip(all_levels[:-1], all_levels[1:], strict=False):
        mapping = obs.groupby(child_col, observed=True)[parent_col].first()
        for child, par in mapping.items():
            parent[(child_col, child)] = par

    # --- nested ordering: root order first, then children ordered within each parent ---
    level_order = dict(level_order or {})
    unknown_levels = [level for level in level_order if level not in all_levels]
    if unknown_levels:
        raise ValueError(f"`level_order` names levels not in the hierarchy: {unknown_levels}")
    for level, keys in level_order.items():
        unused = [k for k in keys if not any(k in label for label in obs[level].unique())]
        if unused:
            logger.warning("`level_order['%s']` keys match no label and are ignored: %s", level, unused)

    root_level = all_levels[0]
    if root_order is None:
        default_order = "category" if isinstance(adata.obs[root_level].dtype, pd.CategoricalDtype) else "size"
        ordered_roots = _ordered_categories(obs[root_level], default_order, level_order.get(root_level))
    else:
        observed = set(obs[root_level].unique())
        unknown = [r for r in root_order if r not in observed]
        if unknown:
            raise ValueError(f"`root_order` names labels absent from '{root_level}': {unknown}")
        ordered_roots = [r for r in root_order if r in observed]
        ordered_roots += [r for r in _ordered_categories(obs[root_level], "size") if r not in set(ordered_roots)]

    nodes: dict[str, list[str]] = {root_level: ordered_roots}
    children: dict[tuple[str, str], tuple[str, ...]] = {}
    for parent_col, child_col in zip(all_levels[:-1], all_levels[1:], strict=False):
        ordered_children: list[str] = []
        for par in nodes[parent_col]:
            sub = obs.loc[obs[parent_col] == par, child_col]
            kids = _ordered_categories(sub, child_order, level_order.get(child_col))
            children[(parent_col, par)] = tuple(kids)
            ordered_children.extend(kids)
        nodes[child_col] = ordered_children
    for node in nodes[all_levels[-1]]:
        children[(all_levels[-1], node)] = ()
    for node in nodes[root_level]:
        parent[(root_level, node)] = None

    # --- leaves per node, in the nested leaf order ---
    leaf_order = tuple(nodes[leaf_level])
    leaf_rank = {leaf: i for i, leaf in enumerate(leaf_order)}
    leaves: dict[tuple[str, str], tuple[str, ...]] = {}
    for level in all_levels:
        grouped = obs.groupby(level, observed=True)[leaf_level].unique()
        for node, node_leaves in grouped.items():
            leaves[(level, node)] = tuple(sorted(node_leaves, key=lambda leaf: leaf_rank[leaf]))

    logger.info(
        "Hierarchy over %s: %s classes, %s cells",
        all_levels,
        [len(nodes[level]) for level in all_levels],
        f"{len(obs):,}",
    )

    return LabelHierarchy(
        levels=tuple(levels),
        leaf_level=leaf_level,
        leaf_order=leaf_order,
        _nodes={level: tuple(nodes[level]) for level in all_levels},
        _parent=parent,
        _children=children,
        _leaves=leaves,
        _counts={(level, node): int(counts[level][node]) for level in all_levels for node in nodes[level]},
        n_cells_total=int(len(obs)),
    )


def hierarchy_row_positions(hierarchy: LabelHierarchy) -> dict[tuple[str, str], float]:
    """Return the vertical position of every node, in leaf-row units.

    Leaves sit at their index in :attr:`LabelHierarchy.leaf_order`; every internal node sits at the
    mean row of its own leaves, so a node is centred on the block of rows it owns.

    Parameters
    ----------
    hierarchy
        The hierarchy to lay out

    Returns
    -------
    dict
        Mapping of ``(level, node)`` to a row position.
    """
    rank = {leaf: float(i) for i, leaf in enumerate(hierarchy.leaf_order)}
    return {
        (level, node): float(np.mean([rank[leaf] for leaf in hierarchy.leaves_of(level, node)]))
        for level, node in hierarchy.iter_nodes()
    }


def _detection_counts(
    adata: AnnData,
    leaf_labels: pd.Series,
    leaf_order: Sequence[str],
    genes: Sequence[str],
    layer: str | None,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Count, per leaf and gene, how many cells reach ``threshold``.

    Returns the ``n_leaves x n_genes`` hit counts and the per-leaf cell counts. Sparse input is
    kept sparse: the full matrix is never densified.
    """
    matrix = adata.layers[layer] if layer is not None else adata.X
    gene_idx = adata.var_names.get_indexer(genes)
    if (gene_idx < 0).any():
        missing = [g for g, i in zip(genes, gene_idx, strict=False) if i < 0]
        raise ValueError(f"Genes not found in `adata.var_names`: {missing}")
    matrix = matrix[:, gene_idx]

    if sp.issparse(matrix) and threshold <= 0:
        raise ValueError(
            f"`threshold={threshold}` on a sparse matrix would mark every zero as detected. Use a threshold above 0."
        )

    hits = np.zeros((len(leaf_order), len(genes)), dtype=np.int64)
    n_cells = np.zeros(len(leaf_order), dtype=np.int64)
    labels = leaf_labels.to_numpy()
    for i, leaf in enumerate(leaf_order):
        mask = labels == leaf
        n_cells[i] = int(mask.sum())
        if n_cells[i] == 0:
            continue
        detected = matrix[mask] >= threshold
        hits[i] = np.asarray(detected.sum(axis=0)).ravel()
    return hits, n_cells


def _detection_rate(hits: np.ndarray, n_cells: np.ndarray, rows: Sequence[int], how: Aggregate) -> np.ndarray:
    """Aggregate per-leaf hit counts into a per-gene detection percentage."""
    rows = list(rows)
    if how == "pooled":
        return hits[rows].sum(axis=0) / n_cells[rows].sum() * 100.0
    return hits[rows] / n_cells[rows][:, None] * 100.0


def hierarchy_markers(
    adata: AnnData,
    levels: Sequence[str],
    *,
    leaf_level: str | None = None,
    genes: Sequence[str] | None = None,
    include_genes: Sequence[str] | None = None,
    layer: str | None = None,
    threshold: float = 1.0,
    aggregate: Aggregate = "pooled",
    n_top: int = 3,
    hierarchy: LabelHierarchy | None = None,
) -> pd.DataFrame:
    """Score marker genes for every node of a label hierarchy, against that node's siblings.

    The score is the *detection margin*: how much more widely a gene is detected inside the node
    than in its siblings. It is a fraction-of-cells statistic, so it does not depend on how many
    cells a node has, which makes a 378-cell node comparable to a 336,000-cell one.

    With the default ``aggregate='pooled'``, both sides pool their cells:

    .. code-block:: text

        det_in (node, gene) = % of the node's cells detecting the gene
        det_out(node, gene) = % of all its siblings' cells detecting the gene
        score               = det_in - det_out

    Pooling the siblings lets a small sibling be outweighed by a large one, so a gene the node
    shares with one close sibling can still score well. ``aggregate='leaf_min'`` is the strict
    alternative: ``det_in`` is the minimum over the node's leaves and ``det_out`` the maximum over
    its siblings' leaves, so a gene only scores well if it holds for *every* child of the node and
    beats *every* competitor.

    Nodes without siblings are skipped: there is nothing to be specific against.

    Parameters
    ----------
    adata
        AnnData object holding the expression matrix and the label columns
    levels
        Names of the ``obs`` columns forming the hierarchy, coarse to fine
    leaf_level
        ``obs`` column defining the leaves. Defaults to the last entry of ``levels``
    genes
        Restrict scoring to these genes. Use this to score a curated marker panel with the same
        specificity metric instead of ranking the whole matrix
    include_genes
        Always emit rows for these genes, in addition to the top ``n_top`` per node. Use this when
        a downstream display names a gene that would not otherwise make the cut
    layer
        Layer to read expression from. ``None`` reads ``adata.X``
    threshold
        A cell counts as detecting a gene when its value is ``>= threshold`` in the chosen matrix.
        On a log1p-normalised ``X`` a threshold of ``1.0`` is not "one transcript", it is
        normalised expression of 1.0. Pass ``layer='counts'`` for a genuine transcript count
    aggregate
        ``'pooled'`` to pool cells on both sides, ``'leaf_min'`` for the strict min/max over leaves
    n_top
        Number of top-scoring genes to report per node
    hierarchy
        A pre-built hierarchy, to avoid re-deriving it. Built from ``adata`` when omitted

    Returns
    -------
    pd.DataFrame
        Tidy frame with one row per (node, gene): ``level``, ``node``, ``gene``, ``score``,
        ``det_in``, ``det_out``, ``rank``.

    Examples
    --------
    >>> markers = hierarchy_markers(adata, levels=["annot_L1_v1", "annot_L2_v1"])
    >>> markers = hierarchy_markers(adata, levels=["annot_L1_v1"], genes=["FOXG1", "PAX6", "DLX2"])
    """
    if aggregate not in ("leaf_min", "pooled"):
        raise ValueError(f"Unknown aggregate '{aggregate}', expected 'leaf_min' or 'pooled'")

    if hierarchy is None:
        hierarchy = build_hierarchy(adata, levels, leaf_level=leaf_level)

    gene_names = list(adata.var_names) if genes is None else list(dict.fromkeys(genes))
    gene_names += [g for g in (include_genes or []) if g not in set(gene_names)]
    if not gene_names:
        raise ValueError("No genes to score")

    leaf_order = list(hierarchy.leaf_order)
    # Only the cells the hierarchy kept: a cell missing a label at any level was dropped from it.
    kept = adata.obs[list(hierarchy.all_levels)].notna().all(axis=1)
    leaf_labels = adata.obs[hierarchy.leaf_level].astype(str).where(kept)
    hits, n_cells = _detection_counts(adata, leaf_labels, leaf_order, gene_names, layer, threshold)
    leaf_pos = {leaf: i for i, leaf in enumerate(leaf_order)}

    gene_array = np.asarray(gene_names)
    forced = set(include_genes or [])
    rows: list[dict] = []
    for level, node in hierarchy.iter_nodes():
        siblings = hierarchy.siblings_of(level, node)
        if not siblings:
            continue
        in_rows = [leaf_pos[leaf] for leaf in hierarchy.leaves_of(level, node)]
        out_rows = [leaf_pos[leaf] for sib in siblings for leaf in hierarchy.leaves_of(level, sib)]
        if not in_rows or not out_rows:
            continue

        if aggregate == "leaf_min":
            det_in = _detection_rate(hits, n_cells, in_rows, "leaf_min").min(axis=0)
            det_out = _detection_rate(hits, n_cells, out_rows, "leaf_min").max(axis=0)
        else:
            det_in = _detection_rate(hits, n_cells, in_rows, "pooled")
            det_out = _detection_rate(hits, n_cells, out_rows, "pooled")
        score = det_in - det_out

        order = np.argsort(-score, kind="stable")
        keep = list(order[:n_top])
        if forced:
            rank_of = {int(idx): int(pos) for pos, idx in enumerate(order)}
            keep += [int(i) for i in order if gene_array[i] in forced and int(i) not in set(keep)]
        else:
            rank_of = {int(idx): int(pos) for pos, idx in enumerate(order[:n_top])}

        for idx in keep:
            rows.append(
                {
                    "level": level,
                    "node": node,
                    "gene": gene_array[idx],
                    "score": round(float(score[idx]), 1),
                    "det_in": round(float(det_in[idx]), 1),
                    "det_out": round(float(det_out[idx]), 1),
                    "rank": rank_of[int(idx)] + 1,
                }
            )

    result = pd.DataFrame(rows, columns=["level", "node", "gene", "score", "det_in", "det_out", "rank"])
    logger.info("Scored %s genes over %s nodes", f"{len(gene_names):,}", result["node"].nunique())
    return result
