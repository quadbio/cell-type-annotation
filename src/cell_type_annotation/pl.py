"""The annotation-hierarchy plot: nested label columns, connecting ribbons, and per-node markers.

The plot is drawn into one :class:`~matplotlib.axes.Axes` whose x unit is the typographic point and
whose y unit is one leaf row. Column widths are *measured* from rendered glyphs, which is what keeps
marker genes from colliding with long labels; any width the axes have beyond what the content needs
widens the boxes. Row ``i`` sits at ``y = i`` and the y limits span exactly the rows, so another
axes with the same vertical extent lines up with the leaves.

Everything typographic comes from :class:`HierarchyStyle`; fonts come from ``rcParams``. The caller
owns the figure, its size and the keys, which :class:`HierarchyLayout` supports with a colour
mappable and line handles for the ribbons.

This is the one paper figure of an annotation run, so the caller builds and exports it following the
``sci-figures`` skill.
"""

import colorsys
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, to_hex, to_rgb
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath

from cell_type_annotation.hierarchy import LabelHierarchy, build_hierarchy, hierarchy_row_positions

#: Ribbon colours for an effect up and down, the ends of RdBu_r.
DIRECTIONAL_COLORS = {"up": "#b2182b", "down": "#2166ac"}

RibbonColor = Literal["enrichment", "family"]
RibbonWidth = Literal["significance", "n_cells", "constant"]
RGB = tuple[float, float, float]


@dataclass(frozen=True)
class HierarchyStyle:
    """Typography, ink and geometry of :func:`annotation_hierarchy`.

    Sizes and distances are in points, except ``box_height`` (a fraction of a row). Pass a
    modified copy, e.g. ``HierarchyStyle(label_size=7)``, to restyle the plot.
    """

    label_size: float = 6.0
    marker_size: float = 5.0
    number_size: float = 5.0
    header_size: float = 7.0
    subheader_size: float = 5.0
    badge_size: float = 5.0
    #: Baseline distance of stacked lines (labels, genes), as a multiple of their font size.
    line_spacing: float = 1.15
    ink: str = "black"
    ink_secondary: str = "#595959"
    #: Fraction of a row a node box fills.
    box_height: float = 0.8
    box_linewidth: float = 0.5
    box_rounding: float = 1.0
    #: Fraction of the way toward white for a box's fill, relative to its edge colour.
    tint: float = 0.72
    badge_diameter: float = 7.0
    #: Inner margin of a box.
    padding: float = 2.0
    #: Space between a box's label, genes and numbers.
    spacing: float = 3.0
    #: Minimum horizontal space between two columns, which the ribbons cross.
    column_gap: float = 12.0
    #: Length of the lead-in ribbon of each root node.
    stub: float = 6.0
    #: Linewidth range a ribbon can take.
    ribbon_linewidth: tuple[float, float] = (0.4, 3.0)
    #: Linewidth of a ribbon that could not be tested. Not testable is a statement about the tree,
    #: not a weak result, so it is fixed rather than the bottom of the significance scale.
    untested_linewidth: float = 1.0
    #: Opacity of the ribbons. Deliberately not an encoding channel: fading by significance on top
    #: of a colour ramp made both unreadable.
    ribbon_alpha: float = 0.8
    #: Height of a row when the function creates its own figure.
    row_height: float = 16.0
    #: Baseline of the level titles above the axes' top edge; ``None`` puts them just above the
    #: marker sub-headers. Raise it to make room for, say, tick labels of a neighbouring axes.
    title_offset: float | None = None
    #: Size and colour of a node's flag symbol, drawn where its markers would be.
    flag_size: float = 7.0
    flag_color: str = "#b2182b"


def _tint(color, amount: float) -> RGB:
    """Blend ``color`` toward white by ``amount``, for a box fill under a saturated edge."""
    r, g, b = to_rgb(color)
    return tuple(channel + (1.0 - channel) * amount for channel in (r, g, b))


def _ink_on(color) -> str:
    """Pick dark or light text for legibility on ``color``, by BT.601 luma."""
    r, g, b = to_rgb(color)
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"


def _splits(label: str, max_lines: int) -> list[list[str]]:
    """Every way to break ``label`` onto at most ``max_lines`` lines.

    Breaks fall at spaces, or after a hyphen before a lowercase letter, so "MB-HB" stays whole.
    """
    out = [[label]]
    if max_lines < 2:
        return out
    for i, char in enumerate(label):
        if char == " ":
            head, tail = label[:i], label[i + 1 :]
        elif char == "-" and 0 < i < len(label) - 1 and label[i + 1].islower():  # not inside "MB-HB"
            head, tail = label[: i + 1], label[i + 1 :]
        else:
            continue
        out += [[head, *rest] for rest in _splits(tail, max_lines - 1)]
    return out


def _wrap(label: str, width: float, measure, max_lines: int = 2) -> list[str]:
    """Break ``label`` onto the fewest lines no wider than ``width`` points.

    Falls back to the narrowest possible break when nothing fits.
    """
    options = sorted(_splits(label, max_lines), key=lambda lines: (len(lines), max(map(measure, lines))))
    fitting = [lines for lines in options if max(map(measure, lines)) <= width + 1e-6]
    return fitting[0] if fitting else min(options, key=lambda lines: max(map(measure, lines)))


def _text_width_points(text: str, size: float, weight: str = "normal", style: str = "normal") -> float:
    """Measure a string's rendered width in points, using the actual glyph outlines."""
    if not text:
        return 0.0
    prop = FontProperties(family="sans-serif", size=size, weight=weight, style=style)
    return TextPath((0, 0), text, prop=prop).get_extents().width


@dataclass
class HierarchyLayout:
    """Where :func:`annotation_hierarchy` placed everything.

    x is in points from the axes' left edge, y in rows (row ``i`` at ``y = i``).

    Attributes
    ----------
    ax
        The axes the tree was drawn into
    hierarchy
        The hierarchy that was drawn
    leaf_order
        Leaf labels top to bottom; row ``i`` is ``leaf_order[i]``
    y_of
        Vertical position of every ``(level, node)``
    leaf_y
        Vertical position of each leaf, i.e. of each row
    colors
        Node colour for every ``(level, node)``, including ``(leaf_level, leaf)``
    column_x
        ``(x0, width)`` of each drawn level column
    marker_x
        x of the marker-gene column of each drawn level, for drawing a custom mark there
    x_end
        Right edge of the drawn tree
    row_points
        Height of one row in points
    header_offsets
        ``(title, subheader)`` baselines in points above the axes' top edge, so neighbouring
        panels can align their own headers
    style
        The style the tree was drawn with
    """

    ax: Axes
    hierarchy: LabelHierarchy
    leaf_order: list[str]
    y_of: dict[tuple[str, str], float]
    leaf_y: dict[str, float]
    colors: dict[tuple[str, str], RGB]
    column_x: list[tuple[float, float]]
    marker_x: list[float]
    x_end: float
    row_points: float
    header_offsets: tuple[float, float]
    style: HierarchyStyle
    _enrichment: dict | None = field(default=None, repr=False)
    _ribbons: dict = field(default_factory=dict, repr=False)

    @property
    def leaf_level_name(self) -> str:
        """Name of the ``obs`` column whose categories are the figure's rows."""
        return self.hierarchy.leaf_level

    def text_width(self, text: str, size: float, weight: str = "normal", style: str = "normal") -> float:
        """Return the rendered width of ``text`` in points, the x unit of the axes."""
        return _text_width_points(text, size, weight, style)

    def enrichment_mappable(self) -> ScalarMappable:
        """Return the ribbon colour scale as a mappable, for a colorbar.

        Raises
        ------
        ValueError
            If the tree was drawn without enrichment colours.
        """
        info = self._require_enrichment()
        scale, metric = info["scale"], info["metric"]
        ramp = [
            _enrichment_color({metric: t * scale, "estimable": True}, scale, info["colors"], metric)
            for t in np.linspace(-1.0, 1.0, 257)
        ]
        return ScalarMappable(norm=Normalize(-scale, scale), cmap=ListedColormap(ramp))

    def ribbon_handles(self, qvalues: Sequence[float] = (1.0, 0.05, 0.001)) -> list[Line2D]:
        """Return legend handles for what the drawn ribbons' widths and dashes encode.

        Width examples per the ``ribbon_width`` in effect (none when every ribbon has one width),
        then the dashed untested ribbon if one was drawn.

        Parameters
        ----------
        qvalues
            Adjusted p-values to show, when width encodes significance. Labels are the values
            themselves; the last is the saturation point if it is at or below it

        Returns
        -------
        list
            :class:`~matplotlib.lines.Line2D` handles with labels, for ``ax.legend(handles=...)``.
        """
        info = self._ribbons
        style = self.style
        handles = []
        if info["width_by"] == "significance":
            for q in qvalues:
                lw = _ribbon_linewidth({"estimable": True, "qvalue": q}, "significance", 0, 1, style)
                label = f"≤{q:g}" if -np.log10(q) >= SIGNIFICANCE_SPAN else f"{q:g}"
                handles.append(Line2D([], [], color=NEUTRAL_RIBBON, lw=lw, solid_capstyle="butt", label=label))
        elif info["width_by"] == "n_cells" and info["counts"]:
            total = max(self.hierarchy.n_cells_total, 1)
            smallest = max(min(info["counts"]), 1)
            for n in (10**k for k in range(int(np.log10(smallest)), int(np.log10(total)) + 1)):
                lw = _ribbon_linewidth(None, "n_cells", n, total, style)
                handles.append(Line2D([], [], color=NEUTRAL_RIBBON, lw=lw, solid_capstyle="butt", label=f"{n:,}"))
        if info["dashed"]:
            lw = style.untested_linewidth
            handles.append(Line2D([], [], color=NEUTRAL_RIBBON, lw=lw, ls=_dashes_for(lw), label="Not tested"))
        return handles

    def _require_enrichment(self) -> dict:
        if self._enrichment is None:
            raise ValueError("This layout was drawn without enrichment colours, so the ribbons have no key")
        return self._enrichment


def hierarchy_palette(
    hierarchy: LabelHierarchy,
    *,
    root_colors: Mapping[str, Any] | None = None,
    hue_span: float = 80.0,
    lightness_spread: float = 0.16,
    sibling_spread: float = 0.12,
    group_level: str | None = None,
    grey_categories: Sequence[str] = ("Unknown", "Unassigned"),
) -> dict[tuple[str, str], RGB]:
    """Derive a colour for every node, coherent within each root family.

    Each root class gets an anchor colour and its leaves fan out around it, keeping saturation, so
    a family reads as one colour band. Every internal node then takes the mean colour of its own
    leaves, which makes a parent visibly the average of what it contains.

    Colour is built from two superposed components. A **family-wide ramp** in hue and lightness
    runs monotonically over the leaves, so a clade's colours stay continuous and neighbouring
    clades stay neighbouring colours. On top of it, a **within-group lightness offset** pulls
    apart the children of one parent, which the ramp alone cannot do: a family of 11 leaves would
    need most of the gamut to separate neighbours by eye, and stretching it that far stops the
    family reading as one colour.

    Both are needed. Driving everything from the family ramp leaves siblings indistinguishable
    (CIE76 ΔE under 3 in practice); driving hue from the group index instead makes every clade
    restart the same lightness ramp, so the last member of one clade collides with the last member
    of the next while jumping away from its own sibling.

    Parameters
    ----------
    hierarchy
        The hierarchy to colour
    root_colors
        Anchor colour per root-level label. Missing labels fall back to matplotlib's ``tab20``
    hue_span
        Hue arc in degrees spanned by one family
    lightness_spread
        Lightness ramp (+/-) across one family, which is what keeps a clade's colours continuous
    sibling_spread
        Extra lightness offset (+/-) applied within one sibling group, on top of that ramp. This
        is what pulls two children of the same parent apart
    group_level
        Level whose nodes define a sibling group. Defaults to the second-deepest drawn level, i.e.
        the parent of the deepest column
    grey_categories
        Root labels rendered as greys instead of hues, for classes such as ``'Unknown'`` that
        should not read as a biological identity

    Returns
    -------
    dict
        Mapping of ``(level, node)`` to an RGB triple.

    Examples
    --------
    >>> colors = hierarchy_palette(hierarchy, root_colors={"NPC": "#2d62a3", "Neuron": "#ED496F"})
    """
    root_level = hierarchy.all_levels[0]
    roots = hierarchy.nodes(root_level)

    anchors = dict(root_colors or {})
    palette = plt.get_cmap("tab20").colors
    for i, root in enumerate(r for r in roots if r not in anchors):
        anchors[root] = to_hex(palette[i % len(palette)])

    levels = hierarchy.levels
    group_level = group_level or (levels[-2] if len(levels) >= 2 else root_level)
    group_of = {leaf: node for node in hierarchy.nodes(group_level) for leaf in hierarchy.leaves_of(group_level, node)}

    leaf_color: dict[str, RGB] = {}
    grey = set(grey_categories)
    for root in roots:
        leaves = hierarchy.leaves_of(root_level, root)
        r, g, b = to_rgb(anchors[root])
        hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)

        # Leaves come in nested order, so one group's leaves are contiguous.
        groups: list[tuple[str, list[str]]] = []
        for leaf in leaves:
            node = group_of.get(leaf, leaf)
            if not groups or groups[-1][0] != node:
                groups.append((node, []))
            groups[-1][1].append(leaf)

        # Hue runs monotonically across the whole family, so a family reads as one continuous band
        # and neighbouring clades stay neighbouring colours. Lightness varies only *within* a
        # group, which is what separates siblings. Driving hue from the group index instead makes
        # every clade restart the same lightness ramp, so the last member of one clade collides
        # with the last member of the next while jumping away from its own sibling.
        n_leaves = len(leaves)
        j = 0
        for _node, members in groups:
            for li, leaf in enumerate(members):
                ramp = 0.5 if n_leaves == 1 else j / (n_leaves - 1)
                offset = 0.0 if len(members) == 1 else (li / (len(members) - 1) - 0.5) * 2
                j += 1
                if root in grey:
                    leaf_color[leaf] = colorsys.hls_to_rgb(0.0, 0.62 - 0.12 * ramp, 0.0)
                    continue
                shifted_hue = (hue + (ramp - 0.5) * hue_span / 360.0) % 1.0
                shifted_lightness = min(
                    0.86, max(0.24, lightness + (ramp - 0.5) * 2 * lightness_spread + offset * sibling_spread)
                )
                leaf_color[leaf] = colorsys.hls_to_rgb(shifted_hue, shifted_lightness, saturation)

    return {
        (level, node): tuple(np.array([leaf_color[leaf] for leaf in hierarchy.leaves_of(level, node)]).mean(0))
        for level, node in hierarchy.iter_nodes()
    }


def _enrichment_lookup(enrichment: pd.DataFrame | None, metric: str = "log_odds") -> dict[tuple[str, str], dict]:
    """Index an enrichment frame by ``(level, node)``."""
    if enrichment is None or enrichment.empty:
        return {}
    missing = {"level", "node", metric} - set(enrichment.columns)
    if missing:
        raise ValueError(f"`enrichment` is missing columns: {sorted(missing)}")
    return {(row["level"], row["node"]): row for _, row in enrichment.iterrows()}


NEUTRAL_RIBBON = "#b8b7b0"
#: -log10(q) at which the significance encoding saturates, i.e. q = 1e-3.
SIGNIFICANCE_SPAN = 3.0


def _is_testable(record: dict | None) -> bool:
    """Whether a node was actually tested, as opposed to having no siblings to test against."""
    return record is not None and bool(record.get("estimable", False))


def _significance(record: dict | None) -> float:
    """Return significance on a 0-1 scale from the adjusted p-value, or 0 when untested."""
    if not _is_testable(record):
        return 0.0
    q = record.get("qvalue", np.nan)
    if pd.isna(q):
        q = record.get("pvalue", np.nan)
    if pd.isna(q):
        return 0.0
    return float(np.clip(-np.log10(max(float(q), 1e-12)) / SIGNIFICANCE_SPAN, 0.0, 1.0))


def _enrichment_color(record: dict | None, scale: float, colors: Mapping[str, Any], metric: str) -> Any:
    """Map an enrichment record to a diverging colour: hue is direction, saturation is magnitude."""
    if not _is_testable(record):
        return to_rgb(NEUTRAL_RIBBON)
    effect = record.get(metric, 0.0)
    effect = 0.0 if pd.isna(effect) else float(effect)
    magnitude = min(abs(effect) / scale, 1.0) if scale > 0 else 0.0
    target = colors["up"] if effect > 0 else colors["down"]
    return tuple(np.array(to_rgb(NEUTRAL_RIBBON)) * (1 - magnitude) + np.array(to_rgb(target)) * magnitude)


def _ribbon_linewidth(record: dict | None, width_by: str, n_cells: int, n_total: int, style: HierarchyStyle) -> float:
    """Return a ribbon linewidth in points, from whichever variable drives width."""
    lo, hi = style.ribbon_linewidth
    if width_by == "n_cells":
        return lo + (hi - lo) * float(np.sqrt(n_cells / max(n_total, 1)))
    if width_by == "significance":
        if not _is_testable(record):
            return style.untested_linewidth
        return lo + (hi - lo) * _significance(record)
    return (lo + hi) / 2


def _dashes_for(lw: float, on: float = 3.0, off: float = 2.2) -> tuple[float, tuple[float, float]]:
    """Return a dash pattern that renders as ``on``/``off`` points regardless of linewidth.

    Matplotlib scales a dash pattern by its linewidth, so a thick dashed ribbon gets
    proportionally longer dashes — over a short span it shows a single dash and reads as solid.
    Dividing through by the linewidth cancels that scaling.
    """
    lw = max(lw, 1e-6)
    return (0.0, (on / lw, off / lw))


def _markers_by_node(markers: pd.DataFrame | None) -> dict[tuple[str, str], list[tuple[str, float, float]]]:
    """Group the markers to show by ``(level, node)``, keeping their order."""
    if markers is None or markers.empty:
        return {}
    missing = {"level", "node", "gene", "det_in", "det_out"} - set(markers.columns)
    if missing:
        raise ValueError(f"`markers` is missing columns: {sorted(missing)}")
    out: dict[tuple[str, str], list[tuple[str, float, float]]] = {}
    for r in markers.itertuples(index=False):
        out.setdefault((r.level, r.node), []).append((str(r.gene), float(r.det_in), float(r.det_out)))
    return out


def annotation_hierarchy(
    adata: AnnData | None = None,
    levels: Sequence[str] | None = None,
    *,
    ax: Axes | None = None,
    hierarchy: LabelHierarchy | None = None,
    leaf_level: str | None = None,
    root_order: Sequence[str] | None = None,
    markers: pd.DataFrame | None = None,
    flags: pd.DataFrame | None = None,
    enrichment: pd.DataFrame | None = None,
    enrichment_metric: Literal["log_odds", "effect_pts"] = "log_odds",
    enrichment_scale: float | None = None,
    enrichment_colors: Mapping[str, Any] | None = None,
    ribbon_color: RibbonColor = "enrichment",
    ribbon_width: RibbonWidth = "significance",
    root_stub: bool = True,
    bold_nodes: Collection[str] | None = None,
    display_names: Mapping[str, str] | None = None,
    level_titles: Sequence[str] | None = None,
    level_subtitles: Sequence[str] | None = None,
    marker_headers: tuple[str, str] | None = ("Marker", "In/out (%)"),
    badge_level: str | None = None,
    draw_badge: Callable[[Axes, float, float, Any, str], Any] | None = None,
    root_colors: Mapping[str, Any] | None = None,
    node_colors: Mapping[tuple[str, str], Any] | None = None,
    max_label_lines: int = 2,
    style: HierarchyStyle | None = None,
) -> HierarchyLayout:
    """Draw a nested annotation hierarchy as columns of labelled boxes joined by ribbons.

    One column is drawn per entry in ``levels``, with one row per leaf. Each node box carries its
    label, the marker genes it is given (italic) with their in/out detection percentages, and
    optionally a numbered badge. Ribbons join each node to its parent; their colour carries the
    condition effect when an enrichment frame is supplied, and their width its significance.

    The function works for any number of levels. ``leaf_level`` may name a column finer than the
    last drawn level, in which case rows are that column's categories and the deepest boxes simply
    have no outgoing ribbons.

    Parameters
    ----------
    adata
        AnnData object holding the label columns. Not needed if ``hierarchy`` is given
    levels
        Names of the ``obs`` columns to draw, coarse to fine
    ax
        Axes to draw into. The tree fills its width: any width beyond what the content needs
        widens the boxes, and too little raises. The y limits are set to span exactly the rows.
        Without ``ax``, a figure just wide enough is created
    hierarchy
        A pre-built hierarchy, used instead of deriving one from ``adata``
    leaf_level
        ``obs`` column defining the rows. Defaults to the last entry of ``levels``
    root_order
        Explicit ordering of the root-level labels, top to bottom
    markers
        The markers to show, one row per gene in display order, with columns ``level``, ``node``,
        ``gene``, ``det_in`` and ``det_out`` (percentage of cells detecting the gene in the node and
        in its siblings). Choosing them is the caller's job, e.g. with
        :func:`~cell_type_annotation.markers.display_markers`. Nodes without rows show none; a
        caller can draw its own mark at :attr:`HierarchyLayout.marker_x`. Pass ``None`` to draw a
        bare tree
    flags
        Nodes to mark with a symbol instead of markers, with columns ``level``, ``node`` and
        ``symbol``, as returned by :func:`~cell_type_annotation.markers.display_markers`
    enrichment
        Per-node condition effects used to colour the ribbons, with columns ``level``, ``node``,
        ``estimable``, ``qvalue`` and ``enrichment_metric``, e.g. from a parent-conditional
        composition model with the replicate as the unit
    enrichment_metric
        Column of ``enrichment`` driving the ribbon colour. ``'log_odds'`` (default) is the model's
        native scale and is independent of how abundant a node is; ``'effect_pts'`` is in
        percentage points, which is easier to read but systematically hides small classes — a node
        holding 3% of the cells cannot move more than a few points however condition-specific it is
    enrichment_scale
        Value of ``enrichment_metric`` that saturates the ribbon colour. Defaults to the largest
        absolute value among estimable nodes
    enrichment_colors
        Mapping with keys ``'up'`` and ``'down'`` giving the two directional colours. Defaults to
        :data:`DIRECTIONAL_COLORS`; pass the conditions' own colours where they have them
    ribbon_color
        ``'enrichment'`` colours ribbons by condition effect, ``'family'`` by the child's own
        colour. Falls back to ``'family'`` when no enrichment frame is given
    ribbon_width
        What drives ribbon thickness: ``'significance'`` (default), ``'n_cells'``, or
        ``'constant'``. A ribbon has only two readable channels, colour and width, so exactly one
        variable goes in each
    root_stub
        Whether to draw a short lead-in ribbon for each root-level node, so the root level shares
        the same encoding as the rest of the tree
    bold_nodes
        Node labels to draw in bold, for marking a class of node (organizing centres, say)
    display_names
        Text to show instead of a node's label, keyed by label; e.g. for sentence case
    level_titles
        Column headers. Defaults to the ``obs`` column names
    level_subtitles
        Text after each header, describing what the level means
    marker_headers
        Sub-headers over the gene and number columns, or ``None`` for none (and no width spent on them)
    badge_level
        Level whose boxes carry a numbered badge, numbered by that column's categorical order
    draw_badge
        ``draw_badge(ax, x, y, color, label)`` draws one badge, so a figure can use the same mark it
        uses elsewhere; its diameter must match ``style.badge_diameter``. Defaults to a filled
        circle with the label in black or white
    root_colors
        Anchor colour per root-level label, passed to :func:`hierarchy_palette`. Defaults to
        ``adata.uns[f"{levels[0]}_colors"]`` when present
    node_colors
        Explicit colour per ``(level, node)``, bypassing :func:`hierarchy_palette` entirely
    max_label_lines
        Maximum number of lines a label may wrap onto. A column is as wide as its widest label at
        its narrowest break; within that width (plus any spare width of the axes) each label takes
        as few lines as fit. Breaks fall at spaces, or after a hyphen before a lowercase letter
    style
        Typography, ink and geometry. Defaults to :class:`HierarchyStyle`'s defaults

    Returns
    -------
    HierarchyLayout
        Where everything was drawn, so further panels can be aligned to the same rows.

    Examples
    --------
    >>> shown, flags = display_markers(scores, hierarchy, markers, adata.var_names)
    >>> layout = annotation_hierarchy(adata, hierarchy=hierarchy, markers=shown, flags=flags)
    >>> layout.ax.scatter([layout.x_end + 2], [layout.leaf_y["Dorsal Telen. NPC"]])
    """
    if hierarchy is None:
        if adata is None or levels is None:
            raise ValueError("Pass either `hierarchy`, or both `adata` and `levels`")
        hierarchy = build_hierarchy(adata, levels, leaf_level=leaf_level, root_order=root_order)
    root_key = f"{hierarchy.all_levels[0]}_colors"
    if root_colors is None and adata is not None and root_key in adata.uns:
        categories = adata.obs[hierarchy.all_levels[0]].astype("category").cat.categories
        root_colors = dict(zip(map(str, categories), adata.uns[root_key], strict=False))  # nodes are strings

    style = style or HierarchyStyle()
    drawn_levels = list(hierarchy.levels)
    n_levels = len(drawn_levels)
    bold = set(bold_nodes or ())
    names = dict(display_names or {})

    titles = list(level_titles) if level_titles is not None else list(drawn_levels)
    subtitles = list(level_subtitles) if level_subtitles is not None else [""] * n_levels
    for name, seq in (("level_titles", titles), ("level_subtitles", subtitles)):
        if len(seq) != n_levels:
            raise ValueError(f"`{name}` has {len(seq)} entries but there are {n_levels} levels")

    colors: dict[tuple[str, str], RGB] = (
        {key: to_rgb(value) for key, value in node_colors.items()}
        if node_colors is not None
        else hierarchy_palette(hierarchy, root_colors=root_colors)
    )
    y_of = hierarchy_row_positions(hierarchy)
    leaf_y = {leaf: y_of[(hierarchy.leaf_level, leaf)] for leaf in hierarchy.leaf_order}
    n_rows = len(hierarchy.leaf_order)

    enrichment_by_node = _enrichment_lookup(enrichment, enrichment_metric)
    use_enrichment = ribbon_color == "enrichment" and bool(enrichment_by_node)
    if enrichment_scale is None:
        effects = [
            abs(float(r[enrichment_metric]))
            for r in enrichment_by_node.values()
            if r.get("estimable", False) and pd.notna(r[enrichment_metric])
        ]
        enrichment_scale = max(effects) if effects else 1.0
    if not np.isfinite(enrichment_scale) or enrichment_scale <= 0:
        enrichment_scale = 1.0
    directional = dict(enrichment_colors or DIRECTIONAL_COLORS)

    # --- column geometry in points, measured from rendered glyphs ---
    def measure(node):
        size, weight = style.label_size, "bold" if node in bold else "normal"
        return lambda line: _text_width_points(line, size, weight)

    def lines_of(node, width=0.0):
        """A label's lines: on as few as fit ``width``; at its narrowest when ``width`` is 0."""
        return _wrap(names.get(node, node), width, measure(node), max_label_lines)

    shown_by_node = _markers_by_node(markers)
    flag_of = {} if flags is None else {(r.level, r.node): str(r.symbol) for r in flags.itertuples(index=False)}
    badge_w, label_w, gene_w, number_w, block_w, min_w = [], [], [], [], [], []
    for i, level in enumerate(drawn_levels):
        badge_w.append(style.badge_diameter + style.spacing if level == badge_level else 0.0)
        label_w.append(
            max(
                _text_width_points(line, style.label_size, "bold" if node in bold else "normal")
                for node in hierarchy.nodes(level)
                for line in lines_of(node)
            )
        )
        shown = [m for node in hierarchy.nodes(level) for m in shown_by_node.get((level, node), [])]
        symbols = [flag_of[(level, node)] for node in hierarchy.nodes(level) if (level, node) in flag_of]
        if shown or symbols:
            heads = marker_headers or ("", "")
            gene_w.append(
                max(
                    [_text_width_points(t, style.marker_size, style="italic") for t, *_ in shown]
                    + [_text_width_points(s, style.flag_size, "bold") for s in symbols]
                    + [_text_width_points(heads[0], style.subheader_size)]
                )
            )
            number_w.append(
                max(
                    [_text_width_points(f"{a:.0f}/{b:.0f}", style.number_size) for _t, a, b in shown]
                    + [_text_width_points(heads[1], style.subheader_size)]
                )
            )
            block_w.append(gene_w[-1] + style.spacing + number_w[-1])
        else:
            gene_w.append(0.0)
            number_w.append(0.0)
            block_w.append(0.0)
        title_w = _text_width_points(titles[i], style.header_size, "bold")
        if subtitles[i]:
            title_w += style.header_size * 0.4 + _text_width_points(subtitles[i], style.subheader_size)
        min_w.append(
            max(
                2 * style.padding + badge_w[i] + label_w[i] + (2 * style.spacing + block_w[i] if block_w[i] else 0),
                title_w,
            )
        )

    stub = style.stub if root_stub else 0.0
    needed = stub + sum(min_w) + style.column_gap * (n_levels - 1)
    default_title = style.subheader_size + 2.0 + 2.5  # just above the marker sub-headers
    header_offsets = (default_title if style.title_offset is None else style.title_offset, 2.0)
    header_pt = header_offsets[0] + style.header_size

    if ax is None:
        fig_w, fig_h = needed / 72.0, (n_rows * style.row_height + header_pt) / 72.0
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes((0, 0, 1, n_rows * style.row_height / 72.0 / fig_h))
    fig = ax.figure
    position = ax.get_position()
    width_pt = position.width * fig.get_figwidth() * 72.0
    height_pt = position.height * fig.get_figheight() * 72.0
    if needed > width_pt + 1e-6:
        raise ValueError(
            f"The hierarchy needs {needed / 72 * 25.4:.1f} mm but the axes are {width_pt / 72 * 25.4:.1f} mm wide. "
            "Widen the axes, allow more `max_label_lines`, or reduce sizes in `style`."
        )
    extra = (width_pt - needed) / n_levels
    widths = [w + extra for w in min_w]
    x0, cursor = [], stub
    for w in widths:
        x0.append(cursor)
        cursor += w + style.column_gap
    x_end = x0[-1] + widths[-1]
    number_x = [x0[i] + widths[i] - style.padding - number_w[i] for i in range(n_levels)]
    marker_x = [x0[i] + widths[i] - style.padding - block_w[i] for i in range(n_levels)]

    ax.set_xlim(0, width_pt)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.axis("off")
    row_pt = height_pt / n_rows

    def rows(points):
        return points / row_pt

    # --- ribbons ---
    # What the drawn ribbons actually encode, for their key: without test records, significance
    # widths collapse to one width, and only drawn untested ribbons are dashed.
    width_by = "constant" if ribbon_width == "significance" and not enrichment_by_node else ribbon_width
    ribbons = {"width_by": width_by, "dashed": False, "counts": []}

    def draw_ribbon(x_start, y_start, x_stop, y_stop, color, linestyle, lw, n_cells):
        ribbons["dashed"] |= linestyle != "solid"
        ribbons["counts"].append(n_cells)
        dx = (x_stop - x_start) * 0.45
        path = MPath(
            [(x_start, y_start), (x_start + dx, y_start), (x_stop - dx, y_stop), (x_stop, y_stop)],
            [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
        )
        ax.add_patch(
            PathPatch(path, fc="none", ec=color, lw=lw, alpha=style.ribbon_alpha, linestyle=linestyle, zorder=1)
        )

    def ribbon_style(level, node):
        """Return ``(colour, linestyle, linewidth, n_cells)`` for one node's incoming ribbon."""
        record = enrichment_by_node.get((level, node)) if enrichment_by_node else None
        lw = _ribbon_linewidth(record, ribbon_width, hierarchy.n_cells(level, node), hierarchy.n_cells_total, style)
        color = (
            _enrichment_color(record, enrichment_scale, directional, enrichment_metric)
            if use_enrichment
            else colors[(level, node)]
        )
        # A node with no siblings was never tested; dashed says so without implying "no effect".
        linestyle = "solid" if (record is None or _is_testable(record)) else _dashes_for(lw)
        return color, linestyle, lw, hierarchy.n_cells(level, node)

    if root_stub:
        root_level = drawn_levels[0]
        for node in hierarchy.nodes(root_level):
            y = y_of[(root_level, node)]
            draw_ribbon(0.0, y, x0[0], y, *ribbon_style(root_level, node))

    for i in range(n_levels - 1):
        parent_level, child_level = drawn_levels[i], drawn_levels[i + 1]
        for node in hierarchy.nodes(child_level):
            parent = hierarchy.parent_of(child_level, node)
            draw_ribbon(
                x0[i] + widths[i],
                y_of[(parent_level, parent)],
                x0[i + 1],
                y_of[(child_level, node)],
                *ribbon_style(child_level, node),
            )

    # --- node boxes ---
    badge_numbers: dict[str, int] = {}
    if badge_level is not None:
        if adata is not None and isinstance(adata.obs[badge_level].dtype, pd.CategoricalDtype):
            badge_numbers = {str(c): i for i, c in enumerate(adata.obs[badge_level].cat.categories, 1)}
        else:
            badge_numbers = {n: i for i, n in enumerate(hierarchy.nodes(badge_level), 1)}

    def stacked(n, size):
        """Offsets in rows that centre ``n`` lines of ``size``-point text on a row."""
        return [rows((k - (n - 1) / 2) * size * style.line_spacing) for k in range(n)]

    text = {"zorder": 3, "va": "center_baseline"}
    for i, level in enumerate(drawn_levels):
        for node in hierarchy.nodes(level):
            color = colors[(level, node)]
            y = y_of[(level, node)]
            ax.add_patch(
                FancyBboxPatch(
                    (x0[i], y - style.box_height / 2),
                    widths[i],
                    style.box_height,
                    boxstyle=f"round,pad=0,rounding_size={style.box_rounding}",
                    mutation_aspect=1 / row_pt,
                    fc=_tint(color, style.tint),
                    ec=color,
                    lw=style.box_linewidth,
                    zorder=2,
                )
            )

            label_x = x0[i] + style.padding
            if level == badge_level and node in badge_numbers:
                bx = label_x + style.badge_diameter / 2
                if draw_badge is not None:
                    draw_badge(ax, bx, y, color, str(badge_numbers[node]))
                else:
                    ax.scatter(
                        [bx], [y], s=style.badge_diameter**2, c=[color], edgecolors="none", zorder=4, clip_on=False
                    )
                    ax.text(
                        bx, y, str(badge_numbers[node]), ha="center", fontsize=style.badge_size,
                        color=_ink_on(color), zorder=5, va="center_baseline",
                    )  # fmt: skip
                label_x += badge_w[i]

            lines = lines_of(node, label_w[i] + extra)
            for line, dy in zip(lines, stacked(len(lines), style.label_size), strict=True):
                ax.text(
                    label_x, y + dy, line, ha="left", fontsize=style.label_size, color=style.ink,
                    weight="bold" if node in bold else "normal", **text,
                )  # fmt: skip

            shown = shown_by_node.get((level, node), [])
            for (gene, det_in, det_out), dy in zip(shown, stacked(len(shown), style.marker_size), strict=True):
                ax.text(
                    marker_x[i], y + dy, gene, ha="left", fontsize=style.marker_size, color=style.ink,
                    style="italic", **text,
                )  # fmt: skip
                ax.text(
                    number_x[i], y + dy, f"{det_in:.0f}/{det_out:.0f}", ha="left", fontsize=style.number_size,
                    color=style.ink_secondary, **text,
                )  # fmt: skip
            if (level, node) in flag_of:
                ax.text(
                    marker_x[i], y, flag_of[(level, node)], ha="left", fontsize=style.flag_size,
                    color=style.flag_color, weight="bold", **text,
                )  # fmt: skip

    # --- column headers, above the axes ---
    def header(x, offset, string, **kwargs):
        ax.annotate(
            string, xy=(x, 1.0), xycoords=("data", "axes fraction"), xytext=(0, offset),
            textcoords="offset points", ha="left", va="baseline", annotation_clip=False, **kwargs,
        )  # fmt: skip

    title_y, sub_y = header_offsets
    for i in range(n_levels):
        header(x0[i], title_y, titles[i], fontsize=style.header_size, color=style.ink, weight="bold")
        if subtitles[i]:
            gap = _text_width_points(titles[i], style.header_size, "bold") + style.header_size * 0.4
            header(x0[i] + gap, title_y, subtitles[i], fontsize=style.subheader_size, color=style.ink_secondary)
        if gene_w[i] and marker_headers:
            header(marker_x[i], sub_y, marker_headers[0], fontsize=style.subheader_size, color=style.ink_secondary)
            header(number_x[i], sub_y, marker_headers[1], fontsize=style.subheader_size, color=style.ink_secondary)

    return HierarchyLayout(
        ax=ax,
        hierarchy=hierarchy,
        leaf_order=list(hierarchy.leaf_order),
        y_of=y_of,
        leaf_y=leaf_y,
        colors=colors,
        column_x=list(zip(x0, widths, strict=True)),
        marker_x=marker_x,
        x_end=x_end,
        row_points=row_pt,
        header_offsets=header_offsets,
        style=style,
        _enrichment=(
            {"metric": enrichment_metric, "scale": enrichment_scale, "colors": directional, "width_by": ribbon_width}
            if use_enrichment
            else None
        ),
        _ribbons=ribbons,
    )
