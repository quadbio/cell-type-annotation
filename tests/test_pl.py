import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import FancyBboxPatch, PathPatch

from cell_type_annotation.hierarchy import build_hierarchy, hierarchy_markers
from cell_type_annotation.pl import HierarchyStyle, _wrap, annotation_hierarchy, hierarchy_palette

LEVELS = ["level_1", "level_2", "level_3"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def enrichment():
    """Per-node effects as a composition model would return them; mid_c1 has no siblings."""
    rows = [
        ("level_2", "mid_a1", 0.4, True, 0.2),
        ("level_2", "mid_a2", -0.4, True, 0.2),
        ("level_2", "mid_b1", 1.5, True, 1e-4),
        ("level_2", "mid_b2", -1.5, True, 1e-4),
        ("level_2", "mid_c1", np.nan, False, np.nan),
    ]
    return pd.DataFrame(rows, columns=["level", "node", "log_odds", "estimable", "qvalue"])


def _counts(ax):
    boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    ribbons = [p for p in ax.patches if isinstance(p, PathPatch)]
    return len(boxes), len(ribbons)


class TestHierarchyPalette:
    def test_internal_node_is_the_mean_of_its_leaves(self, adata_hierarchy):
        hierarchy = build_hierarchy(adata_hierarchy, LEVELS)
        colors = hierarchy_palette(hierarchy, root_colors={"root_a": "#2d62a3"})

        leaf_colors = [colors[("level_3", leaf)] for leaf in hierarchy.leaves_of("level_1", "root_a")]
        assert colors[("level_1", "root_a")] == pytest.approx(np.mean(leaf_colors, axis=0))

    def test_grey_categories_are_desaturated(self, adata_hierarchy):
        hierarchy = build_hierarchy(adata_hierarchy, LEVELS)
        r, g, b = hierarchy_palette(hierarchy, grey_categories=("root_b",))[("level_1", "root_b")]
        assert r == pytest.approx(g) == pytest.approx(b)

    def test_is_deterministic(self, adata_hierarchy):
        hierarchy = build_hierarchy(adata_hierarchy, LEVELS)
        assert hierarchy_palette(hierarchy) == hierarchy_palette(hierarchy)


class TestAnnotationHierarchy:
    def test_draws_one_box_per_node(self, adata_hierarchy):
        layout = annotation_hierarchy(adata_hierarchy, LEVELS)
        assert _counts(layout.ax)[0] == sum(len(layout.hierarchy.nodes(level)) for level in LEVELS)

    def test_root_colours_come_from_uns(self, adata_hierarchy):
        adata = adata_hierarchy.copy()
        adata.uns["level_1_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
        layout = annotation_hierarchy(adata, ["level_1"])
        assert layout.colors[("level_1", "root_a")] == pytest.approx((1.0, 0.0, 0.0))

    def test_root_stub_adds_one_ribbon_per_root(self, adata_hierarchy):
        n_with = _counts(annotation_hierarchy(adata_hierarchy, LEVELS, root_stub=True).ax)[1]
        without = annotation_hierarchy(adata_hierarchy, LEVELS, root_stub=False)
        assert n_with - _counts(without.ax)[1] == len(without.hierarchy.nodes("level_1"))

    def test_layout_rows_match_the_leaves(self, adata_hierarchy):
        layout = annotation_hierarchy(adata_hierarchy, LEVELS)

        assert layout.leaf_order == list(layout.hierarchy.leaf_order)
        assert layout.leaf_y[layout.leaf_order[0]] == 0.0
        assert layout.x_end > 0
        assert layout.text_width("ABC", 7) > 0

    @pytest.mark.parametrize("levels", [["level_1", "level_2"], LEVELS])
    def test_works_for_any_number_of_levels(self, adata_hierarchy, levels):
        assert len(annotation_hierarchy(adata_hierarchy, levels).column_x) == len(levels)

    def test_leaf_level_beyond_drawn_levels(self, adata_hierarchy):
        layout = annotation_hierarchy(adata_hierarchy, ["level_1", "level_2"], leaf_level="level_3")
        assert len(layout.column_x) == 2
        assert len(layout.leaf_order) == 6

    def test_draws_exactly_the_markers_given(self, adata_hierarchy):
        markers = hierarchy_markers(adata_hierarchy, LEVELS, n_top=5)
        chosen = markers[(markers.node == "root_a") & (markers.gene == "flat")]
        drawn = {t.get_text() for t in annotation_hierarchy(adata_hierarchy, LEVELS, markers=chosen).ax.texts}
        row = chosen.iloc[0]
        assert "flat" in drawn and f"{row.det_in:.0f}/{row.det_out:.0f}" in drawn
        assert not set(markers.gene) - {"flat"} & drawn

    def test_flags_are_drawn_in_the_marker_column(self, adata_hierarchy):
        flags = pd.DataFrame({"level": ["level_2"], "node": ["mid_b2"], "symbol": ["†"]})
        layout = annotation_hierarchy(adata_hierarchy, LEVELS, flags=flags)
        (text,) = [t for t in layout.ax.texts if t.get_text() == "†"]
        assert text.get_position() == pytest.approx((layout.marker_x[1], layout.y_of[("level_2", "mid_b2")]))

    def test_markers_need_their_columns(self, adata_hierarchy):
        with pytest.raises(ValueError, match="missing columns"):
            annotation_hierarchy(
                adata_hierarchy, LEVELS, markers=pd.DataFrame({"level": ["level_1"], "gene": ["flat"]})
            )

    def test_ribbon_keys_match_the_scale(self, adata_hierarchy, enrichment):
        layout = annotation_hierarchy(adata_hierarchy, LEVELS, enrichment=enrichment, enrichment_scale=2.0)

        assert layout.enrichment_mappable().get_clim() == (-2.0, 2.0)
        handles = layout.ribbon_handles()
        assert handles[-1].get_label() == "Not tested"
        widths = [h.get_linewidth() for h in handles[:-1]]
        assert widths == sorted(widths)

    def test_family_ribbons_when_no_enrichment_given(self, adata_hierarchy):
        assert _counts(annotation_hierarchy(adata_hierarchy, LEVELS, ribbon_color="enrichment").ax)[1] > 0

    def test_keys_require_enrichment(self, adata_hierarchy):
        with pytest.raises(ValueError, match="without enrichment"):
            annotation_hierarchy(adata_hierarchy, LEVELS).enrichment_mappable()

    def test_integer_categories_keep_badges_and_colours(self, adata_hierarchy):
        adata = adata_hierarchy.copy()
        codes = {name: i for i, name in enumerate(sorted(adata.obs["level_3"].cat.categories))}
        adata.obs["leaf_id"] = pd.Categorical(adata.obs["level_3"].map(codes).astype(int))
        adata.obs["root_id"] = pd.Categorical(adata.obs["level_1"].map({"root_a": 1, "root_b": 2, "root_c": 3}))
        adata.uns["root_id_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
        layout = annotation_hierarchy(adata, ["root_id", "leaf_id"], badge_level="leaf_id")
        r, g, b = layout.colors[("root_id", "1")]  # the mean of a red family, not tab20's blue fallback
        assert r > max(g, b) + 0.3
        assert {"1", "6"} <= {t.get_text() for t in layout.ax.texts}

    def test_badges_are_drawn(self, adata_hierarchy):
        layout = annotation_hierarchy(adata_hierarchy, LEVELS, badge_level="level_3")
        assert "1" in {t.get_text() for t in layout.ax.texts}

    def test_fills_the_axes_and_aligns_rows(self, adata_hierarchy):
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_axes((0, 0, 1, 0.8))
        layout = annotation_hierarchy(adata_hierarchy, LEVELS, ax=ax)

        assert layout.x_end == pytest.approx(12 * 72)
        assert ax.get_ylim() == (len(layout.leaf_order) - 0.5, -0.5)

    def test_too_narrow_axes_raise(self, adata_hierarchy):
        fig = plt.figure(figsize=(1, 4))
        with pytest.raises(ValueError, match="Widen the axes"):
            annotation_hierarchy(adata_hierarchy, LEVELS, ax=fig.add_axes((0, 0, 1, 1)))

    def test_markers_are_italic_and_display_names_are_used(self, adata_hierarchy):
        markers = hierarchy_markers(adata_hierarchy, LEVELS, n_top=5)
        layout = annotation_hierarchy(
            adata_hierarchy,
            LEVELS,
            markers=markers[(markers.node == "root_a") & (markers.gene == "flat")],
            display_names={"root_a": "Root A"},
        )
        texts = {t.get_text(): t for t in layout.ax.texts}
        assert texts["flat"].get_style() == "italic"
        assert "Root" in texts and "root_a" not in texts

    def test_labels_wrap_by_width_and_keep_abbreviations_whole(self):
        assert _wrap("Dorsal MB-HB NPC", 20, len) == ["Dorsal MB-HB NPC"]
        assert _wrap("Dorsal MB-HB NPC", 12, len) == ["Dorsal", "MB-HB NPC"]
        assert _wrap("Thalamo-pretectal neuron", 18, len) == ["Thalamo-", "pretectal neuron"]

    def test_width_key_follows_the_drawn_ribbons(self, adata_hierarchy):
        assert annotation_hierarchy(adata_hierarchy, LEVELS).ribbon_handles() == []
        layout = annotation_hierarchy(adata_hierarchy, LEVELS, ribbon_color="family", ribbon_width="n_cells")
        assert layout.ribbon_handles()

    def test_zero_title_offset_is_honoured(self, adata_hierarchy):
        layout = annotation_hierarchy(adata_hierarchy, LEVELS, style=HierarchyStyle(title_offset=0.0))
        assert layout.header_offsets[0] == 0.0

    def test_mismatched_titles_raise(self, adata_hierarchy):
        with pytest.raises(ValueError, match="level_titles"):
            annotation_hierarchy(adata_hierarchy, LEVELS, level_titles=["only one"])

    def test_requires_adata_or_hierarchy(self):
        with pytest.raises(ValueError, match="Pass either"):
            annotation_hierarchy()
