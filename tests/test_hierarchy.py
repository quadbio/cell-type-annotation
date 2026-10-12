import numpy as np
import pandas as pd
import pytest

from cell_type_annotation.hierarchy import build_hierarchy, hierarchy_markers, hierarchy_row_positions

LEVELS = ["level_1", "level_2", "level_3"]


class TestBuildHierarchy:
    def test_structure(self, adata_hierarchy):
        hierarchy = build_hierarchy(adata_hierarchy, LEVELS)

        assert hierarchy.levels == tuple(LEVELS)
        assert hierarchy.leaf_level == "level_3"
        assert set(hierarchy.nodes("level_1")) == {"root_a", "root_b", "root_c"}
        assert hierarchy.children_of("level_1", "root_a") == ["mid_a1", "mid_a2"]
        assert hierarchy.parent_of("level_2", "mid_a1") == "root_a"
        assert hierarchy.parent_of("level_1", "root_a") is None
        assert sorted(hierarchy.leaves_of("level_1", "root_a")) == ["a1x", "a1y", "a2x"]

    def test_siblings_are_local_to_the_parent(self, adata_hierarchy):
        hierarchy = build_hierarchy(adata_hierarchy, LEVELS)

        assert hierarchy.siblings_of("level_2", "mid_a1") == ["mid_a2"]
        assert hierarchy.siblings_of("level_2", "mid_b1") == ["mid_b2"]
        assert sorted(hierarchy.siblings_of("level_1", "root_a")) == ["root_b", "root_c"]
        assert hierarchy.siblings_of("level_2", "mid_c1") == []

    def test_leaf_order_is_nested(self, adata_hierarchy):
        hierarchy = build_hierarchy(adata_hierarchy, LEVELS, root_order=["root_a", "root_b", "root_c"])

        order = list(hierarchy.leaf_order)
        assert order.index("a1x") < order.index("b1x") < order.index("c1x")
        for root in hierarchy.nodes("level_1"):
            positions = [order.index(leaf) for leaf in hierarchy.leaves_of("level_1", root)]
            assert max(positions) - min(positions) == len(positions) - 1

    def test_root_order_is_respected(self, adata_hierarchy):
        hierarchy = build_hierarchy(adata_hierarchy, LEVELS, root_order=["root_c", "root_b", "root_a"])
        assert hierarchy.nodes("level_1") == ["root_c", "root_b", "root_a"]
        assert hierarchy.leaf_order[0] == "c1x"

    def test_declared_category_order_is_kept(self, adata_hierarchy):
        adata = adata_hierarchy.copy()
        adata.obs["level_1"] = adata.obs["level_1"].cat.reorder_categories(["root_c", "root_a", "root_b"])
        adata.obs["level_2"] = adata.obs["level_2"].cat.reorder_categories(
            ["mid_a2", "mid_a1", "mid_b2", "mid_b1", "mid_c1"]
        )
        hierarchy = build_hierarchy(adata, LEVELS, child_order="category")
        assert hierarchy.nodes("level_1") == ["root_c", "root_a", "root_b"]
        assert hierarchy.children_of("level_1", "root_a") == ["mid_a2", "mid_a1"]

    def test_unobserved_categories_are_not_nodes(self, adata_hierarchy):
        adata = adata_hierarchy.copy()
        adata.obs["level_2"] = adata.obs["level_2"].cat.add_categories(["never_seen"])
        hierarchy = build_hierarchy(adata, LEVELS, child_order="category")
        assert "never_seen" not in hierarchy.nodes("level_2")
        assert hierarchy.children_of("level_1", "root_a") == ["mid_a1", "mid_a2"]

    def test_leaf_level_beyond_drawn_levels(self, adata_hierarchy):
        hierarchy = build_hierarchy(adata_hierarchy, ["level_1", "level_2"], leaf_level="level_3")

        assert hierarchy.levels == ("level_1", "level_2")
        assert hierarchy.all_levels == ("level_1", "level_2", "level_3")
        assert len(hierarchy.leaf_order) == 6

    def test_intermediate_leaf_level_raises(self, adata_hierarchy):
        with pytest.raises(ValueError, match="finest level"):
            build_hierarchy(adata_hierarchy, LEVELS, leaf_level="level_2")

    def test_non_nested_labels_raise(self, adata_hierarchy):
        adata = adata_hierarchy.copy()
        level_1 = adata.obs["level_1"].astype(str)
        level_1.iloc[np.flatnonzero(adata.obs["level_2"] == "mid_a1")[0]] = "root_b"
        adata.obs["level_1"] = pd.Categorical(level_1)

        with pytest.raises(ValueError, match="do not nest strictly"):
            build_hierarchy(adata, LEVELS)

    def test_sub_cluster_ids_are_checked(self, adata_hierarchy):
        """A finer level on sub-cluster ids is checked per cell, not silently skipped."""
        adata = adata_hierarchy.copy()
        adata.obs["sub"] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2, "s0", "s1"))
        with pytest.raises(ValueError, match="do not nest strictly"):
            build_hierarchy(adata, ["level_1", "sub"])

    def test_row_positions_centre_parents_on_their_leaves(self, adata_hierarchy):
        hierarchy = build_hierarchy(adata_hierarchy, LEVELS, root_order=["root_a", "root_b", "root_c"])
        y = hierarchy_row_positions(hierarchy)

        leaf_y = [y[("level_3", leaf)] for leaf in hierarchy.leaves_of("level_1", "root_a")]
        assert y[("level_1", "root_a")] == pytest.approx(np.mean(leaf_y))


class TestHierarchyMarkers:
    def test_margins_are_exact(self, adata_hierarchy):
        markers = hierarchy_markers(adata_hierarchy, LEVELS, n_top=5)
        row = markers[(markers.node == "root_a") & (markers.gene == "mark_a")].iloc[0]

        assert row["det_in"] == pytest.approx(100.0)
        assert row["det_out"] == pytest.approx(0.0)
        assert row["score"] == pytest.approx(100.0)

    def test_scoring_is_against_siblings_only(self, adata_hierarchy):
        markers = hierarchy_markers(adata_hierarchy, LEVELS, n_top=5)

        shared = markers[(markers.node == "mid_a1") & (markers.gene == "shared_a1_b1")].iloc[0]
        assert shared["score"] == pytest.approx(100.0)
        shared_b = markers[(markers.node == "mid_b1") & (markers.gene == "shared_a1_b1")].iloc[0]
        assert shared_b["score"] == pytest.approx(100.0)

    def test_leaf_min_is_stricter_than_pooled(self, adata_hierarchy):
        leaf_min = hierarchy_markers(adata_hierarchy, LEVELS, aggregate="leaf_min", n_top=5)
        pooled = hierarchy_markers(adata_hierarchy, LEVELS, aggregate="pooled", n_top=5)

        strict = leaf_min[(leaf_min.node == "root_a") & (leaf_min.gene == "weak_a")].iloc[0]
        loose = pooled[(pooled.node == "root_a") & (pooled.gene == "weak_a")].iloc[0]
        assert strict["det_in"] == pytest.approx(0.0)
        assert loose["det_in"] > 0.0
        assert strict["score"] < loose["score"]

    def test_cells_dropped_from_the_hierarchy_are_not_counted(self, adata_hierarchy):
        """A root_a cell without a level_1 label must not count toward its leaf's detection."""
        adata = adata_hierarchy.copy()
        outsider = np.flatnonzero(adata.obs["level_3"] == "a2x")[0]
        adata.X[outsider, list(adata.var_names).index("mark_a1")] = 5.0  # a2x otherwise never detects it
        adata.obs["level_1"] = adata.obs["level_1"].astype(object)
        adata.obs.iloc[outsider, adata.obs.columns.get_loc("level_1")] = np.nan

        markers = hierarchy_markers(adata, LEVELS, genes=["mark_a1"], n_top=1)
        row = markers[(markers.node == "mid_a2") & (markers.gene == "mark_a1")].iloc[0]
        assert row["det_in"] == pytest.approx(0.0)

    def test_sibling_less_nodes_are_skipped(self, adata_hierarchy):
        markers = hierarchy_markers(adata_hierarchy, LEVELS)
        assert "mid_c1" not in set(markers["node"])

    def test_genes_restricts_the_panel(self, adata_hierarchy):
        markers = hierarchy_markers(adata_hierarchy, LEVELS, genes=["flat", "mark_a"], n_top=5)
        assert set(markers["gene"]) == {"flat", "mark_a"}

    def test_include_genes_survives_the_top_n_cut(self, adata_hierarchy):
        top1 = hierarchy_markers(adata_hierarchy, LEVELS, n_top=1)
        assert "flat" not in set(top1[top1.node == "root_a"]["gene"])

        forced = hierarchy_markers(adata_hierarchy, LEVELS, n_top=1, include_genes=["flat"])
        assert "flat" in set(forced[forced.node == "root_a"]["gene"])

    def test_layer_and_threshold_change_the_result(self, adata_hierarchy):
        detected = hierarchy_markers(adata_hierarchy, LEVELS, layer="counts", threshold=3.0, n_top=1)
        undetected = hierarchy_markers(adata_hierarchy, LEVELS, layer="counts", threshold=3.5, n_top=1)

        assert detected[detected.node == "root_a"]["det_in"].max() == pytest.approx(100.0)
        assert undetected["det_in"].max() == pytest.approx(0.0)

    def test_zero_threshold_on_sparse_input_raises(self, adata_hierarchy):
        import scipy.sparse as sp

        adata = adata_hierarchy.copy()
        adata.X = sp.csr_matrix(adata.X)
        with pytest.raises(ValueError, match="every zero as detected"):
            hierarchy_markers(adata, LEVELS, threshold=0.0)

    def test_unknown_gene_raises(self, adata_hierarchy):
        with pytest.raises(ValueError, match="not found"):
            hierarchy_markers(adata_hierarchy, LEVELS, genes=["nope"])
