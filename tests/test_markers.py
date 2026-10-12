import pandas as pd
import pytest

from cell_type_annotation.hierarchy import build_hierarchy, hierarchy_markers
from cell_type_annotation.markers import display_markers, read_markers

LEVELS = ["level_1", "level_2"]


def _table(rows):
    return read_markers(pd.DataFrame(rows, columns=["level", "node", "gene", "sign"]))


class TestReadMarkers:
    def test_fills_optional_columns(self, tmp_path):
        path = tmp_path / "markers.csv"
        path.write_text("level,node,gene,sign\nL1,T cell,CD3E,POS\nL1,T cell,MS4A1,neg\n")
        table = read_markers(path)
        assert list(table.columns[:7]) == ["level", "node", "gene", "sign", "source", "pmid", "note"]
        assert table["sign"].tolist() == ["pos", "neg"]

    def test_rejects_unknown_sign(self):
        with pytest.raises(ValueError, match="'pos' or 'neg'"):
            _table([("L1", "T cell", "CD3E", "yes")])

    def test_rejects_a_gene_with_both_signs(self):
        with pytest.raises(ValueError, match="both positive and negative"):
            _table([("L1", "T cell", "CD3E", "pos"), ("L1", "T cell", "CD3E", "neg")])

    def test_a_gene_may_be_positive_for_one_node_and_negative_for_another(self):
        table = _table([("L1", "Neuron", "DCX", "pos"), ("L1", "NPC", "DCX", "neg")])
        assert len(table) == 2

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing columns"):
            read_markers(pd.DataFrame({"node": ["a"], "gene": ["b"]}))


class TestDisplayMarkers:
    @pytest.fixture()
    def setup(self, adata_hierarchy):
        markers = _table(
            [
                ("level_1", "root_a", "mark_a", "pos"),
                ("level_1", "root_a", "flat", "pos"),
                ("level_1", "root_b", "OFFPANEL", "pos"),
                ("level_2", "mid_a1", "shared_a1_b1", "pos"),
                ("level_2", "mid_a2", "flat", "pos"),
            ]
        )
        hierarchy = build_hierarchy(adata_hierarchy, LEVELS, leaf_level="level_3")
        measured = sorted(set(markers["gene"]) & set(adata_hierarchy.var_names))
        scores = hierarchy_markers(adata_hierarchy, LEVELS, leaf_level="level_3", include_genes=measured, n_top=1)
        return scores, hierarchy, markers, adata_hierarchy.var_names

    def test_shows_separating_canonical_markers_only(self, setup):
        shown, _ = display_markers(*setup)
        assert shown.set_index(["level", "node"])["gene"].to_dict() == {
            ("level_1", "root_a"): "mark_a",
            ("level_2", "mid_a1"): "shared_a1_b1",
        }

    def test_flags_nodes_without_a_shown_marker(self, setup):
        _, flags = display_markers(*setup)
        symbol = flags.set_index("node")["symbol"].to_dict()
        assert symbol["root_b"] == "†"  # its only canonical marker is off-panel
        assert symbol["mid_a2"] == "−"  # flat does not separate it from mid_a1
        assert symbol["mid_b1"] == "−"  # nothing curated
        assert "mid_c1" not in symbol  # no siblings, nothing to separate from

    def test_unknown_nodes_get_the_empty_set(self, adata_hierarchy):
        adata = adata_hierarchy.copy()
        adata.obs["level_1"] = adata.obs["level_1"].cat.rename_categories({"root_c": "Unknown"})
        hierarchy = build_hierarchy(adata, ["level_1"])
        scores = hierarchy_markers(adata, ["level_1"], hierarchy=hierarchy)
        _, flags = display_markers(scores, hierarchy, _table([]), adata.var_names)
        assert flags.set_index("node").loc["Unknown", "symbol"] == "∅"

    def test_a_sibling_detecting_it_more_widely_blocks_it(self, adata_hierarchy):
        """flat has margin 0 everywhere, so even a low threshold must not show it."""
        markers = _table([("level_1", "root_a", "flat", "pos")])
        hierarchy = build_hierarchy(adata_hierarchy, ["level_1"])
        scores = hierarchy_markers(adata_hierarchy, ["level_1"], include_genes=["flat"], hierarchy=hierarchy)
        shown, _ = display_markers(scores, hierarchy, markers, adata_hierarchy.var_names, min_margin=-100)
        assert shown.empty

    def test_level_keys_map_table_levels(self, setup):
        scores, hierarchy, markers, var_names = setup
        renamed = markers.assign(level=markers["level"].map({"level_1": "L1", "level_2": "L2"}))
        shown, _ = display_markers(scores, hierarchy, renamed, var_names, level_keys={"L1": "level_1", "L2": "level_2"})
        assert len(shown) == 2

    def test_missing_scores_raise(self, setup):
        _, hierarchy, markers, var_names = setup
        with pytest.raises(ValueError, match="include_genes"):
            display_markers(pd.DataFrame(columns=["level", "node", "gene", "score", "det_in", "det_out"]),
                            hierarchy, markers, var_names)  # fmt: skip
