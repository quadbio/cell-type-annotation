import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cell_type_annotation.checks import (
    cell_cycle_genes,
    confidence_proxy,
    negative_marker_flags,
    normalization_fingerprint,
    panel_gate,
    qc_deviation,
    validate_cl_ids,
)
from cell_type_annotation.markers import read_markers


@pytest.fixture()
def counts():
    rng = np.random.default_rng(0)
    depth = rng.integers(500, 5000, size=300)[:, None]
    return rng.poisson(depth * rng.dirichlet(np.ones(200), size=300)).astype(float)


class TestNormalizationFingerprint:
    def test_raw_counts(self, counts):
        result = normalization_fingerprint(sp.csr_matrix(counts))
        assert result["integer_valued"] and not result["ok"]

    def test_log1p_of_unnormalized_counts(self, counts):
        result = normalization_fingerprint(np.log1p(counts))
        assert not result["ok"]
        assert "un-normalized" in result["verdict"]

    def test_proper_lognorm(self, counts):
        lognorm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
        result = normalization_fingerprint(sp.csr_matrix(lognorm))
        assert result["ok"]

    def test_sparse_and_dense_agree(self, counts):
        lognorm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
        assert normalization_fingerprint(sp.csr_matrix(lognorm)) == normalization_fingerprint(lognorm)
        assert normalization_fingerprint(sp.csr_matrix(counts)) == normalization_fingerprint(counts)

    def test_scaled(self, counts):
        lognorm = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
        assert "scaled" in normalization_fingerprint(lognorm - lognorm.mean(0))["verdict"]


@pytest.fixture()
def markers():
    rows = [
        ("L1", "T cell", "CD3E", "pos"),
        ("L1", "T cell", "CD3D", "pos"),
        ("L1", "T cell", "MS4A1", "neg"),
        ("L1", "B cell", "MS4A1", "pos"),
        ("L1", "B cell", "OFF1", "pos"),
        ("L1", "B cell", "OFF2", "neg"),
    ]
    return read_markers(pd.DataFrame(rows, columns=["level", "node", "gene", "sign"]))


def test_panel_gate_marks_uncallable_candidates(markers):
    gate = panel_gate(markers, ["CD3E", "CD3D", "MS4A1"]).set_index("node")
    assert not gate.loc["T cell", "uncallable"]
    assert gate.loc["B cell", "uncallable"]
    assert gate.loc["B cell", "off_panel_pos"] == "OFF1"
    assert gate.loc["B cell", "off_panel_neg"] == "OFF2"


def test_panel_gate_keeps_its_columns_when_empty(markers):
    gate = panel_gate(markers.iloc[:0], ["CD3E"])
    assert gate.empty and "uncallable" in gate.columns


def test_negative_marker_flags_per_candidate(markers):
    pct = pd.DataFrame({"CD3E": [0.9, 0.0], "MS4A1": [0.3, 0.8]}, index=["0", "1"])
    flags = negative_marker_flags(pct, markers)
    assert flags[["cluster", "level", "node", "gene"]].values.tolist() == [
        ["0", "L1", "T cell", "MS4A1"],
        ["1", "L1", "T cell", "MS4A1"],
    ]
    assert flags["violation"].tolist() == [True, True]
    assert not negative_marker_flags(pct, markers, max_pct=0.5)["violation"].iloc[0]
    with pytest.raises(ValueError, match="missing detection"):
        negative_marker_flags(pct.assign(MS4A1=np.nan), markers)


def test_validate_cl_ids():
    problems = validate_cl_ids({"0": "CL:0000236", "1": "", "2": None, "3": "PCL:0001234", "4": "CL:000681"})
    assert len(problems) == 1 and "'4'" in problems[0]
    assert validate_cl_ids(pd.Series({"0": "cl:0000236"}))


def test_qc_deviation_flags_the_outlier():
    medians = pd.DataFrame({"total_counts": [1000, 1100, 950, 200], "flat": [1, 1, 1, 1]}, index=list("abcd"))
    z = qc_deviation(medians)
    assert z.loc["d", "total_counts"] < -3
    assert (z["flat"] == 0).all()


def test_qc_deviation_shows_a_lone_outlier_among_tied_medians():
    """pct_counts_mt is often exactly 0 in most clusters of cleaned nuclei; MAD is 0 there."""
    z = qc_deviation(pd.DataFrame({"pct_counts_mt": [0, 0, 0, 0, 0, 14.0]}, index=list("abcdef")))
    assert z.loc["f", "pct_counts_mt"] > 3
    assert (z.loc[list("abcde"), "pct_counts_mt"] == 0).all()


class TestConfidenceProxy:
    @pytest.fixture()
    def filtered(self):
        return pd.DataFrame(
            {
                "group": ["0"] * 6 + ["1"] * 2 + ["2"],
                "names": ["g1", "g2", "g3", "g4", "g5", "g6", "h1", "h2", np.nan],
                "logfoldchanges": [4, 3.5, 3, 3, 2, 1.5, 1.2, 0.3, 5.0],
            }
        )

    def test_bands(self, filtered):
        conf = confidence_proxy(filtered, ["0", "1", "2", "3"]).set_index("cluster")["suggested_conf"]
        assert conf["0"] == 0.9
        assert conf["1"] == 0.7

    def test_clusters_without_surviving_markers_get_the_base_band(self, filtered):
        conf = confidence_proxy(filtered, ["0", "2", "3"]).set_index("cluster")["suggested_conf"]
        assert conf["2"] == conf["3"] == 0.45  # 2 has only a rejected (blank-name) gene

    def test_agreement_adjusts(self, filtered):
        conf = confidence_proxy(filtered, ["1"], agree={"1": 3}).set_index("cluster")["suggested_conf"]
        assert conf["1"] == 0.8
        as_series = confidence_proxy(filtered, ["1"], agree=pd.Series({"1": 3})).set_index("cluster")
        assert as_series.loc["1", "suggested_conf"] == 0.8


def test_cell_cycle_genes_match_case_insensitively():
    s, g2m = cell_cycle_genes()
    assert len(s) == 43 and len(g2m) == 54
    s_mouse, g2m_mouse = cell_cycle_genes(["Mcm5", "Top2a", "Actb"])
    assert s_mouse == ["Mcm5"] and g2m_mouse == ["Top2a"]
