"""Unit tests for `cta crosscheck azimuth` / `cta reference fetch-azimuth`.

These run in the `default` env, where panhumanpy is NOT installed — so they exercise only the
pure helpers, which is exactly why those must stay importable without the heavy dependency
(the panhumanpy import lives inside the command body, as it does for scimilarity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cta import fetch_azimuth_model as faz
from cta import run_azimuth as az


def test_leaf_splits_pipe_paths():
    assert az._leaf("Immune cell|Lymphoid cell|T cell") == "T cell"
    assert az._leaf("Adipocyte") == "Adipocyte"
    assert az._leaf("") == ""


def test_cl_lookup_prefers_exact_then_leaf():
    ids = {"Memory CD4 T cell": "CL:0000897", "T cell": "CL:0000084"}
    match = {"Memory CD4 T cell": "skos:exactMatch", "T cell": "skos:narrowMatch"}
    assert az._cl_for("Memory CD4 T cell", ids, match) == ("CL:0000897", "skos:exactMatch")
    # a pipe-path falls back to its leaf
    assert az._cl_for("Immune cell|T cell", ids, match) == ("CL:0000084", "skos:narrowMatch")
    # unmapped (e.g. `Unassigned`) yields no assertion at all
    assert az._cl_for("Unassigned", ids, match) == ("", "")


def test_cl_lookup_drops_the_meaningless_root_term():
    """`Doublet like cell` maps to CL:0000000 ('cell') — syntactically valid, semantically empty."""
    ids = {"Doublet like cell": "CL:0000000"}
    assert az._cl_for("Doublet like cell", ids, {}) == ("", "")


@pytest.mark.parametrize("layer", ["X", "lognorm", "logcounts", "data"])
def test_counts_layer_gate_refuses_anything_but_the_two_canonical_layers(layer):
    """The silent-corruption path: panhumanpy reads .X and GUESSES the normalization.

    Handing it the canonical `X` (the user's own log-norm, not necessarily CP10K) would be
    silently mis-scaled with no error, so only 'counts' and 'lognorm_cp10k' are accepted.
    """
    import anndata as ad

    a = ad.AnnData(X=np.ones((4, 3), dtype="float32"))
    with pytest.raises(SystemExit) as e:
        az._azimuth_input(a, layer)
    assert "--counts-layer must be one of" in str(e.value)


def test_counts_layer_gate_sets_normalization_override_only_for_the_precomputed_layer():
    import anndata as ad

    a = ad.AnnData(X=np.ones((4, 3), dtype="float32"))
    a.layers["counts"] = np.ones((4, 3), dtype="float32")
    a.layers["lognorm_cp10k"] = np.ones((4, 3), dtype="float32")
    _, override = az._azimuth_input(a, "counts")
    assert override is False, "raw counts: panhumanpy must do its own CP10K+log1p"
    _, override = az._azimuth_input(a, "lognorm_cp10k")
    assert override is True, "already exact CP10K+log1p: panhumanpy must leave it alone"


def test_counts_layer_gate_reports_a_missing_layer():
    import anndata as ad

    a = ad.AnnData(X=np.ones((4, 3), dtype="float32"))
    with pytest.raises(SystemExit) as e:
        az._azimuth_input(a, "counts")
    assert "canonicalize" in str(e.value)


def _meta(fine, broad, conf):
    return pd.DataFrame(
        {"azimuth_fine": fine, "azimuth_broad": broad, "final_level_confidence": conf}
    )


def test_state_table_schema_and_fractions():
    clusters = ["a", "a", "a", "a", "b", "b"]
    meta = _meta(
        fine=["T cell", az.INCONSISTENT, "T cell", "T cell", "B cell", az.INCONSISTENT],
        broad=["Immune cell"] * 6,
        conf=[0.9, 0.5, 0.8, 0.8, 0.6, 0.4],
    )
    broad = np.array(
        [
            "Immune cell",
            "Unassigned",
            "Doublet like cell",
            "Immune cell",
            "Immune cell",
            "Immune cell",
        ]
    )
    out = az._state_table(clusters, meta, broad)
    assert list(out.columns) == [
        "cluster",
        "n_cells",
        "frac_unassigned",
        "frac_doublet_like",
        "frac_inconsistent_hierarchy",
        "mean_conf",
    ]
    a = out.set_index("cluster").loc["a"]
    assert a["n_cells"] == 4
    assert a["frac_unassigned"] == pytest.approx(0.25)
    assert a["frac_doublet_like"] == pytest.approx(0.25)
    assert a["frac_inconsistent_hierarchy"] == pytest.approx(0.25)
    b = out.set_index("cluster").loc["b"]
    assert b["frac_inconsistent_hierarchy"] == pytest.approx(0.5)


def test_fetch_azimuth_constants_are_self_consistent():
    assert faz.DEFAULT_VERSION in faz.MODEL_URLS
    assert faz.MODEL_FILENAME == "inference_model.keras"
    assert faz.CACHE_DIR.name == "panhumanpy"
    for url in faz.MODEL_URLS.values():
        assert faz.ZENODO_RECORD in url


def test_fetch_azimuth_constants_match_panhumanpy():
    """Guard the MIRRORED constants against upstream drift.

    Skipped in `default` (panhumanpy lives in the `azimuth` env); runs when the suite is
    executed there — which is why pytest is a dependency of that feature.
    """
    tools = pytest.importorskip("panhumanpy.ANNotate_tools")
    assert faz.MODEL_URLS == tools.MODEL_URLS
    assert faz.CACHE_DIR == tools.CACHE_DIR
