"""The routing table: one primary, hard gates hold, and the published page matches the code."""

import itertools
from pathlib import Path

import pytest

from cell_type_annotation.routing import AXES, evaluate, update_page

ROOT = Path(__file__).parents[1]
PAGE = ROOT / "skills" / "cell-type-annotation" / "references" / "method_selection.md"
PAN_BODY = {"azimuth", "scimilarity"}


def _primary(**axes):
    primaries = [r for r in evaluate(**axes) if r["status"] == "primary"]
    assert len(primaries) == 1
    return primaries[0]["method"]


def _contexts():
    values = {axis: [v for v, _ in vals] for axis, vals in AXES.items()}
    for combo in itertools.product(*values.values(), [False, True], [False, True]):
        *axes, atlas, celltypist = combo
        yield {**dict(zip(values, axes, strict=True)), "atlas": atlas, "celltypist_model": celltypist}


def test_page_matches_the_code(tmp_path):
    copy = tmp_path / PAGE.name
    copy.write_text(PAGE.read_text())
    assert not update_page(copy), f"{PAGE} is stale: regenerate with routing.update_page"


def test_exactly_one_primary_and_gates_hold():
    for ctx in _contexts():
        method = _primary(**ctx)
        if method in PAN_BODY:
            assert ctx["species"] == "human" and ctx["stage"] == "adult", ctx
            assert ctx["state"] != "tumor" and ctx["technology"] != "spatial_panel", ctx
        if ctx["technology"] == "spatial_panel":
            assert method not in {*PAN_BODY, "celltypist"}, ctx


def test_ranking():
    assert _primary(species="human", stage="adult", state="normal", obs_pred=True) == "obs_pred"
    assert _primary(species="mouse", stage="adult", profile="gpu", scanvi_model=True, atlas=True) == "scarches"
    assert _primary(species="mouse", stage="adult", profile="cpu", scanvi_model=True, atlas=True) == "reference_map"
    assert _primary(species="human", stage="adult", state="normal") == "azimuth"
    assert _primary(species="human", stage="adult", state="normal", celltypist_model=True) == "celltypist"
    assert _primary(species="human", stage="organoid", celltypist_model=True) == "celltypist"
    assert _primary(species="mouse", stage="developmental") == "census"


def test_unknown_species_is_not_reported_as_non_human():
    azimuth = next(r for r in evaluate(stage="adult") if r["method"] == "azimuth")
    assert azimuth["status"] == "disqualified" and "not established" in azimuth["reason"]


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="species"):
        evaluate(species="zebrafish")
    with pytest.raises(ValueError, match="Unknown routing input"):
        evaluate(has_atlas=True)
