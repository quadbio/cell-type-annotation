"""The routing tree must have exactly ONE definition.

`src/cta/routing.py` is it. `reference/method_selection.md` renders it for humans and
`cta route` renders it for Claude mid-run — these tests fail if either drifts from the module,
so the page a human reads and the answer the skill acts on can never disagree. Same contract as
`tests/test_registry_sync.py`, which polices the command surface.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cta.cli import command_paths
from cta.route_cmd import _emit_markdown
from cta.routing import ROUTES, default_context, evaluate

DOC = Path(__file__).resolve().parents[1] / "reference" / "method_selection.md"
BEGIN = "<!-- BEGIN GENERATED: cta route --emit-markdown -->"
END = "<!-- END GENERATED -->"


def _generated_block() -> str:
    text = DOC.read_text()
    assert BEGIN in text and END in text, f"{DOC} is missing the generated-block markers"
    return text.split(BEGIN, 1)[1].split(END, 1)[0].strip()


def test_doc_generated_block_matches_module():
    """The committed doc must equal what `cta route --emit-markdown` produces right now."""
    assert _generated_block() == _emit_markdown().strip(), (
        "reference/method_selection.md is stale. Regenerate the block between the markers:\n"
        "    cta route --emit-markdown\n"
        "Do NOT hand-edit it — routing is defined only in src/cta/routing.py."
    )


def test_every_routed_command_exists_in_the_cli():
    """A route may only name commands the CLI actually has (`cta X -> cta Y` chains split)."""
    known = set(command_paths())
    for r in ROUTES:
        for part in r.command.split("->"):
            cmd = part.strip().removeprefix("cta ").strip()
            assert cmd in known, f"route {r.command!r} names unknown command {cmd!r}"


def test_every_routed_command_is_in_the_tool_registry():
    """Routing and the canonical cost table must cover the same tools."""
    registry = (DOC.parent / "tool_registry.md").read_text()
    for r in ROUTES:
        for part in r.command.split("->"):
            cmd = part.strip()
            assert f"`{cmd}`" in registry, (
                f"{cmd!r} is routed but has no reference/tool_registry.md row"
            )


def test_exactly_one_primary_for_any_context():
    """Evaluation must be deterministic: never two primaries, and never zero for a live query."""
    ctx = default_context()
    ctx.update(species="human", stage="adult", state="normal")
    results = evaluate(ctx)
    assert sum(r["status"] == "primary" for r in results) == 1


@pytest.mark.parametrize(
    ("axes", "must_not_be_primary"),
    [
        # The measured gates: pan-body human models are refused outside their training set.
        ({"species": "mouse", "stage": "adult"}, "cta crosscheck azimuth"),
        ({"species": "mouse", "stage": "adult"}, "cta crosscheck scimilarity"),
        ({"species": "human", "stage": "organoid"}, "cta crosscheck azimuth"),
        ({"species": "human", "stage": "developmental"}, "cta crosscheck azimuth"),
        ({"species": "human", "stage": "adult", "state": "tumor"}, "cta crosscheck azimuth"),
        (
            {"species": "human", "stage": "adult", "technology": "spatial_panel"},
            "cta crosscheck azimuth",
        ),
    ],
)
def test_pan_body_models_are_disqualified_out_of_distribution(axes, must_not_be_primary):
    ctx = default_context()
    ctx.update({"state": "normal", **axes})  # axes may override state (the tumour case)
    chosen = [r["command"] for r in evaluate(ctx) if r["status"] == "primary"]
    assert must_not_be_primary not in chosen, (
        f"{must_not_be_primary} must not be primary for {axes}"
    )


def test_a_reference_crosscheck_always_remains_available():
    """No query may end up with no cross-check at all — the Census fallback is unconditional."""
    for species in ("human", "mouse", "other"):
        for stage in ("adult", "developmental", "organoid"):
            ctx = default_context()
            ctx.update(species=species, stage=stage, state="normal")
            assert any(r["status"] == "primary" for r in evaluate(ctx)), (
                f"no primary cross-check for species={species} stage={stage}"
            )


def test_skill_md_points_at_the_doc_and_the_command():
    """SKILL.md must delegate routing, not restate it."""
    skill = (DOC.parent.parent / "SKILL.md").read_text()
    assert "reference/method_selection.md" in skill
    assert "cta route" in skill
    # The old prose cascade listed the branches inline; make sure it hasn't crept back.
    cascade_markers = re.findall(r"else \*\*(?:Reference-model|kNN|CellTypist|Census)", skill)
    assert not cascade_markers, (
        "SKILL.md appears to restate the cross-check cascade inline. Routing has exactly one "
        "home (src/cta/routing.py, rendered into reference/method_selection.md); SKILL.md must "
        "point at `cta route` instead."
    )
