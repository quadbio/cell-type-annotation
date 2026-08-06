#!/usr/bin/env python
"""THE single source of truth for cross-check method selection.

Stage 3 step 4 asks one question: *given this dataset and this machine, which ONE automated
cross-check do I run?* Before this module that answer lived as prose in SKILL.md, with the
species / stage / technology conditions scattered across three other reference docs — so the
docs, the tool registry and the actual behaviour could (and did) drift apart.

Now the ranking and every gating condition live HERE, as data:

  * `cta route` evaluates it and tells you (and Claude, mid-run) what to run and why not the
    rest — see `cta/route_cmd.py`.
  * `reference/method_selection.md` renders its tree and table FROM this module, between
    generated-block markers, with `tests/test_method_selection_sync.py` failing if the
    committed doc drifts from what this module would emit.
  * `tests/test_method_selection_sync.py` also pins the disqualifiers themselves, so the
    measured gates (non-human, developmental/organoid, tumour, targeted panel) cannot be
    loosened by accident.

So there is exactly one definition, and the human-facing page and the machine-facing command
are two renderings of it. Change the routing here; never in the prose.

WHAT THIS MODULE DOES NOT DECIDE. Claude + marker evidence is the annotator on every path.
Everything routed here is ONE hypothesis to reconcile against markers — never a verdict, never
a silent overwrite (`reference/automated_tools.md`). Run exactly one primary cross-check; a
second only to break a tie. Gates A and B are unchanged by which branch you take.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# --- the six axes ---------------------------------------------------------------------------
# Values are the vocabulary `cta route` accepts and the doc renders. Keep them short and
# unambiguous; every one must be answerable from Stage-1 intake or a `cta` probe.
AXES: dict[str, tuple[tuple[str, str], ...]] = {
    "species": (
        ("human", "Homo sapiens"),
        ("mouse", "Mus musculus"),
        ("other", "any other organism"),
    ),
    "stage": (
        ("adult", "adult / postnatal primary tissue"),
        ("developmental", "fetal, embryonic, or otherwise developing primary tissue"),
        ("organoid", "organoid / iPSC-derived / in-vitro differentiated"),
        ("unknown", "not established yet — ask before routing"),
    ),
    "state": (
        ("normal", "healthy / untreated"),
        ("disease", "non-cancer disease or perturbation"),
        ("tumor", "malignant cells expected"),
        ("unknown", "not established yet"),
    ),
    "technology": (
        ("dissociated", "scRNA-seq / snRNA-seq, whole transcriptome"),
        ("spatial_wt", "whole-transcriptome spatial (Visium / VisiumHD), segmented"),
        ("spatial_panel", "targeted imaging panel (MERSCOPE / Xenium / CosMx), 100s of genes"),
    ),
    "profile": (
        ("local_cpu", "CPU / Apple-Silicon; no CUDA GPU reachable here"),
        ("local_gpu", "an NVIDIA CUDA GPU is reachable in this process"),
        ("scheduler", "a batch scheduler exists but no GPU is reachable here"),
    ),
}

#: Reference assets, as independent yes/no flags (several can hold at once).
REFERENCES: tuple[tuple[str, str], ...] = (
    ("obs_pred", "adata.obs already carries a label-transfer / classifier prediction"),
    ("atlas", "a labelled reference AnnData for this tissue/context"),
    ("scanvi_model", "a tissue-matched scANVI model (trained, or from the scvi-tools hub)"),
    ("celltypist_model", "a tissue- AND stage-matched pretrained CellTypist model"),
)

#: How each axis is established — the doc renders this, so nobody has to guess.
HOW_TO_DETERMINE: dict[str, str] = {
    "species": "Stage-1 intake (`context.md`); corroborated by the gene-ID convention in `cta data inspect`.",
    "stage": "Stage-1 intake — there is NO data probe for this. Ask; do not infer from tissue.",
    "state": "Stage-1 intake for the design; `cta data qc` -> `cta cluster profile` for per-cluster quality/cycling.",
    "technology": "`cta data inspect` reports a `targeted_panel` flag; confirm the platform with the user.",
    "profile": "`cta detect-env`, then CONFIRM with the user (a scheduler on PATH is ambiguous).",
    "references": "`cta data inspect` obs listing; `cta crosscheck celltypist list --contains <tissue>`.",
}


@dataclass(frozen=True)
class Route:
    """One candidate cross-check: what it is, when it applies, and when it must not."""

    command: str
    #: One line: the situation this route is for.
    when: str
    #: Predicate over a resolved context; True = preconditions met.
    applies: Callable[[dict], bool] = field(repr=False)
    #: (label, predicate) — a TRUE predicate means this route is REFUSED, whatever its rank.
    #: Disqualifiers beat preconditions and beat rank; they are the safety rail.
    disqualifiers: tuple[tuple[str, Callable[[dict], bool]], ...] = field(default=(), repr=False)
    #: Why it sits where it sits.
    why: str = ""
    #: A caveat that travels with the result — printed by `cta route`, rendered in the doc.
    caveat: str = ""


def _human_pan_body_ok(c: dict) -> bool:
    """Shared gate for the two pan-body HUMAN models (Azimuth, SCimilarity)."""
    return c["species"] == "human" and c["stage"] == "adult" and c["state"] != "tumor"


#: Pan-body human models are refused outside their training distribution. Written once and
#: shared so Azimuth and SCimilarity can never drift apart on the species/stage/tumour gate.
_PAN_BODY_DISQUALIFIERS: tuple[tuple[str, Callable[[dict], bool]], ...] = (
    ("non-human query (the model has a fixed human gene space)", lambda c: c["species"] != "human"),
    (
        "developmental / organoid query (trained on adult tissue; not designed for developmental stages)",
        lambda c: c["stage"] in ("developmental", "organoid"),
    ),
    (
        "tumour expected (malignant cells were excluded from training)",
        lambda c: c["state"] == "tumor",
    ),
    (
        "targeted imaging panel (missing panel genes are zero-filled, producing confident garbage)",
        lambda c: c["technology"] == "spatial_panel",
    ),
)


#: THE RANKING. Order is the routing order: the first route whose preconditions hold and whose
#: disqualifiers are all false becomes the PRIMARY cross-check. Everything below is fallback.
ROUTES: tuple[Route, ...] = (
    Route(
        command="cta crosscheck obs-pred",
        when="`obs` already carries a prediction that fits this level",
        applies=lambda c: c["refs"].get("obs_pred", False),
        disqualifiers=(
            (
                "the existing column's provenance is unknown, or the method that produced it is "
                "itself disqualified here (species / stage / tissue / panel)",
                lambda c: False,  # cannot be auto-checked — Claude must confirm; see caveat
            ),
        ),
        why="Free and already reconciled with the object — no recompute, no new assumptions.",
        caveat=(
            "It summarises SOMEONE ELSE'S prediction and silently inherits that method's "
            "disqualifiers. Establish what produced the column before trusting it; if you "
            "cannot, treat this route as unavailable."
        ),
    ),
    Route(
        command="cta reference map-scarches",
        when="a GPU profile AND a well-matched scANVI reference model (trained or from the hub)",
        applies=lambda c: (
            c["profile"] in ("local_gpu", "scheduler") and c["refs"].get("scanvi_model", False)
        ),
        disqualifiers=(
            ("no CUDA GPU reachable from this process", lambda c: c["profile"] == "local_cpu"),
        ),
        why=(
            "The strongest option when it fits: labels WITH calibrated per-cell uncertainty, in a "
            "reference latent space built for this tissue. A well-matched, curated reference beats "
            "any generic model."
        ),
        caveat="Stochastic — seed it and still report predictions as a hypothesis.",
    ),
    Route(
        command="cta crosscheck reference-map",
        when="a labelled reference AnnData of your own (the user's atlas, or a Census slice)",
        applies=lambda c: c["refs"].get("atlas", False),
        disqualifiers=(
            (
                "fewer than MIN_SHARED_GENES shared genes after symbol alignment",
                lambda c: False,  # enforced in reference_map.py at run time
            ),
        ),
        why=(
            "The user's own curated atlas for their tissue/context is usually the strongest "
            "reference available (`reference/knowledge_sources.md`) — and CellMapper builds the "
            "joint embedding itself, including cross-modality CCA for a targeted panel."
        ),
        caveat="On a targeted spatial panel this is the PREFERRED route (cross-modality CCA).",
    ),
    Route(
        command="cta crosscheck azimuth",
        when="human, adult/postnatal, non-tumour, whole-transcriptome — and no matched reference above",
        applies=_human_pan_body_ok,
        disqualifiers=_PAN_BODY_DISQUALIFIERS,
        why=(
            "One supervised model spanning 23 human tissues, so a single reference covers a broad "
            "or multi-tissue query that no one tissue-specific model does. It is also the only "
            "cross-check here that returns a Cell Ontology id per cluster, a CALIBRATED confidence, "
            "and trained `Unassigned` / `Doublet like cell` classes that feed the `state` axis. "
            "CPU-only and cheap (~50-60k cells/min, 84 MB model), so it costs little to consult."
        ),
        caveat=(
            "Two structural limits to weigh its labels against. (1) It CANNOT be constrained to a "
            "tissue — it always predicts over the full pan-body space, so on a single-organ query "
            "it can return a type belonging to a different organ; check the winning label against "
            "the tissue in context.md. (2) Its training labels were transferred from "
            "tissue-specific references, and for lymph node and spleen those were the CellTypist "
            "organ atlases — so on immune tissue it is NOT an independent second opinion relative "
            "to CellTypist. Where a genuinely tissue- AND stage-matched CellTypist or scANVI model "
            "exists, treat that as the stronger reference and use it as the tie-break. Its CL ids, "
            "calibrated confidence and state signals are useful independently of how you weigh its "
            "labels."
        ),
    ),
    Route(
        command="cta crosscheck celltypist",
        when="a tissue- AND stage-matched pretrained model exists (`celltypist list` first)",
        applies=lambda c: c["refs"].get("celltypist_model", False),
        disqualifiers=(
            (
                "no tissue- AND stage-matched model — OMIT CellTypist rather than force an "
                "ill-fitting one (a mismatched model collapses, it does not degrade gracefully)",
                lambda c: not c["refs"].get("celltypist_model", False),
            ),
            (
                "targeted imaging panel (most model genes missing)",
                lambda c: c["technology"] == "spatial_panel",
            ),
        ),
        why=(
            "When a genuinely matched model exists it is a strong, nearly free cross-check trained "
            "on exactly this tissue. The gate is the whole point: stage matters as much as tissue, "
            "and a mismatched model collapses rather than degrading gracefully."
        ),
        caveat="Scale-sensitive; `predict` rebuilds log1p-CP10K itself from the canonical counts.",
    ),
    Route(
        command="cta crosscheck scimilarity",
        when="broad / multi-tissue human adult query with no matched reference or model",
        applies=_human_pan_body_ok,
        disqualifiers=_PAN_BODY_DISQUALIFIERS
        + (("under ~27 GB of usable RAM", lambda c: c["ram_gb"] is not None and c["ram_gb"] < 27),),
        why=(
            "A long-standing option for broad / multi-tissue human queries where no tissue-matched "
            "model exists. Kept below Azimuth because it costs far more (9 GB bundle, 16-27 GB RAM) "
            "and returns a flat label space with no ontology ids, no calibrated confidence and no "
            "state signals — not because either has been shown to annotate better."
        ),
        caveat="Zero-shot foundation model — an additional opinion, never a primary annotator.",
    ),
    Route(
        command="cta reference fetch-census -> cta crosscheck reference-map",
        when="nothing above fits — build a reference instead of going without one",
        applies=lambda c: True,
        disqualifiers=(("offline (Census is a remote TileDB store)", lambda c: False),),
        why="Keeps a reference cross-check alive for ANY species/tissue, so no level goes unchecked.",
        caveat=(
            "Census `cell_type` is author-supplied and heterogeneous across contributing datasets, "
            "so a transferred label is a coarser/noisier hypothesis than one from a single curated "
            "atlas — corroborating signal, not ground truth."
        ),
    ),
)


def default_context() -> dict:
    """A context with everything unknown — every axis must be supplied deliberately."""
    return {
        "species": "unknown",
        "stage": "unknown",
        "state": "unknown",
        "technology": "dissociated",
        "profile": "local_cpu",
        "ram_gb": None,
        "refs": {k: False for k, _ in REFERENCES},
    }


def evaluate(ctx: dict) -> list[dict]:
    """Score every route against `ctx`, in ranking order.

    Returns one dict per route: ``{command, rank, status, reason, why, caveat}`` where status is
    ``"primary"`` (the first route that passes), ``"fallback"`` (would pass, but outranked),
    ``"skipped"`` (preconditions not met) or ``"disqualified"`` (a hard gate refused it).
    Disqualification is checked BEFORE preconditions, so a refusal is always reported as such.
    """
    out: list[dict] = []
    chosen = False
    for rank, r in enumerate(ROUTES, start=1):
        blocked = [label for label, pred in r.disqualifiers if pred(ctx)]
        if blocked:
            status, reason = "disqualified", "; ".join(blocked)
        elif not r.applies(ctx):
            status, reason = "skipped", f"preconditions not met ({r.when})"
        elif not chosen:
            status, reason = "primary", r.when
            chosen = True
        else:
            status, reason = "fallback", "outranked by a higher route that also applies"
        out.append(
            {
                "command": r.command,
                "rank": rank,
                "status": status,
                "reason": reason,
                "why": r.why,
                "caveat": r.caveat,
            }
        )
    return out
