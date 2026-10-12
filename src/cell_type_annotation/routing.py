"""Which one automated cross-check to run, given the query and the machine.

The ranking and every gate live here as data. ``references/method_selection.md`` renders its tables
from :func:`to_markdown`, and a test fails when the committed page drifts from it, so the page a
human reads and the answer :func:`evaluate` gives cannot disagree.

Whatever is routed here is one hypothesis to reconcile against the marker evidence, never a verdict.
Run exactly one primary cross-check, and a second only to break a tie.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

#: The axes a query is described on, with their allowed values.
AXES: dict[str, tuple[tuple[str, str], ...]] = {
    "species": (
        ("human", "Homo sapiens"),
        ("mouse", "Mus musculus"),
        ("other", "any other organism"),
        ("unknown", "not established yet: ask before routing"),
    ),
    "stage": (
        ("adult", "adult / postnatal primary tissue"),
        ("developmental", "fetal, embryonic, or otherwise developing primary tissue"),
        ("organoid", "organoid / iPSC-derived / in-vitro differentiated"),
        ("unknown", "not established yet: ask before routing"),
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
        ("cpu", "no CUDA GPU reachable from where the step runs"),
        ("gpu", "an NVIDIA CUDA GPU is reachable, here or through the scheduler"),
    ),
}

#: Reference assets, as independent yes/no flags.
REFERENCES: tuple[tuple[str, str], ...] = (
    ("obs_pred", "`obs` already carries a label-transfer or classifier prediction"),
    ("atlas", "a labelled reference AnnData for this tissue and context"),
    ("scanvi_model", "a tissue-matched scANVI model (trained, or from the scvi-tools hub)"),
    ("celltypist_model", "a tissue- and stage-matched pretrained CellTypist model"),
)

#: How each axis is established.
HOW_TO_DETERMINE: dict[str, str] = {
    "species": "Stage 1 intake; corroborated by the gene-ID convention.",
    "stage": "Stage 1 intake. There is no data probe for this: ask, do not infer from tissue.",
    "state": "Stage 1 intake for the design; QC deviation per cluster for quality and cycling.",
    "technology": "A few hundred genes in `var` means a targeted panel; confirm the platform with the user.",
    "profile": "Where the cross-check will run, including a GPU job submitted through the scheduler.",
    "references": "`obs` listing; `celltypist.models.get_models_index(force_update=True)` filtered to the tissue.",
}


@dataclass(frozen=True)
class Route:
    """One candidate cross-check: what it is, when it applies, and when it must not."""

    method: str
    #: How to run it, in a few words; the recipe is in ``references/method_selection.md``.
    name: str
    #: The situation this route is for.
    when: str
    #: Predicate over a context; true when the preconditions hold.
    applies: Callable[[dict], bool] = field(repr=False)
    #: ``(label, predicate)``; a true predicate refuses the route whatever its rank.
    disqualifiers: tuple[tuple[str, Callable[[dict], bool]], ...] = field(default=(), repr=False)
    why: str = ""
    #: Travels with the result into the evidence cards.
    caveat: str = ""


def _human_adult_normal(c: dict) -> bool:
    return c["species"] == "human" and c["stage"] == "adult" and c["state"] in ("normal", "disease")


#: Shared by Azimuth and SCimilarity, so the two pan-body human models cannot drift apart.
_PAN_BODY_DISQUALIFIERS: tuple[tuple[str, Callable[[dict], bool]], ...] = (
    (
        "species not human, or not established (the model has a fixed human gene space)",
        lambda c: c["species"] != "human",
    ),
    (
        "developmental / organoid query (trained on adult tissue; not designed for developmental stages)",
        lambda c: c["stage"] in ("developmental", "organoid"),
    ),
    ("tumour expected (malignant cells were excluded from training)", lambda c: c["state"] == "tumor"),
    (
        "stage or state not established (no data probe for either: ask before routing)",
        lambda c: "unknown" in (c["stage"], c["state"]),
    ),
    (
        "targeted imaging panel (missing panel genes are zero-filled, producing confident garbage)",
        lambda c: c["technology"] == "spatial_panel",
    ),
)

#: The ranking. The first route whose preconditions hold and whose disqualifiers are all false is
#: the primary cross-check; everything below it is fallback.
ROUTES: tuple[Route, ...] = (
    Route(
        method="obs_pred",
        name="existing `obs` prediction",
        when="`obs` already carries a prediction that fits this level",
        applies=lambda c: c["refs"].get("obs_pred", False),
        disqualifiers=(
            (
                "the column's provenance is unknown, or the method that produced it is itself "
                "disqualified here (species / stage / tissue / panel)",
                lambda c: False,  # not checkable from the context; see the caveat
            ),
        ),
        why="Free and already reconciled with the object: no recompute, no new assumptions.",
        caveat=(
            "It summarises someone else's prediction and inherits that method's disqualifiers. "
            "Establish what produced the column before trusting it; if you cannot, treat this route "
            "as unavailable."
        ),
    ),
    Route(
        method="scarches",
        name="scANVI / scArches mapping",
        when="a GPU AND a well-matched scANVI reference model (trained or from the hub)",
        applies=lambda c: c["profile"] == "gpu" and c["refs"].get("scanvi_model", False),
        disqualifiers=(("no CUDA GPU reachable", lambda c: c["profile"] == "cpu"),),
        why=(
            "The strongest option when it fits: labels with calibrated per-cell uncertainty, in a "
            "reference latent space built for this tissue. A well-matched, curated reference beats "
            "any generic model."
        ),
        caveat="Stochastic: seed it, and still report predictions as a hypothesis.",
    ),
    Route(
        method="reference_map",
        name="CellMapper on your reference",
        when="a labelled reference AnnData of your own (the user's atlas)",
        applies=lambda c: c["refs"].get("atlas", False),
        disqualifiers=(
            (
                "fewer than 30 genes shared with the query after symbol alignment",
                lambda c: False,  # checked when the mapping runs
            ),
        ),
        why=(
            "The user's own curated atlas for their tissue and context is usually the strongest "
            "reference available, and CellMapper builds the joint embedding itself, including "
            "cross-modality CCA for a targeted panel."
        ),
        caveat="On a targeted spatial panel this is the preferred route (cross-modality CCA).",
    ),
    Route(
        method="celltypist",
        name="CellTypist",
        when="a tissue- AND stage-matched pretrained model exists",
        applies=lambda c: c["refs"].get("celltypist_model", False),
        disqualifiers=(
            (
                "no tissue- and stage-matched model: omit CellTypist rather than force an "
                "ill-fitting one (a mismatched model collapses, it does not degrade gracefully)",
                lambda c: not c["refs"].get("celltypist_model", False),
            ),
            ("targeted imaging panel (most model genes missing)", lambda c: c["technology"] == "spatial_panel"),
        ),
        why=(
            "A genuinely matched model is a strong, nearly free cross-check trained on exactly this "
            "tissue and stage, so it outranks a pan-body model. The gate is the whole point: stage "
            "matters as much as tissue."
        ),
        caveat="Scale-sensitive: feed it log1p of counts normalized to 10,000 per cell.",
    ),
    Route(
        method="azimuth",
        name="pan-human Azimuth",
        when="human, adult/postnatal, non-tumour, whole-transcriptome, and no matched reference or model above",
        applies=_human_adult_normal,
        disqualifiers=_PAN_BODY_DISQUALIFIERS,
        why=(
            "One supervised model spanning 23 human tissues, so a single reference covers a broad "
            "or multi-tissue query that no tissue-specific model does. It is also the only "
            "cross-check here that returns a Cell Ontology id per cluster, a calibrated confidence, "
            "and trained `Unassigned` / `Doublet like cell` classes that feed the `state` axis. "
            "CPU-only and cheap (~50-60k cells/min, 84 MB model)."
        ),
        caveat=(
            "Two structural limits. (1) It cannot be constrained to a tissue: it predicts over the "
            "full pan-body space, so on a single-organ query it can return a type from another "
            "organ; check the winning label against the tissue. (2) For lymph node and spleen its "
            "training labels came from the CellTypist organ atlases, so on immune tissue it is not "
            "independent of CellTypist. Its CL ids, calibrated confidence and state signals are useful "
            "however you weigh its labels, which also makes it a good tie-break below a matched model."
        ),
    ),
    Route(
        method="scimilarity",
        name="SCimilarity",
        when="broad / multi-tissue human adult query with no matched reference or model",
        applies=_human_adult_normal,
        disqualifiers=_PAN_BODY_DISQUALIFIERS
        + (("under ~27 GB of usable RAM", lambda c: c["ram_gb"] is not None and c["ram_gb"] < 27),),
        why=(
            "An option for broad / multi-tissue human queries where no tissue-matched model exists. "
            "Ranked below Azimuth because it costs far more (9 GB bundle, 16-27 GB RAM) and returns "
            "a flat label space with no ontology ids, no calibrated confidence and no state signals."
        ),
        caveat="Zero-shot foundation model: an additional opinion, never a primary annotator.",
    ),
    Route(
        method="census",
        name="Census slice, then CellMapper",
        when="nothing above fits: build a reference instead of going without one",
        applies=lambda c: True,
        disqualifiers=(("offline (Census is a remote TileDB store)", lambda c: False),),
        why="Keeps a reference cross-check alive for any species and tissue, so no level goes unchecked.",
        caveat=(
            "Census `cell_type` is author-supplied and heterogeneous across datasets, so a "
            "transferred label is a coarser, noisier hypothesis than one from a curated atlas."
        ),
    ),
)

STATUSES = ("primary", "fallback", "skipped", "disqualified")


def default_context() -> dict:
    """A context with everything unknown, so every axis has to be supplied deliberately."""
    return {
        "species": "unknown",
        "stage": "unknown",
        "state": "unknown",
        "technology": "dissociated",
        "profile": "cpu",
        "ram_gb": None,
        "refs": dict.fromkeys((k for k, _ in REFERENCES), False),
    }


def evaluate(ctx: dict | None = None, /, **axes) -> list[dict]:
    """Score every route against a query, in ranking order.

    Parameters
    ----------
    ctx
        A context as returned by :func:`default_context`. Keyword arguments update it, and any
        reference flag (``atlas=True``) goes into ``refs``.
    **axes
        Axis values (``species="human"``), ``ram_gb``, or reference flags.

    Returns
    -------
    One dict per route, ``{method, name, rank, status, reason, why, caveat}``. ``status`` is
    ``primary`` (the first route that passes), ``fallback`` (passes but outranked), ``skipped``
    (preconditions not met) or ``disqualified`` (a hard gate refused it). Disqualifiers are
    checked first, so a refusal is always reported as such.

    Raises
    ------
    ValueError
        If an axis value or a keyword is not recognized.

    Examples
    --------
    >>> primary = next(r for r in evaluate(species="human", stage="adult", state="normal") if r["status"] == "primary")
    """
    c = default_context() if ctx is None else {**ctx, "refs": dict(ctx["refs"])}
    refs = {k for k, _ in REFERENCES}
    for key, value in axes.items():
        if key in refs:
            c["refs"][key] = bool(value)
        elif key in AXES or key == "ram_gb":
            c[key] = value
        else:
            raise ValueError(f"Unknown routing input {key!r}; expected one of {[*AXES, 'ram_gb', *refs]}")
    for axis, values in AXES.items():
        allowed = [v for v, _ in values]
        if c[axis] not in allowed:
            raise ValueError(f"{axis}={c[axis]!r} is not one of {allowed}")

    out, chosen = [], False
    for rank, route in enumerate(ROUTES, start=1):
        blocked = [label for label, refuses in route.disqualifiers if refuses(c)]
        if blocked:
            status, reason = "disqualified", "; ".join(blocked)
        elif not route.applies(c):
            status, reason = "skipped", f"preconditions not met ({route.when})"
        elif not chosen:
            status, reason, chosen = "primary", route.when, True
        else:
            status, reason = "fallback", "outranked by a higher route that also applies"
        out.append(
            {
                "method": route.method,
                "name": route.name,
                "rank": rank,
                "status": status,
                "reason": reason,
                "why": route.why,
                "caveat": route.caveat,
            }
        )
    return out


def to_markdown() -> str:
    """The generated block of ``references/method_selection.md``: axes, ranking, disqualifiers, caveats."""
    lines = ["### Axes\n", "| Axis | Values | How you determine it |", "| --- | --- | --- |"]
    for axis, values in AXES.items():
        lines.append(f"| **{axis}** | {' · '.join(f'`{v}`' for v, _ in values)} | {HOW_TO_DETERMINE[axis]} |")
    refs = " · ".join(f"`{k}`" for k, _ in REFERENCES)
    lines.append(f"| **references** (flags) | {refs} | {HOW_TO_DETERMINE['references']} |")

    lines += [
        "\n### Ranking\n",
        "Evaluated top-down. The first route whose preconditions hold and whose disqualifiers are all",
        "false is the primary cross-check. A disqualifier beats rank.\n",
        "| # | Method | Use when | Why here |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {i} | `{r.method}`: {r.name} | {r.when} | {r.why} |" for i, r in enumerate(ROUTES, start=1)]

    lines += ["\n### Disqualifiers\n", "| Method | Refused when |", "| --- | --- |"]
    lines += [
        f"| `{r.method}` | {'; '.join(label for label, _ in r.disqualifiers)} |" for r in ROUTES if r.disqualifiers
    ]

    lines.append("\n### Caveats that travel with the result\n")
    lines += [f"- **`{r.method}`**: {r.caveat}" for r in ROUTES if r.caveat]
    return "\n".join(lines)


_GENERATED = re.compile(r"(<!-- BEGIN GENERATED -->\n).*?(\n<!-- END GENERATED -->)", re.S)


def update_page(path: str | PathLike) -> bool:
    """Regenerate the block between the ``GENERATED`` markers of the page at ``path``.

    Returns
    -------
    Whether the page changed.
    """
    path = Path(path)
    text = path.read_text()
    if not _GENERATED.search(text):
        raise ValueError(f"{path} has no <!-- BEGIN GENERATED --> ... <!-- END GENERATED --> block")
    new = _GENERATED.sub(lambda m: m[1] + to_markdown() + m[2], text)
    path.write_text(new)
    return new != text
