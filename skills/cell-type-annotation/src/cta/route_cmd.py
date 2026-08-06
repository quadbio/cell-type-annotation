#!/usr/bin/env python
"""`cta route` — evaluate the cross-check routing tree for THIS dataset and machine.

Stage 3 step 4 runs this instead of reading a prose cascade: describe the query on the six
axes and it names the ONE primary cross-check, plus why every higher-ranked option was
skipped or refused. The ranking and gates live in `cta/routing.py`, which is also what
`reference/method_selection.md` is generated from — so the page a human reads and the answer
Claude acts on cannot disagree.

  cta route --species human --stage adult --technology dissociated --profile local_cpu
  cta route --species mouse --stage adult --has-atlas --profile local_gpu --has-scanvi-model
  cta route --emit-markdown          # the generated blocks of reference/method_selection.md

No heavy imports — this is a decision table, not an analysis step.
"""

from __future__ import annotations

import typer

from cta.io import eprint
from cta.routing import AXES, HOW_TO_DETERMINE, REFERENCES, ROUTES, default_context, evaluate

_STATUS_MARK = {"primary": "->", "fallback": "  ", "skipped": "  ", "disqualified": " x"}


def _emit_markdown() -> str:
    """Render the generated half of reference/method_selection.md (axes, tree, table)."""
    L: list[str] = []
    L.append("### The six axes\n")
    L.append("| Axis | Values | How you determine it |")
    L.append("| --- | --- | --- |")
    for axis, vals in AXES.items():
        vs = " · ".join(f"`{v}`" for v, _ in vals)
        L.append(f"| **{axis}** | {vs} | {HOW_TO_DETERMINE[axis]} |")
    refs = " · ".join(f"`{k}`" for k, _ in REFERENCES)
    L.append(f"| **references** (flags) | {refs} | {HOW_TO_DETERMINE['references']} |")

    L.append("\n### The ranking\n")
    L.append("Evaluated top-down. The first route whose preconditions hold **and** whose")
    L.append("disqualifiers are all false is the primary cross-check. A disqualifier beats rank.\n")
    L.append("| # | Cross-check | Use when | Why here |")
    L.append("| --- | --- | --- | --- |")
    for i, r in enumerate(ROUTES, start=1):
        L.append(f"| {i} | `{r.command}` | {r.when} | {r.why} |")

    L.append("\n### Disqualifiers (a hard refusal, whatever the rank)\n")
    L.append("| Cross-check | Refused when |")
    L.append("| --- | --- |")
    for r in ROUTES:
        items = [label for label, _ in r.disqualifiers]
        if items:
            L.append(f"| `{r.command}` | " + "; ".join(items) + " |")

    L.append("\n### Caveats that travel with the result\n")
    for r in ROUTES:
        if r.caveat:
            L.append(f"- **`{r.command}`** — {r.caveat}")

    L.append("\n### The tree\n")
    L.append("```mermaid")
    L.append("flowchart TD")
    L.append(
        '    A([Stage 3 step 4: pick ONE primary cross-check]) --> P{"targeted imaging panel?"}'
    )
    L.append(
        '    P -->|yes| PANEL["cta crosscheck reference-map<br/>cross-modality CCA, against your atlas<br/>or a Census slice if you have none<br/><i>pan-transcriptome models are DISQUALIFIED</i>"]'
    )
    L.append('    P -->|no| OBS{"obs already has a<br/>prediction you can vouch for?"}')
    L.append('    OBS -->|yes| R1["cta crosscheck obs-pred"]')
    L.append('    OBS -->|no| SC{"GPU profile AND a<br/>matched scANVI model?"}')
    L.append('    SC -->|yes| R2["cta reference map-scarches<br/>labels + calibrated uncertainty"]')
    L.append('    SC -->|no| AT{"a labelled reference<br/>AnnData of your own?"}')
    L.append('    AT -->|yes| R3["cta crosscheck reference-map"]')
    L.append('    AT -->|no| HU{"human AND adult<br/>AND non-tumour?"}')
    L.append(
        '    HU -->|yes| R4["cta crosscheck azimuth<br/>+ CL ids, calibrated conf, state signals"]'
    )
    L.append('    HU -->|no| CT{"tissue AND stage-matched<br/>CellTypist model?"}')
    L.append("    R4 --> CT")
    L.append('    CT -->|yes| R5["cta crosscheck celltypist"]')
    L.append('    CT -->|no| SM{"broad human adult query,<br/>>=27 GB RAM?"}')
    L.append('    SM -->|yes| R6["cta crosscheck scimilarity"]')
    L.append('    SM -->|no| R7["cta reference fetch-census<br/>-> cta crosscheck reference-map"]')
    L.append(
        "    R1 & R2 & R3 & R4 & R5 & R6 & R7 & PANEL --> Z([Reconcile against markers.<br/>Disagreement is SIGNAL.<br/>A second method ONLY to break a tie.])"
    )
    L.append("```")
    L.append("")
    L.append(
        "*(The mermaid tree is a reading aid for the common path; `cta route` is authoritative —"
    )
    L.append("both are generated from `src/cta/routing.py`.)*")
    return "\n".join(L)


def cli(
    species: str = typer.Option("unknown", help="human | mouse | other"),
    stage: str = typer.Option("unknown", help="adult | developmental | organoid | unknown"),
    state: str = typer.Option("unknown", help="normal | disease | tumor | unknown"),
    technology: str = typer.Option("dissociated", help="dissociated | spatial_wt | spatial_panel"),
    profile: str = typer.Option(
        "local_cpu", help="local_cpu | local_gpu | scheduler (from `cta detect-env`)"
    ),
    ram_gb: float = typer.Option(None, help="usable RAM in GB (from `cta detect-env`)"),
    has_obs_pred: bool = typer.Option(False, help="obs already carries a usable prediction column"),
    has_atlas: bool = typer.Option(False, help="a labelled reference AnnData is available"),
    has_scanvi_model: bool = typer.Option(False, help="a tissue-matched scANVI model is available"),
    has_celltypist_model: bool = typer.Option(
        False,
        help="a tissue- AND stage-matched CellTypist model exists (check `celltypist list` first)",
    ),
    emit_markdown: bool = typer.Option(
        False,
        "--emit-markdown",
        help="print the generated blocks of reference/method_selection.md and exit",
    ),
) -> None:
    """Pick the ONE primary cross-check for this query. A hypothesis, never a verdict."""
    if emit_markdown:
        print(_emit_markdown())
        return

    for name, value in (("species", species), ("stage", stage), ("state", state)):
        allowed = [v for v, _ in AXES[name]] + (["unknown"] if name != "technology" else [])
        if value not in allowed:
            raise SystemExit(
                f"ERROR: --{name} must be one of {sorted(set(allowed))}, got {value!r}"
            )
    for name, value in (("technology", technology), ("profile", profile)):
        allowed = [v for v, _ in AXES[name]]
        if value not in allowed:
            raise SystemExit(f"ERROR: --{name} must be one of {allowed}, got {value!r}")

    ctx = default_context()
    ctx.update(
        species=species,
        stage=stage,
        state=state,
        technology=technology,
        profile=profile,
        ram_gb=ram_gb,
    )
    ctx["refs"] = {
        "obs_pred": has_obs_pred,
        "atlas": has_atlas,
        "scanvi_model": has_scanvi_model,
        "celltypist_model": has_celltypist_model,
    }

    results = evaluate(ctx)
    primary = next((r for r in results if r["status"] == "primary"), None)

    if species == "unknown" or stage == "unknown":
        eprint(
            "!! species and/or stage are 'unknown'. Both gate the pan-body human models, and "
            "there is NO data probe for stage — establish them with the user (Stage 1) before "
            "acting on this routing.\n"
        )

    if primary:
        print(f"PRIMARY: {primary['command']}")
        print(f"  because: {primary['reason']}")
        if primary["caveat"]:
            print(f"  CAVEAT : {primary['caveat']}")
    else:
        print(
            "PRIMARY: none — no route applies. Consult the user before proceeding without a cross-check."
        )

    print("\nranked evaluation:")
    for r in results:
        print(
            f" {_STATUS_MARK[r['status']]} {r['rank']}. {r['command']:<52s} {r['status']:<13s} {r['reason']}"
        )

    print(
        "\nREMINDER: this selects ONE hypothesis to reconcile against the marker evidence. "
        "Claude + markers is the annotator; never let a cross-check overwrite a call. "
        "Run a second method only to break a tie. (reference/method_selection.md)"
    )
