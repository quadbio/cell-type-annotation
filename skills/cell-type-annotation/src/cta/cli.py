"""`cta` — the grouped Typer CLI for the cell-type-annotation skill.

Commands are organized into groups (Sopa-style): `cta markers compute`,
`cta crosscheck reference-map`, etc. Each leaf command is a `cli()` function defined in its
own module (or, for `celltypist`, a nested `typer.Typer()` app), with heavy imports done
lazily inside the command body — so `cta --help` and tab-completion are instant and don't
pull scanpy/torch.

`GROUPS` is the single source of truth for the command surface. It is mirrored by
`reference/tool_registry.md` (the canonical step→tool table) and checked by
`tests/test_registry_sync.py`, so docs, CLI, and package can't silently drift.
"""

from __future__ import annotations

import typer

from cta import (
    assemble_markers,
    build_report,
    canonicalize,
    cluster_confidence,
    compute_markers,
    crosscheck_obs_pred,
    detect_env,
    fetch_azimuth_model,
    fetch_census_reference,
    fetch_scimilarity_model,
    inspect_adata,
    map_query_scarches,
    profile_clusters,
    profile_markers,
    qc,
    reference_map,
    route_cmd,
    run_azimuth,
    run_celltypist,
    run_scimilarity,
    scaffold_cards,
    score_signatures,
    subcluster,
    train_reference_model,
    write_back,
)

# group -> {command name: leaf command}. group None = top-level. A leaf is either a `cli`
# function or a nested Typer app (celltypist's list/predict/markers).
GROUPS: dict[str | None, dict[str, object]] = {
    None: {"detect-env": detect_env.cli, "route": route_cmd.cli},
    "data": {"inspect": inspect_adata.cli, "canonicalize": canonicalize.cli, "qc": qc.cli},
    "markers": {
        "assemble": assemble_markers.cli,
        "compute": compute_markers.cli,
        "profile": profile_markers.cli,
        "score": score_signatures.cli,
    },
    "crosscheck": {
        "obs-pred": crosscheck_obs_pred.cli,
        "reference-map": reference_map.cli,
        "celltypist": run_celltypist.app,
        "azimuth": run_azimuth.cli,
        "scimilarity": run_scimilarity.cli,
    },
    "reference": {
        "fetch-azimuth": fetch_azimuth_model.cli,
        "fetch-census": fetch_census_reference.cli,
        "fetch-scimilarity": fetch_scimilarity_model.cli,
        "train-model": train_reference_model.cli,
        "map-scarches": map_query_scarches.cli,
    },
    "cluster": {
        "subcluster": subcluster.cli,
        "profile": profile_clusters.cli,
        "confidence": cluster_confidence.cli,
    },
    "report": {
        "scaffold-cards": scaffold_cards.cli,
        "build": build_report.cli,
        "write-back": write_back.cli,
    },
}

_GROUP_HELP = {
    "data": "Inspect / slim the AnnData (Stage 1).",
    "markers": "Marker evidence: priors, DE/specificity, panel profiling, signature scoring.",
    "crosscheck": "Automated cross-checks — hypotheses to reconcile, never verdicts.",
    "reference": "Reference side: fetch/train a reference model, map the query onto it.",
    "cluster": "Sub-cluster and investigate clusters; per-cluster confidence proxy.",
    "report": "Evidence cards, the per-level HTML report, and the final write-back.",
}


def _build_app() -> typer.Typer:
    app = typer.Typer(
        add_completion=False,
        help="Evidence-driven, compute-environment-aware cell-type annotation.",
    )
    for group, commands in GROUPS.items():
        target = (
            app if group is None else typer.Typer(no_args_is_help=True, help=_GROUP_HELP.get(group))
        )
        for name, cmd in commands.items():
            if isinstance(cmd, typer.Typer):
                target.add_typer(cmd, name=name)
            else:
                target.command(name)(cmd)
        if group is not None:
            app.add_typer(target, name=group)

    @app.callback(invoke_without_command=True)
    def _root(
        ctx: typer.Context,
        version: bool = typer.Option(False, "--version", help="Show the cta version and exit."),
    ):
        if version:
            from cta import __version__

            typer.echo(__version__)
            raise typer.Exit()
        if ctx.invoked_subcommand is None:
            typer.echo(ctx.get_help())
            raise typer.Exit()

    return app


app = _build_app()


def command_paths() -> list[str]:
    """Full command paths, e.g. 'detect-env', 'markers compute', 'crosscheck celltypist'."""
    return [
        name if group is None else f"{group} {name}"
        for group, commands in GROUPS.items()
        for name in commands
    ]


def main() -> None:
    app()


if __name__ == "__main__":
    main()
