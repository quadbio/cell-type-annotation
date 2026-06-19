"""`cta` command-line interface — a thin dispatcher over the tool modules.

Each subcommand maps to a module in this package that owns its own argparse `main()`.
The dispatcher rewrites `sys.argv` and delegates, so every tool keeps its exact flags —
including aliases like `--adata`/`--query` and `celltypist`'s `list`/`predict`/`markers`
sub-subcommands — with zero re-plumbing. Run `cta <command> --help` for a command's
options, or `cta` / `cta --help` to list commands.

`COMMANDS` is the single source of truth for the command set; it is checked against
`reference/tool_registry.md` by `tests/test_registry_sync.py`, so the docs table, the CLI,
and the package can't silently drift.
"""
from __future__ import annotations

import importlib
import sys

# command name -> dotted module path (each module exposes `main()`).
COMMANDS: dict[str, str] = {
    "detect-env": "cta.detect_env",
    "inspect": "cta.inspect_adata",
    "prep-slim": "cta.prep_slim",
    "assemble-markers": "cta.assemble_markers",
    "compute-markers": "cta.compute_markers",
    "profile-markers": "cta.profile_markers",
    "score-signatures": "cta.score_signatures",
    "crosscheck-obs-pred": "cta.crosscheck_obs_pred",
    "reference-map": "cta.reference_map",
    "celltypist": "cta.run_celltypist",
    "fetch-census-reference": "cta.fetch_census_reference",
    "train-reference-model": "cta.train_reference_model",
    "map-scarches": "cta.map_query_scarches",
    "scimilarity": "cta.run_scimilarity",
    "subcluster": "cta.subcluster",
    "profile-clusters": "cta.profile_clusters",
    "cluster-confidence": "cta.cluster_confidence",
    "scaffold-cards": "cta.scaffold_cards",
    "build-report": "cta.build_report",
    "write-back": "cta.write_back",
}


def _usage() -> str:
    lines = [
        "usage: cta <command> [options]",
        "       cta <command> --help     # options for one command",
        "",
        "commands:",
        *(f"  {name}" for name in COMMANDS),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"cta: unknown command '{cmd}'\n\n{_usage()}", file=sys.stderr)
        raise SystemExit(2)
    module = importlib.import_module(COMMANDS[cmd])
    # Give the module its own argv so its argparse prog/flags work unchanged.
    sys.argv = [f"cta {cmd}", *rest]
    module.main()


if __name__ == "__main__":
    main()
