# cell-type-annotation

A Claude Code plugin for evidence-driven cell-type annotation of pre-clustered single-cell and spatial
transcriptomics data. Claude is the annotator:
- it gathers marker evidence and runs one routed automated cross-check;
- it reasons level by level down a label hierarchy;
- the user signs off its assumptions before any label exists;
- critic subagents try to refute every call.

Labels are normalized to Cell Ontology ids. Low-quality, cycling and doublet clusters are flagged on a separate
state axis. Off-panel genes on targeted spatial panels are treated as unmeasured, never as absent.

| Part | What it does |
| --- | --- |
| `skills/cell-type-annotation/` | The method: principles, the per-level workflow, and references on methodology, method selection, marker tools and knowledge sources. |
| `src/cell_type_annotation/` | What task scripts can't get from scanpy or decoupler: cross-check routing, normalization and panel checks, the nested label hierarchy with sibling-restricted marker scoring, the level report, and the annotation-hierarchy figure. |

It runs as a task of [analysis-workflow](https://github.com/quadbio/analysis-workflow), which owns the task
directory, the working object and write-back. The final hierarchy figure follows the `sci-figures` skill.

## Use

Inside an analysis-workflow task, ask Claude to annotate a working object's clusters: "what are these clusters?",
"annotate the leiden clusters".
- **Inputs:** the working object, with a cluster key, an embedding and a log-normalized layer.
- **Outputs:** in the task directory:
  - the plan;
  - the signed-off assumptions ledger;
  - the curated markers;
  - evidence tables, and a labels table and Markdown report per level;
  - the hierarchy figure.
- **Write-back:** after sign-off, one versioned `obs` column per level.

It requires the [knowledgebase MCP](https://github.com/biocontext-ai/knowledgebase-mcp). The
[Zotero MCP](https://github.com/54yyyu/zotero-mcp) is strongly preferred.

## Install

The plugin, once per machine, from the [quadbio marketplace](https://github.com/quadbio/claude-plugins).

The package, in the analysis repo's `pixi.toml`, next to the scverse stack the task scripts use:

```toml
[pypi-dependencies]
cell-type-annotation = { git = "https://github.com/quadbio/cell-type-annotation", tag = "v0.4.1" }
```

Cross-check tools (celltypist, cellmapper, panhumanpy, scvi-tools, scimilarity) are added as a run needs them. Their
install notes are in `references/method_selection.md`.

The last release with the `cta` command line is [v0.3.0](https://github.com/quadbio/cell-type-annotation/tree/v0.3.0).

## Develop

```bash
uv sync --group dev
uv run pytest
uv run pre-commit run --all-files
claude plugin validate .
claude --plugin-dir .        # try the skill from this checkout
```

The routing tables in `references/method_selection.md` are generated; after changing `routing.py`, regenerate them
(a test fails until you do):

```bash
uv run python -c "from cell_type_annotation.routing import update_page; \
update_page('skills/cell-type-annotation/references/method_selection.md')"
```

For live package edits in an analysis repo, depend on the checkout:
`cell-type-annotation = { path = "../cell-type-annotation", editable = true }`.

## Release

Bump `version` in both `pyproject.toml` and `.claude-plugin/plugin.json` (a test checks they agree).
That is the release: the marketplace follows `main`, users update when the version changes, and the merge tags
`vX.Y.Z`, which analysis repos pin.

## License

MIT
