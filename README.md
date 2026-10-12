# cell-type-annotation

A Claude Code **skill** for evidence-driven cell-type annotation of pre-clustered
single-cell and spatial transcriptomics data. It is general and project-agnostic:
no organism, tissue, or panel is hard-coded.

The skill turns Claude into a careful, transparent annotator. It does **not**
black-box a classifier over your data — it gathers marker evidence, runs the
standard scVerse scoring/DE tools, reasons cluster-by-cluster down a label
hierarchy, exposes its assumptions for your sign-off, and lets adversarial critic
subagents try to refute every call before it is written down.

## What it does

- **Input:** an `AnnData` (`.h5ad`) with pre-existing Leiden/Louvain clusters and a
  user-provided embedding (used consistently for scoring, neighbors, sub-clustering,
  and label transfer).
- **Output:** a hierarchy of cell-type labels (normalized to Cell Ontology / CL IDs),
  per-cluster evidence cards, an assumptions ledger, and an HTML report per
  hierarchy level — all written as durable files in a per-run working directory so
  the run is resumable and auditable.
- **Scope:** annotation of existing clusters, including cluster *refinement*
  (sub-clustering / higher-resolution Leiden) when markers demand it.
- **Out of scope:** normalization, batch correction, doublet/ambient removal. The
  skill *detects and alerts* on these but does not silently fix them.

Works for both dissociated and spatial data, including limited targeted panels
(e.g. MERSCOPE/Xenium/CosMx), where it treats panel-absent genes as *unmeasured*,
never as negative evidence.

## Design principles

- **Compute-environment aware, scVerse-first.** Runs on CPU / Apple-Silicon by
  default — no GPU required. When a GPU or large memory is detected (and you confirm
  it), the skill can *additionally* train/refine a reference model and map the query
  onto it (scANVI/scArches), use a GPU kNN backend, pull larger reference atlases, and
  add a foundation-model cross-check. It never submits jobs or installs the GPU stack
  unasked, and a bigger model only ever earns one more opinion to weigh — never more
  authority.
- **Claude is the annotator.** Automated classifiers (CellTypist, SingleR,
  CellMapper, scANVI) produce *hypotheses*, not verdicts; disagreement is flagged,
  never silently overwritten.
- **Two hard gates:** (1) an assumptions ledger you sign off on before any labels
  are produced; (2) honest `Unknown` as a first-class label for irreconcilable
  evidence.
- **Artifact-centric.** Heavy compute runs as standalone `cta` CLI commands that
  write durable tables/plots; Claude reads the artifacts and reasons. This is what
  makes runs resumable and lets critic subagents (which can't see the chat or the
  in-memory `AnnData`) review only exported evidence.

## How it works — the three stages

The skill runs as a guided, three-stage workflow (the detailed algorithm lives in
`skills/cell-type-annotation/SKILL.md` and `reference/`):

1. **Scene-setting, environment & tooling gate.** First verify the required evidence
   tooling (the `knowledgebase` MCP; see *Requirements*), detect the compute environment
   (CPU / GPU / scheduler), and bootstrap an isolated `pixi` environment inside the skill
   (never touching your project environment). Then establish what can't be safely
   guessed — the `.h5ad` path, the **working-directory path** (always
   asked), the cluster key, the embedding to use everywhere, which layer is raw counts
   vs log-normalized, the assay/platform (and panel, if spatial), the biology, any
   **reference datasets** to map against, the **label write-back key**, and the
   label-vocabulary contract. The object is inspected to verify every key/layer exists,
   and a **normalization fingerprint** hard-stops the run if the declared log-normalized
   layer isn't actually size-normalized-then-logged.
2. **Plan.** Explore the object, gather prior knowledge for the tissue/context from the
   knowledge stack, draft the label hierarchy and per-level vocabulary, present for review.
3. **Execute (top-down, recursive).** For each level, and within each parent group:
   assemble candidate types + markers, panel-gate them, score (Wilcoxon DE + a
   `filter_rank_genes_groups` specificity pass; AUCell/pyUCell signatures with a
   z-scored + consolidated cross-check), cross-check
   against a tissue-matched classifier / a reference mapping (CellMapper) as a
   *hypothesis*, sign off the assumptions ledger, profile ambiguous clusters before any
   `Unknown`, write per-cluster evidence cards, then assign a label — or sub-cluster and
   recurse, or mark `Unknown`. Independent **critic subagents** try to refute each label
   from the exported raw tables; survivors go to an HTML report (with UMAP/dotplot plots)
   and the run is checkpointed before descending. After the last level, the labels are
   written back into the AnnData `obs` (gated on explicit confirmation).

## Installation

### For development (live-editing the skill)

Clone the repo, then load it from your local checkout so edits are picked up without
reinstalling.

**Terminal CLI** — point Claude Code at the checkout:

```bash
claude --plugin-dir /path/to/cell-type-annotation
```

**Desktop app** (no `--plugin-dir` flag) — symlink the skill into your personal
skills directory:

```bash
ln -s /path/to/cell-type-annotation/skills/cell-type-annotation \
      ~/.claude/skills/cell-type-annotation
```

Either way, edits to `SKILL.md`'s body and to the bundled `src/cta/` package and
`reference/` files are read on the next invocation (the `cta` CLI is installed
editable into the pixi env, so code edits need no reinstall). Changes to the SKILL.md **frontmatter**
(`name`/`description`) need a `/reload-plugins` or an app restart to re-register.

### As a plugin (lab / shared)

```bash
# one-time, per machine
claude plugin marketplace add quadbio/claude-plugins
claude plugin install cell-type-annotation@quadbio
```

## Requirements

- **[Claude Code](https://docs.claude.com/en/docs/claude-code/setup)** — the host the
  skill runs in.
- **[`knowledgebase` MCP server](https://github.com/biocontext-ai/knowledgebase-mcp)**
  (BioContextAI Knowledgebase) — **required**. Provides grounded marker / ontology /
  literature evidence (PanglaoDB, Cell Ontology, Human Protein Atlas, Europe PMC,
  STRING, Open Targets, …). The skill stops and asks you to configure it rather than
  silently degrading to web-only evidence. See
  [adding MCP servers to Claude Code](https://docs.claude.com/en/docs/claude-code/mcp).
- **[`zotero` MCP server](https://github.com/54yyyu/zotero-mcp)** — strongly preferred
  but optional. Surfaces your own curated literature; if absent, the skill notes it
  and falls back to the knowledgebase's Europe PMC tools.
- **[`pixi`](https://pixi.sh)** — the skill ships a `pixi.toml` and, on first run,
  bootstraps its own isolated Python 3.12 environment inside the skill directory
  (`.pixi/`); nothing is installed into your project environment. Tools run as
  `pixi run cta <command> …` (CPU / Apple-Silicon default). On a Linux + CUDA host,
  `pixi install -e gpu` then `pixi run -e gpu cta …` adds the optional GPU stack
  (scvi-tools, cuML, SCimilarity); on other platforms the `gpu` env is a no-op.

## Layout

```
skills/cell-type-annotation/
├── SKILL.md                       # orchestration: the 3-stage workflow
├── pixi.toml                      # environments (default = CPU, gpu = +CUDA) + deps (source of truth)
├── pyproject.toml                 # the `cta` package, installed editable into the pixi env
├── reference/                     # loaded as needed (progressive disclosure)
│   ├── methodology.md             # top-down recursive algorithm, gates, critics
│   ├── artifacts.md               # working-dir layout, run_state schema, reports
│   ├── compute_environments.md    # the 3 compute profiles, env detection, scheduler handoff
│   ├── tool_registry.md           # canonical step → cta command → GPU/profile map
│   ├── marker_tools.md            # decoupler, pyUCell, snapseed
│   ├── automated_tools.md         # CellTypist, CellMapper, scANVI/scArches, Census, what to skip
│   ├── knowledge_sources.md       # MCP coverage, CL normalization, atlases, loaders
│   └── differential_expression.md # markers (Job A) vs condition DE (Job B)
├── src/cta/                       # the `cta` CLI — heavy compute, run via `pixi run cta …`
│   │                              #   22 commands in 6 groups: data · markers · crosscheck ·
│   │                              #   cluster · report · reference (full table in SKILL.md)
│   ├── cli.py                     # Typer app wiring the command groups
│   ├── io.py                      # shared artifact/IO helpers + run-state checkpointing
│   └── …                          # one module per command (inspect_adata, compute_markers, …)
└── tests/                         # smoke test (cta end-to-end) + CLI↔registry sync guard
```

## Status

The annotation workflow and its tooling are complete and usable end to end. The skill
is under active development and we welcome issues and feedback. The released version is
tracked in `.claude-plugin/plugin.json`.

## License

MIT
