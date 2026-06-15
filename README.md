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

- **CPU / Apple-Silicon, scVerse-first.** No GPU, no foundation models, no
  multi-GB atlas downloads, no training large models.
- **Claude is the annotator.** Automated classifiers (CellTypist, SingleR,
  CellMapper) produce *hypotheses*, not verdicts; disagreement is flagged, never
  silently overwritten.
- **Two hard gates:** (1) an assumptions ledger you sign off on before any labels
  are produced; (2) honest `Unknown` as a first-class label for irreconcilable
  evidence.
- **Artifact-centric.** Heavy compute runs as standalone Python scripts that write
  durable tables/plots; Claude reads the artifacts and reasons. This is what makes
  runs resumable and lets critic subagents (which can't see the chat or the
  in-memory `AnnData`) review only exported evidence.

## How it works — the three stages

The skill runs as a guided, three-stage workflow (the detailed algorithm lives in
`skills/cell-type-annotation/SKILL.md` and `reference/`):

1. **Scene-setting, environment & tooling gate.** First verify the required evidence
   tooling (the `knowledgebase` MCP; see *Requirements*) and bootstrap a dedicated,
   reusable Python venv (never touching your project environment). Then establish what
   can't be safely guessed — the `.h5ad` path, the **working-directory path** (always
   asked), the cluster key, the embedding to use everywhere, which layer is raw counts
   vs log-normalized, the assay/platform (and panel, if spatial), the biology, any
   **reference datasets** to map against, the **label write-back key**, and the
   label-vocabulary contract. The object is inspected to verify every key/layer exists,
   and a **normalization fingerprint** hard-stops the run if the declared log-normalized
   layer isn't actually size-normalized-then-logged.
2. **Plan.** Explore the object, gather prior knowledge for the tissue/context from the
   knowledge stack, draft the label hierarchy and per-level vocabulary, present for review.
3. **Execute (top-down, recursive).** For each level, and within each parent group:
   assemble candidate types + markers, panel-gate them, score (Wilcoxon + COSG markers;
   AUCell/pyUCell signatures with a z-scored + consolidated cross-check), cross-check
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

Either way, edits to `SKILL.md`'s body and to the bundled `scripts/` and `reference/`
files are read on the next invocation. Changes to the SKILL.md **frontmatter**
(`name`/`description`) need a `/reload-plugins` or an app restart to re-register.

### As a plugin (lab / shared)

```bash
# one-time, per machine
claude plugin marketplace add quadbio/cell-type-annotation
claude plugin install cell-type-annotation@quadbio-cell-type-annotation
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
- **[`uv`](https://docs.astral.sh/uv/getting-started/installation/)** — the skill
  bootstraps its own dedicated Python 3.12 venv under
  `~/.cache/cell-type-annotation/venv` on first run; nothing is installed into your
  project environment.

## Layout

```
skills/cell-type-annotation/
├── SKILL.md                       # orchestration: the 3-stage workflow
├── reference/                     # loaded as needed (progressive disclosure)
│   ├── methodology.md             # top-down recursive algorithm, gates, critics
│   ├── artifacts.md               # working-dir layout, run_state schema, reports
│   ├── marker_tools.md            # decoupler, pyUCell, COSG, snapseed
│   ├── automated_tools.md         # CellTypist, CellMapper, SingleR, what to skip
│   ├── knowledge_sources.md       # MCP coverage, CL normalization, loaders
│   └── differential_expression.md # markers (Job A) vs condition DE (Job B)
└── scripts/                       # heavy compute, run via Bash
    ├── setup_env.sh               # bootstrap the dedicated uv venv
    ├── requirements.txt           # core scVerse stack (source of truth)
    ├── requirements-extra.txt     # best-effort extras (pyUCell, singler, ...)
    ├── cta_io.py                  # shared artifact/IO helpers
    ├── inspect_adata.py           # Stage 1: verify keys/layers + normalization fingerprint
    ├── prep_slim.py               # optional: cache a slim AnnData for big objects
    ├── assemble_markers.py        # optional: seed marker JSON from a model / reference
    ├── compute_markers.py         # rank_genes_groups (wilcoxon) + COSG
    ├── score_signatures.py        # AUCell + pyUCell + DE-overlap; z-score + crosscheck
    ├── run_celltypist.py          # tissue-gated CellTypist (list / predict / markers)
    ├── reference_map.py           # CellMapper reference label transfer (joint CCA/PCA)
    ├── fetch_census_reference.py  # optional: on-demand CELLxGENE Census reference subset
    ├── profile_clusters.py        # QC + composition + spatial (investigate before Unknown)
    ├── scaffold_cards.py          # generate evidence-card stubs from tables
    ├── build_report.py            # per-level HTML report (template/assembly)
    ├── report_plots.py            # plot generation for build_report (UMAP/dotplot/composition)
    └── write_back.py              # write labels into obs (final, gated)
```

## Status

The annotation workflow and its tooling are complete and usable end to end. The skill
is under active development and we welcome issues and feedback. The released version is
tracked in `.claude-plugin/plugin.json`.

## License

MIT
