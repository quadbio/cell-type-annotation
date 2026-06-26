# Reference: Working directory, run state & reports

The skill is **artifact-centric**: heavy compute writes durable files, and Claude
reasons over those files. This is what makes a run (a) resumable after an
interruption and (b) reviewable by critic subagents that cannot see the chat or the
in-memory `AnnData`. Keep everything in the working directory current.

## Where the working directory lives

**Always ask the user for the working-directory path at Stage 1 — do not pick a default.**
(Next-to-the-data is often a poor location.) It must live **inside the user's project**,
never inside this plugin repo. Record the chosen path in `run_state.json`.

## Layout

```
<dataset>-annotation-run/
├── inspect_report.json   # `cta data inspect --workdir` dump (the facts context.md is written from)
├── context.md            # the data: paths, keys, layers, assay, panel, tissue (you author it)
├── assumptions.md        # Gate A ledger — signed off before any labels
├── plan.md               # the hierarchy, vocabulary contract, planned workflow
├── run_state.json        # machine-readable checkpoint (resume from here)
├── tables/               # exported by scripts: DE, specificity, scores, per-cluster stats
│   ├── markers_<level>.csv          # rank_genes_groups (wilcoxon) + pts
│   ├── markers_filtered_<level>.csv # filter_rank_genes_groups survivors (cluster-specific markers, ranked by log2FC)
│   ├── panel_gating_<level>.csv     # on-panel vs dropped (off-panel) markers
│   ├── scores_overlap_<level>.csv   # per-cluster top-DE ∩ marker-set overlap
│   ├── scores_aucell_<level>.csv    # decoupler aucell, per-cell → cluster mean
│   ├── scores_aucell_z_<level>.csv  # aucell z-scored across clusters (relative enrichment)
│   ├── scores_ucell_<level>.csv     # pyUCell (pos+neg), per-cell → cluster mean
│   ├── crosscheck_<level>.csv       # consolidated: each method's top signature + agreement
│   ├── refmap_<level>_<col>.csv     # CellMapper reference label transfer (per cluster)
│   ├── qc_profile_<level>.csv       # profile_clusters: per-cluster QC medians vs global
│   ├── qc_deviation_<level>.csv     # profile_clusters: cluster medians as robust-z vs across-cluster norm
│   ├── crosstab_<level>_<col>.csv   # profile_clusters: composition over a user-named obs
│   └── spatial_profile_<level>.csv  # profile_clusters: per-cluster spatial centroid/spread
├── evidence/             # one card per cluster per level (markdown; cluster id sanitized)
│   └── L<level>_cluster<id>.md
├── plots/                # umap-by-label, dotplot, optional composition (png), embedded in report
├── reports/              # human-facing HTML, one per hierarchy level
│   └── report_L<level>.html
└── labels/               # the deliverable: cluster→label maps (+ written to obs via cta report write-back)
    └── annotation_L<level>.csv
```

## `labels/annotation_<level>.csv` schema (Claude authors this by hand)

No script writes this file — **you** write it, so the columns must be exact or the
downstream scripts break. Required: **`cluster`**, **`label`**. Recommended:
`state`, `cl_id`, `confidence` (0–1 float), `status`, `note`.

- `state` is the orthogonal **cell-state** axis (`normal` / `low-quality` / `cycling` /
  `doublet` / …, `methodology.md` §6) — written *next to* the type label, exempt from CL-ID
  normalization and the hierarchy-nesting check. Leave blank for ordinary clusters.
- `cta report build` reads `cluster,label,state,cl_id,confidence,status,note`; `cta report write-back`
  hard-requires `cluster`+`label` and also writes `state`/`cl_id`/`confidence` as companion
  obs columns (`<prefix>_<level>_state`, …) if present.
- **`status` MUST be lowercase**, one of `{assigned, unknown, needs_subcluster, refuted}`
  — the report keys flagging off these exact lowercase strings. Write the *label* as
  `Unknown` (display) but the *status* column as `unknown`. A capitalized `status`
  silently fails to flag.
- `cluster` values are matched as strings against the cluster-key categories — keep ids
  verbatim (e.g. `24/25_0`).

## `assumptions.md` (the sign-off ledger)

The Gate A artifact a non-developer biologist actually reads and approves (rules:
`methodology.md` §2). It is a **living document** — one block per hierarchy level,
appended as you descend. Write it in plain language: the reader is approving *what the
labels will mean*, not a config. Each level block has four parts: a marker→type table,
the priors you are importing, a panel-caveats table (targeted panels only), and the
label-vocabulary contract. **The on-disk ledger must mirror the table you render in
chat for sign-off** — same rows, same wording — so the audit trail stays legible and a
reviewer can match what was approved to what was recorded.

A filled-in example for a generic PBMC run (L1 compartment), Xenium targeted panel:

```markdown
# Assumptions ledger — pbmc-demo

## L1 — compartment  (status: SIGNED OFF 2026-06-15)

### Markers I'll rely on
| Cell type   | Markers I'll rely on | Why / source                  | What this means for your results                          |
| ----------- | -------------------- | ----------------------------- | --------------------------------------------------------- |
| T cell      | CD3D, CD3E, TRAC     | canonical pan-T (PanglaoDB)   | Anything CD3+ is grouped as T; NK split happens at L2.     |
| B cell      | CD79A, MS4A1, CD19   | canonical pan-B               | Plasma cells fold in here at L1, separated at L2.          |
| Myeloid     | LYZ, CD14, FCGR3A    | monocyte/macrophage consensus | Mono, DC and macrophage are one L1 box, resolved at L2.    |
| Erythroid   | HBB, HBA1            | hemoglobin genes              | If present, flagged; often ambient contamination, not cells. |

### Priors I'm importing
- 10x dissociated PBMC, healthy adult donor → expect the usual lymphoid/myeloid split,
  no tissue-resident or stromal compartments.
- No perturbation; composition is treated as a readout, not assumed.

### Panel caveats (targeted panel — absence ≠ negative evidence)
| Identity to call | Canonical markers | On panel? | Consequence                                              |
| ---------------- | ----------------- | --------- | -------------------------------------------------------- |
| NK cell          | NCAM1, KLRD1      | NCAM1 off | Can flag NK at L1, but cannot cleanly separate NK from cytotoxic T at L2 on this panel. |
| Plasmacytoid DC  | CLEC4C, IL3RA     | both off  | pDC is uncallable on this panel → contributes to Unknown, never a confident call. |

### Label-vocabulary contract
Free 4-level hierarchy: L1=compartment, L2=lineage, L3=type, L4=state. Final labels
normalized to Cell Ontology (CL) IDs. Reusing no specific paper's labels.
```

Append an analogous block per level (L2, L3, …) as you descend; re-prompt for sign-off
only on a materially new assumption (`methodology.md` §2). Omit the panel-caveats table
on non-targeted (dissociated whole-transcriptome) data.

## `context.md` (Stage 1 output — Claude authors it)

Plain-language record of what the data *is*, captured from the scene-setting
questions and verified against the object by `cta data inspect`. **You write this file**
(and `run_state.json`) from the inspect report — `cta data inspect` only verifies the
object and prints/`--workdir`-dumps the structured facts (`inspect_report.json`); it does
not author the narrative files. Contents:

- file path(s); `n_obs`, `n_vars`;
- the **cluster key** (`obs` column with the pre-existing Leiden/Louvain);
- the **embedding key** (`obsm`) used everywhere, and the **viz embedding**
  (UMAP/tSNE) used for plots;
- which **layer** is raw counts vs log-normalized (explicitly — never guess);
- assay / chemistry; for spatial: platform + panel (and panel size);
- tissue / organism / developmental context;
- the **label-vocabulary contract** (see plan.md).

Skeleton (keep generic; fill from `cta data inspect`):

```markdown
# Context — pbmc-demo

## Data
- file: /abs/path/to/pbmc.h5ad  (n_obs=12,431, n_vars=480)
- canonical working file: /abs/path/to/pbmc_cta.h5ad  (`cta data canonicalize`)
- cluster key (obs): leiden  (14 clusters)
- embedding key (obsm): X_scvi   |   viz embedding: X_umap
- gene IDs: symbols (canonicalized from Ensembl via `var['feature_name']`; Ensembl kept in `var['ensembl_id']`)
- layers: counts = `counts` (raw; pulled from `.raw`) ; log-normalized = `X` (verified by fingerprint) ; `lognorm_cp10k` = log1p(CP10k) precomputed for scale-sensitive classifiers (CellTypist)
- QC signal: depth/mito + cell-cycle in obs (`cta data qc`; see `qc.json`) — used to set the `state` axis

## Assay
- platform: Xenium (targeted, 480-gene panel)
- organism / tissue: human / peripheral blood
- developmental / experimental context: healthy adult, no perturbation

## Composition covariates (for report --split-by)
- condition, donor   (donor has uns colours; condition does not)

## Panel capability  (targeted panels only)
Expected identities this panel CANNOT resolve because their canonical markers are
off-panel — distinct from biology simply not found in the data:
- pDC: CLEC4C / IL3RA off-panel → uncallable.
- NK vs cytotoxic T: NCAM1 off-panel → may be merged.
Revisit per level against `panel_gating_<level>.csv` (dropped / uncallable sets).

## Label-vocabulary contract
Free 4-level hierarchy (L1 compartment → L4 state); final labels normalized to CL IDs.
```

The **Panel capability** section is required when `targeted_panel` is true; omit it
otherwise. It states *what the panel can't ask*, which is different from a candidate
that was tested and simply not found.

## `run_state.json` schema

Minimal, machine-readable, the source of truth for resuming:

```json
{
  "schema_version": 1,
  "dataset": "my_dataset",
  "workdir": "/abs/path/to/my_dataset-annotation-run",
  "adata_path": "/abs/path/to/data.h5ad",
  "keys": {
    "cluster_key": "leiden",
    "embedding_key": "X_scpoli",
    "viz_embedding_key": "X_umap",
    "counts_layer": "counts",
    "lognorm_layer": "lognorm",
    "var_convention": "symbol"
  },
  "canonical_adata": "/abs/path/to/data_cta.h5ad",
  "panel": {"is_spatial": true, "platform": "MERSCOPE", "n_panel_genes": 500},
  "hierarchy": ["L1_compartment", "L2_lineage", "L3_type", "L4_state"],
  "vocabulary_contract": "free hierarchy; per-level semantics defined in plan.md; normalize to CL",
  "stage_reached": "execute",
  "gates": {"signed_off_levels": ["L1_compartment"]},
  "progress": {
    "current_level": "L1_compartment",
    "current_step": "critic",
    "completed_levels": [],
    "assigned": {}
  },
  "compute": {
    "profile": "local_cpu",
    "accelerator": "mps",
    "n_gpus": 0,
    "vram_gb": null,
    "ram_gb": 64.0,
    "scheduler": null,
    "execution_mode": "inline",
    "pixi_env": "default",
    "python": "3.12"
  }
}
```

- `canonical_adata` is the `<stem>_cta.h5ad` written by `cta data canonicalize`; every
  downstream `load_adata` picks it up transparently, so it is the object the run actually
  operates on. `keys.var_convention ∈ {symbol, ensembl}` records the canonical `var_names`
  convention (normally `symbol`, so curated/CellTypist/literature markers panel-gate
  directly). The command also writes a `canonicalization.json` beside the sidecar recording
  what it did (var-name source, counts source, genes renamed/unmapped/deduped, and the
  upfront gene filter — `gene_filter.min_cells` / `gene_filter.dropped`) — fold its
  essentials into `context.md`. The gene filter is a small **absolute** floor
  (`min_cells=5`) that drops genes in ≤4 cells, so every downstream step shares one clean
  gene set; after it, a curated marker that's missing is "not measured" rather than "not
  expressed" — but only for genes in <0.1% of cells, so panel-gating is unaffected in
  practice. `cta data qc` then fills standard QC + cell-cycle into the
  same sidecar's obs (when absent) and writes a `qc.json` beside it (columns written,
  mito/ribo gene counts, cell-cycle coverage) — also fold its essentials into `context.md`.
- `stage_reached ∈ {scene_setting, plan, execute}` tells a resuming session which
  artifacts should already exist (e.g. `plan.md` only after `plan`). (The env + MCP gate
  is part of Stage 1 scene-setting now.)
- `gates.signed_off_levels` is the list of hierarchy levels whose assumptions the user
  has explicitly signed off (Gate A is a *living* ledger — see `methodology.md` §2).
  A level not in this list must not have labels assigned yet. In a **non-interactive**
  (opt-in benchmark) run the gate is still recorded here but tagged as pre-authorized —
  e.g. `"gates": {"signed_off_levels": ["L1_compartment"], "auto_approved": "non_interactive"}`
  — so the audit trail shows it was approved against user-supplied context, not prompted
  (`methodology.md` §2).
- `progress.assigned` maps cluster id → `{label, state, cl_id, confidence, status}` where
  `status ∈ {assigned, unknown, needs_subcluster, refuted}` and `state` is the optional
  cell-state axis (`normal` / `low-quality` / `cycling` / `doublet` / …; `methodology.md` §6).
- `progress.current_step ∈ {score, cross_check, gate_a, evidence, assign, critic,
  report}` is an intra-level cursor (see the resumability contract below).
- `compute` records the detected, user-confirmed environment (Stage 1;
  `reference/compute_environments.md`). `profile ∈ {local_cpu, local_gpu, scheduler}`;
  `accelerator ∈ {cpu, cuda, mps}`; `execution_mode ∈ {inline, submit}` — `submit` means
  GPU-heavy steps are written into `plan.md` to run via the user's own job-submission
  mechanism, never by this skill. Seeded from `cta detect-env`. (Replaces the old
  thin `env` block; `pixi_env` ∈ {default, gpu} records which pixi env was built.)

To resume, read this file and continue from `progress.current_level` /
`progress.current_step`.

## `plan.md` (Stage 2 output)

Presented to the user at Stage 2: the hierarchy they're agreeing to annotate against,
the vocabulary contract, and the workflow outline. Keep it short and readable — it
sets expectations before any compute.

Skeleton (generic example):

```markdown
# Plan — pbmc-demo

## Hierarchy
| Level | Semantics   | Example labels                         |
| ----- | ----------- | -------------------------------------- |
| L1    | compartment | Lymphoid, Myeloid, Erythroid           |
| L2    | lineage     | T cell, B cell, NK, Monocyte, DC       |
| L3    | type        | CD4 T, CD8 T, Naive B, Memory B, cDC   |
| L4    | state       | naive / memory / activated / exhausted |

Each level **strictly refines** the one above (every finer label has exactly one
parent). Depth is annotated only as far as the data + panel support.

## Vocabulary contract
Free hierarchy with the per-level semantics above; final labels normalized to Cell
Ontology (CL) IDs. Not reusing a specific paper's labels.

## Workflow (per level, top-down + recursive)
1. assemble priors → candidate types + markers
2. panel-gate markers; score (DE/specificity + AUCell/UCell); one automated cross-check
3. Gate A: sign off the assumptions ledger (blocking first time)
4. evidence card per cluster → assign / sub-cluster / Unknown
5. critic pass (refute), reconcile, HTML report, checkpoint
6. descend into each assigned parent group
```

## HTML reports (one per level)

A self-contained HTML file per hierarchy level for user review between checkpoints.
Generated by `cta report build` from the tables + evidence cards. Each report
shows, per cluster: the assigned label + CL ID + confidence, the top markers and
their dotplot, the candidate scores, the off-panel/dropped genes, the automated-method
agreement, the critic objections and their resolution, and an explicit
"rejected because…" note. Flagged clusters (`Unknown`, low confidence, **thin
top1–top2 margin despite high confidence**, an **uncallable** winning candidate, or a
label that survived a critic challenge only weakly) are highlighted at the top.

## Resumability contract

Every script writes to `tables/` / `plots/` deterministically and never holds state
in memory across invocations. Claude updates `run_state.json` at every checkpoint —
after each level, **and on each intra-level step transition** by setting
`progress.current_step`. A new session can reconstruct the full picture from the
working directory alone — re-read `context.md`, `assumptions.md`, `plan.md`,
`run_state.json`, then the evidence cards.

**Mid-level resume.** Intra-level steps are **idempotent**: every script overwrites its
`tables/` outputs deterministically, so on resume you may simply re-run the current
level from the start of `progress.current_step`. The two non-idempotent actions need
care: do not re-spawn critic subagents for clusters already reconciled (the evidence
card records the resolution), and do not double-write `labels/` (overwrite, don't
append). If `current_step` is unset, restart the level from `score` — re-running the
deterministic scoring scripts is cheap and safe.
