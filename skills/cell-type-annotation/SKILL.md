---
name: cell-type-annotation
description: >-
  Evidence-driven cell-type annotation of pre-clustered single-cell or spatial transcriptomics data (AnnData). Use
  whenever the user wants to annotate, label or identify the cell types or states of Leiden/Louvain clusters, build or
  refine a cell-type hierarchy, or interpret scRNA-seq / snRNA-seq / MERSCOPE / Xenium / Visium / CosMx clusters,
  even if they just say "what are these clusters?" or "label these cells". Claude is the annotator: it gathers marker
  evidence, runs one routed automated cross-check, reasons level by level down a hierarchy, has the user sign off its
  assumptions and critic subagents try to refute every call. Flags low-quality, cycling and doublet clusters on a
  separate state axis; does not normalize, batch-correct or remove doublets.
---

# Cell-type annotation

Turn existing clusters into a legible, contestable label hierarchy. **You are the annotator**: tools produce
evidence, you weigh it, and every label rests on numbers a reader can check. Task scripts call scanpy, decoupler and
the cross-check tools directly; the `cell_type_annotation` package holds only what they lack: routing, checks, the
label hierarchy, the report and the hierarchy figure.

Paths starting with `references/` are in this skill's folder; all others are relative to the task directory.

**Elsewhere.**
- **analysis-workflow** (load it first): an annotation run is one of its tasks. It owns:
  - the task directory and its scripts;
  - reading the working object;
  - write-back with `commit_adata`;
  - environments.
- **sci-figures:** only for the run's one paper figure, the annotation hierarchy. Intermediate plots are plain scanpy.
- **Compute and job submission:** the environment's own skill.

## Principles

- **Two gates** (`references/methodology.md` §2):
  - **A.** The user signs off the assumptions ledger before the first label exists, and every materially new
    assumption after.
  - **B.** A justified `Unknown` is a result, not a failure.
- **Every automated call is a hypothesis.** Run exactly one routed cross-check per level, and a second only to break
  a tie. Flag disagreement; never overwrite with it. A bigger model earns one more opinion, not more authority.
- **Two axes.** A cluster has a type and a state (`normal`, `low-quality`, `cycling`, `doublet`, ...). A cycling
  cluster keeps its type.
- **Panel-aware.** An off-panel gene is unmeasured, never negative evidence.
- **Alert, don't fix.**
  - In scope: computing QC metrics and a cell-cycle score, and sub-clustering.
  - Out of scope: normalization, batch correction, doublet detection.
- **Interactive by default.** A non-interactive run is an explicit opt-in with the gates' answers supplied up front;
  gaps still stop it (`references/methodology.md` §2).
- **Keep the user oriented.** At each stage and level, say where you are, what comes next, and what will be slow.

| File | Covers |
| --- | --- |
| `references/methodology.md` | gates, termination, cards, sub-clustering, state axis, critics, panel rules |
| `references/method_selection.md` | the routing table (generated from code), why, and a recipe per cross-check |
| `references/marker_tools.md` | between-cluster DE, sibling margins, the curated panel, scores, QC, figures |
| `references/knowledge_sources.md` | the MCPs, Cell Ontology, atlases, the order to consult |

## What the task directory holds

| file | written in | holds |
| --- | --- | --- |
| `reports/plan.md` | stages 1-2 | the data, keys, assay and biology as confirmed; covariates; panel capability; hierarchy and vocabulary; routing result |
| `results/markers.csv` | 2, 3 | signed, sourced curated markers (`read_markers`) |
| `reports/assumptions.md` | 3 | the Gate A ledger, one block per level, with its sign-off |
| `results/<table>_<level>.csv` | 3 | evidence: markers, sibling margins, panel detection, negative flags, panel gate, cross-check, QC deviation, confidence |
| `results/labels_<level>.csv` | 3 | the call and its reasoning (`LABEL_COLUMNS`): the single source of truth |
| `reports/<level>.md` | 3 | `render_level`: overview, open problems, one card per cluster |
| `figures/` | 3-4 | quick scanpy plots; the final hierarchy PDF |

Resuming: read the task README, `plan.md`, `assumptions.md`, the latest labels table and report; continue from there.

## Workflow

At the start, give the user the map: scene-setting, plan, then level by level with a sign-off and a report each,
and nothing written to their object until the end.

### 1. Scene-setting
1. **Tooling gate:** the `knowledgebase` MCP is required: stop and ask for it if missing. `zotero` is preferred.
   Check both respond (`references/knowledge_sources.md`).
2. **Inspect** the working object without loading it whole (`obs`, `var`, `obsm` keys, layer stats).
3. **Confirm with the user**, pre-filled from the inspection, in two or three batched AskUserQuestion rounds:
   - the cluster key, the embedding to use everywhere, the plotting embedding, and the spatial coordinates;
   - which layer is counts and which is log-normalized (ask, never infer);
   - assay and platform, and for a targeted panel its size;
   - organism, tissue, stage (adult / developmental / organoid) and state (normal / disease / tumour). Stage and
     state cannot be probed from the data;
   - the covariates to show composition over;
   - the label vocabulary (a paper's labels, an ontology, or a free hierarchy with per-level meaning), with CL ids;
   - references: path, label columns, counts or not, gene ids;
   - the write-back key stem.
4. **Hard stop on normalization:** `normalization_fingerprint(X_lognorm)`. If it is not `ok`, stop: markers on
   mis-normalized data track depth, not identity. Show the verdict; the user fixes the layer or confirms a
   non-standard normalization.
5. **Gene ids:** markers are symbols. For Ensembl `var_names`, use a symbol column (`feature_name`, ...). Without
   one, map the marker genes via the knowledgebase MCP (`bc_get_ensembl_id_from_gene_symbol`).
6. **QC and cell cycle**, if absent, computed in memory (`references/marker_tools.md`).
7. Write `reports/plan.md`. For a targeted panel, add which expected identities it cannot resolve, because their
   canonical markers are off-panel.

### 2. Plan
1. Gather priors in the order of `references/knowledge_sources.md`, and write them to `results/markers.csv` with
   sources.
2. Route the cross-check: `routing.evaluate(...)` (`references/method_selection.md`). Record the ranked result.
3. Draft the hierarchy and each level's meaning. Present the plan.

### 3. Each level, top-down, within each parent group
Orient the user each time: "L2 within Neuron: 9 clusters, ~120k cells; scores, one cross-check, the ledger, then a
report."
1. **Markers:** curate this level's candidates in `results/markers.csv`, including negatives. Run `panel_gate`;
   an uncallable candidate contributes to `Unknown`.
2. **Evidence**, into `results/` (`references/marker_tools.md`):
   - within-parent `rank_genes_groups` and `filter_rank_genes_groups`;
   - `hierarchy_markers` sibling margins;
   - panel detection and `negative_marker_flags`;
   - scores;
   - `qc_deviation`.
3. **Cross-check:** the routed method's recipe, summarized with `prediction_summary`. Its caveat goes into the cards.
4. **Gate A:** append the level to `reports/assumptions.md` and get the sign-off (`references/methodology.md` §2).
5. **Assign** by filling `results/labels_<level>.csv`:
   - normalize the label to a CL id and set the state;
   - for mutually exclusive co-expression, sub-cluster once and recurse;
   - investigate before `Unknown`;
   - calibrate the confidence against `confidence_proxy`.
6. **Critics:** independent subagents read the raw tables, re-derive the markers, fetch an external competing
   hypothesis, and check the label against its parent path. Reconcile; take conflicts you cannot resolve to the user
   (`references/methodology.md` §7).
7. **Report:** `render_level(...)` to `reports/<level>.md`; `check_level` must return nothing. Make quick scanpy
   plots of the level. Present the flagged clusters and pause: the user approves descending, and into which groups.
   If more than half the clusters are `Unknown`, consult the user first.

### 4. Finish
1. **CL ids:** run `bc_get_term_details` on every `cl_id`: it exists, is current, and fits the label. Reconcile with
   the user.
2. **Write-back**, following analysis-workflow:
   - one categorical `obs` column per level, keyed `<stem>_L<n>_<version>`, plus `_state` and `_cl_id` companions,
     and any sub-cluster key;
   - colours in `uns["<key>_colors"]`;
   - `build_hierarchy` over the level keys is the nesting check;
   - `commit_adata` dry run, the user's sign-off on that diff, then `yes=True`.
3. **Figure:** the annotation hierarchy, made with sci-figures: `display_markers` picks the canonical markers per
   node, and `pl.annotation_hierarchy` draws them (`references/marker_tools.md`).
4. **README:**
   - inputs and the routed method;
   - the write-back keys;
   - which critic objections were declined, and why;
   - anything still provisional.
