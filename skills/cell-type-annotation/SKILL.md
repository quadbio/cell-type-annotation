---
name: cell-type-annotation
description: >-
  Evidence-driven cell-type annotation of pre-clustered single-cell or spatial
  transcriptomics data (AnnData/scanpy). Use this whenever the user wants to
  annotate, label, or identify cell types / cell states for Leiden or Louvain
  clusters, assign cell identities from marker genes, build or refine a cell-type
  hierarchy, or interpret scRNA-seq / snRNA-seq / MERSCOPE / Xenium / Visium /
  CosMx clusters — even if they just say "what are these clusters?", "annotate my
  h5ad", "label these cells", or "figure out the cell types". Claude itself is the
  annotator: it gathers marker evidence, runs scVerse scoring/DE tools, reasons
  cluster-by-cluster down a label hierarchy, exposes its assumptions for sign-off,
  and has critic subagents try to refute every call. Compute-environment aware:
  scVerse-first on CPU/Apple-Silicon by default, and when a GPU or large memory is
  detected it can also download reference atlases, train/refine a reference model and
  map the query onto it (scANVI/scArches) — every automated call still a hypothesis,
  never a verdict. Does NOT do normalization, batch correction, or doublet removal
  (it detects and alerts only).
license: MIT
---

# Cell-type annotation

Turn pre-existing clusters into a legible, contestable cell-type hierarchy: every label
is backed by an auditable evidence card, and nothing is written until it survives an
adversarial challenge. **You (Claude) are the annotator** — the tools produce evidence;
you weigh it. This file is the orchestration; detail lives in `reference/` (read on
demand). Rationale/algorithm: `reference/methodology.md` §1.

## Operating principles (read once, hold throughout)

- **Scope.** Annotate pre-existing Leiden/Louvain clusters on a **user-provided
  embedding**, used consistently for scoring, neighbors, sub-clustering, and label
  transfer. Cluster *refinement* (sub-clustering / higher-res Leiden) is in scope.
  Normalization, batch correction, and doublet/ambient removal are **out of scope** —
  detect and alert, never silently fix.
- **Two hard gates** (details in `reference/methodology.md`):
  - **Gate A:** keep a *living* assumptions ledger. Get a blocking sign-off (via
    AskUserQuestion) *before* the first labels; at deeper levels, append new
    assumptions and re-prompt only when you introduce a materially new one.
  - **Gate B:** `Unknown` is a first-class label for irreconcilable evidence — a
    justified `Unknown` is a success, not a failure.
- **Every automated call is a hypothesis, not a verdict.** Flag disagreement between
  marker reasoning and CellTypist/SingleR/CellMapper/scANVI; never silently overwrite.
  A bigger model (a GPU reference model, a foundation model) earns one more independent
  opinion to weigh, **not** more authority.
- **Compute-environment aware.** Detect the environment early and plan accordingly. On
  CPU/Apple-Silicon (`local_cpu`) the skill runs exactly as it always has; a GPU/large-memory
  host (`local_gpu`) or login node (`scheduler`) additionally unlocks reference-model
  training + scArches mapping, GPU kNN, and larger atlases. **Detection only — never submits
  a job or installs the GPU stack unasked.** Full matrix and the scheduler handoff:
  `reference/compute_environments.md`.
- **Artifact-centric.** Heavy compute runs as standalone `cta` commands (via Bash) that
  write durable files; you reason over the files. This is what makes runs resumable
  and lets critic subagents review exported evidence. Keep the working directory
  current. See `reference/artifacts.md`.
- **Panel-aware.** On targeted panels (MERSCOPE/Xenium/CosMx), a gene's absence from
  the panel is **not** evidence of non-expression (panel-gating is enforced per step).
- **Stay in chat.** No Jupyter loop. Present intermediate results as
  tables/plots/HTML/markdown for the user to react to.
- **Keep the user oriented.** This is a long, multi-stage, partly-autonomous run — the
  user should never wonder where they are. At each stage and each hierarchy level,
  *say* what you're about to do, why, and what they'll get; before a potentially slow
  script (scoring a large object, `cta crosscheck reference-map`, the critic fan-out) give a quick
  heads-up so a quiet terminal reads as "working," not "hung."

## The workflow: three stages

At the very start, give the user the map: **(1) Scene-setting** — inspect the object and
confirm data/keys/biology with you, check tooling; **(2) Plan** — gather prior knowledge
and draft the label hierarchy for your review; **(3) Execute** — score, cross-check, and
label cluster-by-cluster down the hierarchy, with a report and your sign-off at each
level, then write back into your AnnData. No labels are produced until you've signed off
the assumptions; nothing is written to your file until the very end, after you confirm.
Tell the user up front that the run is **resumable** — everything is checkpointed to the
working dir, so a new session can pick up from the last step.

Resuming an interrupted run? Read `run_state.json`, `context.md`, `assumptions.md`,
and `plan.md` in the working directory first, then continue from
`progress.current_level`. Otherwise start at Stage 1.

---

### Stage 1 — Scene-setting, environment & tooling gate

**1a. Tooling gate — check this FIRST (fail fast, before the env build).** Two tiers:

- **`knowledgebase` MCP — hard requirement (strict stop).** The grounded marker/ontology
  evidence (BioContextAI's Knowledgebase MCP — PanglaoDB, Cell Ontology/OLS, HPA, Europe
  PMC, …). If not configured, **STOP and ask the user to set it up** — point them at
  `github.com/biocontext-ai/knowledgebase-mcp` (registry: `biocontext.ai`) — rather than
  degrading to web-only, which is the confident-but-unfounded annotation this skill exists
  to prevent.
- **`zotero` MCP — strongly preferred, not required.** If missing, say so, note the
  fallback to the `knowledgebase` Europe PMC tools, and offer to proceed.
- Verify whichever MCPs are present actually respond before proceeding.

**1b. Detect the compute environment (do this early — it shapes the plan).** Run
`cta detect-env` (stdlib-only — works with any python). It reports the accelerator
(CUDA / Apple-Silicon MPS / none), usable RAM (clamped to cgroup/scheduler limits), any
batch scheduler, and a **suggested profile**: `local_cpu` / `local_gpu` / `scheduler`.
**Confirm it with the user** — and you *must* ask in the one ambiguous case the probe flags:
a host with a scheduler on PATH but no local GPU is **either** a login/submit node whose
GPUs live on compute nodes (`scheduler`) **or** a plain CPU host (`local_cpu`). Record the
confirmed `compute` block in `run_state.json` (schema in `reference/artifacts.md`). This is
**detection only** — never submit a job, never install the GPU stack unasked. Full model:
`reference/compute_environments.md`.

**1c. Bootstrap the environment with pixi** (needed before `cta data inspect`). The skill ships a
`pixi.toml` with two environments; pixi builds an isolated env inside the skill dir
(`.pixi/`), **never** touching the user's project environment. From the skill directory:
- **CPU / Apple-Silicon (`local_cpu`):** `pixi install` (the `default` env: scVerse stack).
  Run every tool as **`pixi run cta <command> …`**.
- **GPU (`local_gpu`):** `pixi install -e gpu` (adds scvi-tools + cuML/cuDF GPU kNN +
  scimilarity; Linux + CUDA). Run tools as **`pixi run -e gpu cta <command> …`**.

**Build the env where the work runs.** pixi does not check for a GPU, so only run
`pixi install -e gpu` **where a GPU actually is**. On `scheduler`: `pixi install` (default)
on the login node — enough for detection + light steps — and `pixi install -e gpu`
**inside a GPU job**. On a cluster, put pixi's envs on scratch (not quota-limited home) so
the multi-GB CUDA env fits and both nodes see it: `pixi config set detached-environments
<scratch-path>`. Details: `reference/compute_environments.md`.

**1d. Scene-setting.** Inspect the object first to pre-fill inferred answers, then use
**AskUserQuestion** in **2–3 batched confirm-or-correct rounds** (don't interrogate
one item at a time) to establish:

- **Data:** path to the `.h5ad`; **the working-directory path — ALWAYS ASK, no
  default** (next to the data is often a poor choice).
- **Keys:** the cluster key (`obs`); the **embedding key** (`obsm`) to use
  everywhere; the **viz embedding** (UMAP/tSNE) for plots; for **spatial** data, the
  **obsm key holding spatial coordinates** (used by `cta cluster profile`).
- **Layers:** which layer is **raw counts** and which is **log-normalized** — ask
  explicitly, never infer. (Scoring runs on log-normalized; counts are needed only for
  the normalization fingerprint and optional slim caching.)
- **Assay:** dissociated vs spatial; chemistry; for spatial, the **platform and
  panel** (and panel size) — confirm even when inferable from `obs`.
- **Biology:** organism, tissue, developmental/experimental context (e.g. a
  perturbation whose expected effect is a *readout*, not an assumption).
- **Composition covariates — ASK explicitly:** which categorical `obs` columns should
  the per-level reports show cluster composition over (e.g. condition / timepoint /
  batch / donor)? Don't infer silently — surface the candidate categorical `obs`
  (from `cta data inspect`) and have the user pick the ones they care about. These drive
  both `cta cluster profile --cat-cols` (investigate-before-`Unknown`) and the report's
  `cta report build --split-by` (comma-separated → one composition panel each). Record
  them in `run_state.json` `cat_cols`. `cta report build` automatically reuses each
  covariate's **original colours** from `adata.uns['<col>_colors']` when present —
  `cta data inspect` reports which columns have them, so confirm the user is happy with
  those colours (or note none exist). Do NOT assume specific column names.
- **Label-vocabulary contract:** reuse a specific paper's labels? a controlled
  ontology (Cell Ontology / CellxGene)? or a free hierarchy with explicit per-level
  semantics (e.g. L1=compartment, L2=lineage, L3=type, L4=state — define the levels
  to fit the tissue)? Whether to normalize final labels to **CL IDs** (recommended).
- **Reference datasets:** any to map against? Capture, per reference: **path**, the
  **`obs` label column(s)** to transfer, **whether its X is raw counts or
  log-normalized** (ask — `cta crosscheck reference-map --ref-counts`), its **gene-ID
  convention** (Ensembl vs symbols, and any symbol var-column), and an optional **`obs`
  column to stratify the subsample by** (e.g. its cell-type column). These feed
  `cta crosscheck reference-map` (CellMapper) and the CellTypist tissue match.
  - **Reference gene IDs are aligned to the query's symbols automatically.** Because the
    canonical query is symbol-indexed, `reference-map` and `markers assemble --reference`
    rename the reference's `var_names` to symbols (auto-detected symbol var-column,
    `feature_name`/`gene_symbols`/…) before intersecting — so an Ensembl reference (the
    common CELLxGENE case) no longer shares ~0 genes. If the reference is Ensembl with **no**
    symbol var-column, build the map via the knowledgebase MCP exactly as for the query and
    pass `--ref-symbol-map ensembl_to_symbol.tsv` (or name the column with `--ref-symbol-col`).
- **Marker sources:** preferred references, the user's own atlases, marker lists.
- **Write-back target:** the `obs` key/prefix for the final labels (default
  `cta_L1`..`cta_L4`) — confirm now; the actual write is gated to the END of Stage 3.

**1e. Inspect, then canonicalize.** First run `cta data inspect` to verify every
key/layer exists and capture `n_obs`/`n_vars`, cluster sizes, layer stats, `obsm` keys, a
**normalization fingerprint**, the **gene-ID convention** (`gene_ids.var_names_look_like`
+ any symbol var-column), **`.raw`** contents, the candidate **`categorical_obs`** (with
which have `uns` colours), and a **`targeted_panel`** flag. Reconcile any mismatch with the user.

Then **always run `cta data canonicalize`** (the default Stage-1 data step) to write the
one canonical working sidecar (`<stem>_cta.h5ad`) every downstream command transparently
reads — so no step has to re-resolve input variability. It standardizes the object to:
**symbol `var_names`** (Ensembl IDs kept in `var['ensembl_id']`), **counts in
`layers['counts']`** (pulled from a layer or from `.raw`), **log-norm in `X`**, and the
chosen embedding(s) + cluster key. Pass the keys you confirmed
(`--cluster-key`/`--embedding-key`/`--viz-key`, `--lognorm-layer`, `--counts-layer`).

**Gene symbols — the Ensembl case.** When `inspect` reports `var_names_look_like:
ensembl`, canonicalize needs a symbol source: it auto-detects a symbol var-column
(`feature_name`/`gene_symbols`/…); if `inspect` found none, **build the map via the
knowledgebase MCP** — call `get_ensembl_id_from_gene_symbol` (symbol→Ensembl, its natural
direction) for your candidate **marker** symbols, write a `symbol_map.tsv`
(`ensembl<TAB>symbol`, one pair per line), and pass it to canonicalize as
`--symbol-map symbol_map.tsv`. Only the marker-relevant genes need renaming for
panel-gating to match; the rest keep their Ensembl id.
Write `context.md` and seed `run_state.json` (schemas in `reference/artifacts.md`).
**If `targeted_panel` is true, write a short panel-capability note in `context.md`** —
which expected identities this panel **cannot** resolve because their canonical markers
are off-panel (sets it apart from biology you simply didn't find); revisit it as
`panel_gating_<level>.csv` reports dropped/uncallable sets per level. This sets
expectations honestly and prevents reading panel gaps as negative biology.
(`cta data canonicalize` also writes the slim sidecar, so later scripts never reload the
full file — there is no separate slim-cache step.)

**Hard stop on normalization.** If `cta data inspect` reports anything under `fatal`
— in particular a declared log-normalized layer that fails the fingerprint (e.g.
`log1p` of *un-normalized* counts, raw counts, or scaled data) — **STOP**. Do not
compute markers. Wrong normalization makes every downstream marker track sequencing
depth / cell size rather than identity, producing confident-but-wrong calls that no
later gate can catch. Surface the specific verdict, and have the user point at the
correct layer (or normalize) before continuing. Treat this with the same strictness
as the missing-MCP gate.

---

### Stage 2 — Plan

With the data understood:

1. **Explore** the object from the `cta data inspect` report — cluster sizes,
   existing annotations/predictions already in `obs`, embeddings, QC signals.
   Surface any QC red flags (Gate principle: alert, don't fix).
2. **Gather prior knowledge** for the tissue/context via the knowledge stack
   (Zotero first if configured, then the knowledgebase MCP, CellTypist models, curated DBs, the
   user's atlases). See `reference/knowledge_sources.md`.
3. **Branch the workflow on the compute profile** (from Stage 1b; full matrix in
   `reference/compute_environments.md`). On `local_cpu` the plan is exactly as it always has
   been — say so, and don't burden the user with GPU options. On `local_gpu` / `scheduler`,
   decide and record in `plan.md`: the **reference strategy** (map onto the user's atlas; pull
   a tissue-matched **shipped scANVI model** from the scvi-tools hub if one fits; **train**
   one with `cta reference train-model`; or a larger Census slice), whether to use the **GPU
   kNN backend** for `cta crosscheck reference-map`, and the **execution mode** — on `scheduler`, mark the
   GPU-heavy steps "submit via your environment's job-submission mechanism" with their exact
   standalone commands, and never embed a submission command yourself.
4. **Draft the hierarchy** and the per-level vocabulary, and outline the workflow.
   Write `plan.md`. Present it to the user.

---

### Stage 3 — Execute (top-down, recursive)

This is the executable loop, per hierarchy level and recursively within each parent group.
Rationale, the two gates, the card schema, critic design, and sub-cluster & panel rules:
`reference/methodology.md` (read it once at the start of this stage).

**At the start of every level/parent-group iteration, orient the user** — state your
position in the hierarchy and the size of the work, e.g. *"Now annotating L2 within the
Neuron compartment — 9 clusters, ~120k cells. I'll score markers, run one cross-check,
bring you the assumptions ledger, then a report."* Do this every iteration, not just the
first, so a deep recursion never leaves the user lost.

1. **Assemble priors** → candidate types + markers for this level/context. Optionally
   seed the marker JSON with `cta markers assemble` (from a CellTypist model
   and/or a reference's `rank_genes_groups`), then curate.
2. **Panel-gate** every marker set against `adata.var_names`; record dropped genes.
3. **Score** — two complementary views:
   - *bottom-up (what defines each cluster):* `cta markers compute`
     (`rank_genes_groups` wilcoxon + COSG specificity) → marker tables;
     `cta markers score` (`aucell` + pyUCell + DE-overlap → `scores_aucell_z` +
     a `crosscheck_<level>` consolidation).
   - *top-down (how your candidate markers land):* `cta markers profile` — score the
     **curated panel you assembled in step 1** across clusters into per-cluster pct + mean
     matrices (`panel_pct_<level>.csv` / `panel_mean_<level>.csv`) and a dot plot. This is the
     workhorse you reason from cluster-by-cluster ("is marker X on in cluster Y?") and it
     feeds the cards directly via `cta report scaffold-cards --evidence panel_pct_<level>.csv`. Run it
     once per level on the level's marker JSON.

   On targeted panels read `crosscheck`/`scores_aucell_z`, not raw AUCell — COSG specificity +
   the reference cross-check are primary (`reference/marker_tools.md`).
4. **Cross-check** — run **exactly one** primary automated hypothesis (the second only to
   break a tie); flag agreement/disagreement, never overwrite (`reference/automated_tools.md`):
   - **In-object prediction (preferred when one exists):** if `obs` already carries a
     label-transfer / classifier prediction that fits the level (e.g. a previously-mapped
     atlas `*_pred`, a CellTypist `predicted_labels`), use `cta crosscheck obs-pred`
     — free, already reconciled with the object, no recompute.
   - else **Reference-model mapping (GPU profiles — scANVI/scArches):** when on
     `local_gpu`/`scheduler` and a reference model fits the level — a **shipped** scANVI model
     pulled from the scvi-tools hub (`cta reference map-scarches --hf-repo …`) or one you trained
     on the user's atlas (`cta reference train-model` → `cta reference map-scarches --model-dir …`).
     scArches surgery transfers labels **with per-cell uncertainty**; prefer this when a
     well-matched model exists. (`reference/compute_environments.md`.)
   - else **kNN reference mapping:** `cta crosscheck reference-map` (CellMapper; CCA cross-modality,
     PCA same-modality; builds its own joint embedding). On a GPU profile add
     `--knn-method rapids` (needs cuml-cu12) to scale to millions of cells.
   - else **CellTypist (tissue-gated):** `cta crosscheck celltypist list` → `predict` only if a
     model fits; **omit if none does** (don't force an ill-fitting model).
   - else **Census reference (fallback):** if the user has no atlas and no CellTypist
     model fits, `cta reference fetch-census` pulls a capped, tissue-matched
     subset from CELLxGENE Census (optional `cellxgene-census` extra; network-dependent)
     to feed `cta crosscheck reference-map` — so every tissue still gets a reference cross-check.
     Census labels are author-supplied and heterogeneous, so weight it as corroborating
     signal, not ground truth (`reference/automated_tools.md`).
5. **Gate A — assumptions ledger (living).** Before any labels at this level, extend
   `assumptions.md` with the level's marker→type mappings, priors, and panel caveats.
   **First** sign-off (before any labels exist) is **blocking** — render the ledger, and
   when you **AskUserQuestion** (*Approve / Revise / Add constraint*) give the user one
   line on *why* this gate exists (these are the assumptions every label at this level
   rests on; catching a wrong one here prevents confidently-wrong labels downstream) and
   what each answer does (Approve → start assigning labels; Revise → tell me which
   mapping to change; Add constraint → I'll fold in a rule, e.g. "treat marker X as
   ambiguous"). Record in `run_state.json` `gates.signed_off_levels`. Later levels:
   append, and re-prompt only on a **materially new** assumption — but in that level's
   report/checkpoint, **tell the user what was appended** so silent additions stay
   visible. Full rules: `reference/methodology.md` §2.
6. **Assign — filling the labels CSV *is* the assignment.** The per-level table
   `labels/annotation_<level>.csv` is the single source of truth: fill
   `label,cl_id,confidence,rationale,rejected` per cluster *there* (label normalized to a
   **CL ID**), then `cta report scaffold-cards --evidence panel_pct_<level>.csv` **renders**
   the cards from it — embedding the ACTUAL numbers behind each call and never wiping your
   reasoning on re-scaffold (a resumed run recovers it from this one file). For a cluster
   you can't call cleanly: on mutually-exclusive marker co-expression **sub-cluster and
   recurse** (`cta cluster subcluster` — Leiden on the anchor embedding + per-sub
   marker/QC/composition summary); on irreconcilable conflict **mark `Unknown`** (Gate B) —
   but **investigate first** with `cta cluster profile` (QC medians + user-named categorical
   crosstabs + optional spatial), since good-quality cells that resist a clean call are
   often a real early/transitional state, not an artifact. Set `confidence` by judgement,
   **calibrated against `cta cluster confidence`** (a COSG-margin + agreement proxy — high
   confidence on a weak-margin cluster is a flag to re-examine). **Nesting is mandatory:**
   each level's labels must **strictly refine the level above** (every finer label maps to
   exactly one coarser label); if a regional split (e.g. ventral telencephalic) cuts across
   an L1 compartment (NPC vs Neuron), *qualify the label by compartment*
   (`Ventral telencephalic NPC` vs `… neuron`) — never reuse one label across two parents.
   Verify any time with `cta report write-back --dry-run` (runs
   `cta_io.check_hierarchy_nesting`, refusing a non-nested hierarchy). Why the CSV is the
   source of truth, and the full card schema: `reference/methodology.md` §4.
7. **Critic pass.** Spawn independent critic subagents (Agent tool) that read the
   exported **raw tables** (not just the card), re-derive markers, fetch ≥1 external
   hypothesis, and try to **refute** each label — biologically **and hierarchically**:
   is the label a coherent refinement of its parent path (flag maturation/regional
   mismatches like an "IPC"/"neuron" L3 under an "NPC" L1 — the kind the deterministic
   nesting check can't catch)? Reconcile; revise or downgrade. (Structural nesting itself
   is enforced in code — `cta_io.check_hierarchy_nesting` / `cta report write-back`.)
8. **Report, gate & descend.** Build the level's HTML report with
   `cta report build` (pass `--adata --cluster-key --viz-key`, plus
   `--embedding-key` for the similarity-ordered palette, to embed the numbered
   cell-map + marker dotplot; `--split-by <cols>` — the user's `cat_cols`,
   comma-separated — adds one composition panel per covariate, reusing `uns` colours).
   **Run it with `--strict`: this is the level-completeness gate** — it refuses to pass
   (exit 1) unless every cluster has a non-blank label AND a filled (non-TODO) rationale,
   the report exists, and the assumptions gate is signed. That is what stops a level from
   *looking* finished while its reasoning was silently skipped. **Render the report ONCE
   (with `--adata`).** The first `--strict` run before sign-off is *expected* to exit 1
   (gate unsigned) — the report is still written for you to review. After you record the
   sign-off in `run_state.json`, **clear the gate with `cta report build --check-only --strict`,
   which re-checks read-only and writes nothing — do NOT re-run the renderer to flip the gate**
   (a rebuild that omits `--adata` would needlessly reload the data; figures themselves are
   embedded from the durable PNGs in `plots/`, so they survive such a rebuild either way).
   Then update `run_state.json`. **Present the report and pause:** point the user at the
   flagged clusters, and let them **approve descending — and into which parent groups, in
   what order — or request revisions.** Don't begin the next level in the same turn unless
   they've said to proceed (or pre-authorized auto-proceed at Stage 2). If a level lands
   more than ~half its clusters in `Unknown`, stop and consult the user
   (`reference/methodology.md`). On approval, recurse at the next level.

**At the end (after the last level): write-back.** Run `cta report write-back --dry-run`
first and **show the user its output and the hierarchy-nesting verdict** — that is what
they're approving. Pass `--embedding-key` (the anchor embedding) so the saved labels get
the **same similarity-ordered category order and colour palette the reports used**: the
script writes `adata.uns['<col>_colors']`, so a later `sc.pl.umap(color="cta_L1")` shows
the colours the user signed off on rather than fresh arbitrary ones. The real write adds
the labels as `obs` columns **in-place** by default (atomic temp-file replace); offer
`--output <path>` to write a copy instead if the user prefers not to mutate the original.
Never write without explicit sign-off.

When condition/contrast DE comes up (e.g. comparing conditions within a type), that is **Job B** —
pseudobulk + PyDESeq2 with a replication requirement, never per-cell. See
`reference/differential_expression.md`; do not conflate it with marker ID.

## Reference map

| File | When to read it |
| --- | --- |
| `reference/methodology.md` | The *why* behind Stage 3: the two gates, evidence-card & critic design, sub-cluster decision, panel rules, termination/failure handling. (The executable loop is Stage 3 here.) Read at the start of Stage 3 (Execute). |
| `reference/artifacts.md` | Working-dir layout, `context.md`/`run_state.json` schemas, report format. Read in Stage 1. |
| `reference/compute_environments.md` | The three compute profiles, environment detection, the scheduler-handoff convention, GPU caveats. Read in Stage 1 (after `cta detect-env`). |
| `reference/tool_registry.md` | **Canonical** step→tool→GPU-requirement→profile table. Consult when choosing which tool to run for a step, or whether it needs a GPU. |
| `reference/marker_tools.md` | decoupler 2.x, pyUCell, COSG, snapseed; scanpy annotation idioms. |
| `reference/differential_expression.md` | Job A (markers) vs Job B (conditions); exact `rank_genes_groups` call. |
| `reference/automated_tools.md` | CellTypist, CellMapper, SingleR; what to skip and why. |
| `reference/knowledge_sources.md` | MCP coverage, CL normalization, Python loaders, atlas references. |

## Commands (`cta`)

Every tool is a subcommand of the `cta` CLI, run via **`pixi run cta <command> …`** (add
`-e gpu` for the GPU tools) after `pixi install` (Stage 1c). Each writes durable artifacts
into the working directory and holds no cross-call state — run `cta <command> --help` for
exact arguments. This table is a quick **purpose index**; for which tool serves which step
and **whether it needs a GPU**, `reference/tool_registry.md` is the canonical map.

| Command | Purpose |
| --- | --- |
| `pixi install [-e gpu]` | Bootstrap the env (Stage 1c): `default` = CPU scVerse core; `-e gpu` adds the GPU stack. (Not a `cta` command — pixi builds the env.) |
| `cta detect-env` | Stage 1: detect the compute environment (CUDA/MPS, RAM, scheduler) → suggested profile. Detection only; no jobs, no installs. |
| `cta data inspect` | Stage 1: verify keys/layers, gene-ID convention, `.raw`, normalization fingerprint → dump structure to `context`. |
| `cta data canonicalize` | Stage 1 (default): write the one canonical sidecar every downstream step reads — symbol `var_names` (+ kept Ensembl map), counts in `layers['counts']` (from a layer or `.raw`), log-norm `X`, slimmed embeddings/obs. Also avoids reloading the huge file. |
| `cta markers assemble` | Optional: seed `markers_<level>.json` from a CellTypist model and/or a reference's `rank_genes_groups`. |
| `cta markers compute` | `rank_genes_groups` (wilcoxon, tie_correct) + COSG → marker tables (bottom-up: what defines each cluster). |
| `cta markers profile` | Top-down complement: score a CURATED gene panel across clusters → per-cluster pct + mean matrices + dot plot. The workhorse you reason from; feeds the cards via `--evidence`. |
| `cta markers score` | decoupler `aucell` + pyUCell + DE-overlap; writes z-scored AUCell + a consolidated `crosscheck`. |
| `cta crosscheck celltypist` | `list` the remote model index (tissue-gated, no downloads); `predict`/`markers` for a matched model only. |
| `cta crosscheck reference-map` | CellMapper kNN label transfer from a reference (joint CCA/PCA; CPU pynndescent, or GPU `--knn-method rapids` via cuml) — cross-check when no in-object prediction fits. |
| `cta reference train-model` | GPU/large-memory: train (or refine) a scANVI reference model on a labeled atlas (scvi-tools). Reference side of the train-then-map workflow. |
| `cta reference map-scarches` | GPU/large-memory: map the query onto a scANVI reference (a model you trained, or a shipped scvi-tools hub model) via scArches surgery — labels + per-cell uncertainty, in the `refmap_*` schema. |
| `cta crosscheck scimilarity` | Optional large-memory cross-check: zero-shot per-cell labels via SCimilarity (~64GB RAM + model bundle). Foundation model → additional hypothesis only, never primary. |
| `cta reference fetch-census` | Optional: pull a capped (or, on a large-memory profile, larger) tissue-matched reference subset from CELLxGENE Census (no full-atlas download) to feed `cta crosscheck reference-map` when the user has no atlas and no CellTypist model fits. Needs the `cellxgene-census` extra + network. |
| `cta crosscheck obs-pred` | Cross-check from an EXISTING in-object prediction column (e.g. a previously-mapped atlas `*_pred`) — no recompute. Prefer over `cta crosscheck reference-map` when such a column exists and fits the level. |
| `cta cluster subcluster` | Refine one/more clusters: Leiden on the anchor embedding + per-sub marker-fraction / QC / composition summary. Standardized sub-clustering (in scope). |
| `cta cluster profile` | QC medians + user-named categorical crosstabs + optional spatial — investigate before `Unknown`. |
| `cta cluster confidence` | Advisory: per-cluster confidence proxy from COSG specificity margin + cross-check agreement — calibrate hand-set confidences against it. |
| `cta report scaffold-cards` | Render per-cluster evidence cards FROM the labels CSV (`rationale`/`rejected`/`label`/`confidence` columns = single source of truth, so re-scaffold never wipes reasoning); `--evidence <tables>` embeds the ACTUAL numbers behind the call. Sanitizes `/` in ids. |
| `cta report build` | Per-level HTML report from tables + cards; (re)generates the numbered cell-map + marker dotplot + one composition panel per `--split-by` covariate **only with `--adata`**, but always embeds whatever PNGs are in `plots/` (so a no-`--adata` rebuild keeps the figures). `--strict` = the level-completeness gate (labels + filled rationale + signed assumptions gate) before descending; add `--check-only` to re-evaluate the gate read-only (no render, no write) after a sign-off. |
| `cta report write-back` | Final step (gated): write the per-level labels into `obs` in-place (`--dry-run` first). |
