# Reference: Automated classifiers, label transfer & reference mapping

**Profile-gated (see `reference/compute_environments.md`).** The CPU default
(`local_cpu`) uses lightweight pretrained classifiers and small reference-mapping — no GPU,
no from-scratch training, no multi-GB atlas downloads. On a GPU / large-memory profile
(`local_gpu` / `scheduler`) the heavier tier below is unlocked: reference-model training +
scArches mapping, GPU kNN, large atlas / Census, and an optional foundation-model
cross-check.

**Core principle (does NOT change with the hardware): every automated call is a HYPOTHESIS,
not a verdict.** (This is the single-cell best-practices consensus.) Claude's marker
reasoning is the arbiter — a bigger model earns one more independent opinion to weigh, not
more authority. When an automated method disagrees with the marker evidence, **flag the
disagreement** — do not silently overwrite. Disagreement is signal: it often marks
ambiguous progenitor / transitional states that are biologically real and important.

## Worth including (all profiles — the CPU core)

- **CellTypist** (`pip install celltypist`) — logistic-regression classifier,
  per-cell with optional `majority_voting` over clusters. ~60 pretrained models, each a
  few MB. **Tissue-gate it — don't download blind:**
  - The model index (name + `details` + `No_celltypes` + `source`) is available WITHOUT
    downloading via `celltypist.models.get_models_index(force_update=True)` (the JSON at
    `celltypist.cog.sanger.ac.uk/models/models.json`). Use `cta crosscheck celltypist list
    [--contains <tissue>]` to read it and decide whether any model matches.
    (⚠️ `models.get_all_models()` lists only *already-downloaded* models, and
    `models_description(on_the_fly=True)` is unreliable across versions — use the index.)
  - `celltypist.annotate(model=...)` needs the `.pkl` **downloaded first** (Model.load
    does not auto-fetch); `cta crosscheck celltypist predict` downloads just the chosen model.
  - Real, valid tissue-matched models exist (e.g. `Developing_Human_Brain` = 129 cell
    types, first-trimester human brain; `Adult_Human_PrefrontalCortex`;
    `Developing_Human_Hippocampus`). **If none matches the tissue, OMIT CellTypist** —
    don't force an ill-fitting model; use the reference mapping below instead.
  - `model.extract_top_markers()` mines markers **without predicting** (priors).
  - *Caveat:* on a limited/targeted panel most model genes are missing → accuracy
    degrades; the output is a hypothesis to reconcile, never an overwrite.
- **CellMapper** (`pip install cellmapper`) — kNN label / expression / embedding
  transfer; the **preferred reference cross-check**, via `cta crosscheck reference-map`.
  It **builds the joint embedding itself** from the shared genes —
  `compute_joint_pca()` (same-modality) or `compute_fast_cca()` (cross-modality, e.g. a
  dissociated scRNA reference onto a targeted spatial query) — so it does **not** need a
  pre-existing joint embedding. CPU kNN via **pynndescent** (scales to millions); gives
  per-cell predictions **with confidence** and `compute_presence_score()`. (Earlier
  guidance to "skip if there's no shared embedding" was wrong for this tool.)
  - **Gene-ID alignment:** the reference's `var_names` are aligned to the query's symbol
    convention before the shared-gene intersection (auto-detected symbol var-column, or
    `--ref-symbol-col` / `--ref-symbol-map` — the skill builds the map via the knowledgebase
    MCP). An Ensembl-indexed reference therefore maps onto a symbol-canonical query without
    manual pre-processing. `markers assemble --reference` applies the same alignment so its
    DE marker names panel-gate against the query. (Symbol coverage is partial — ~25-30% of
    full-transcriptome genes have no symbol and stay Ensembl; expected, not a failure.)
- **CELLxGENE Census as an on-demand reference** (`cellxgene-census`, optional extra) —
  when the user has **no atlas of their own and no CellTypist model fits**, this is the
  fallback that keeps the reference cross-check alive. `cta reference fetch-census`
  opens the remote TileDB-SOMA store (no atlas download), reads only the matching obs to
  pick a **capped, stratified, fixed-seed** subset, fetches just those cells, and writes a
  small reference `.h5ad` that feeds `cta crosscheck reference-map`. CL/UBERON-filtered, so it lines
  up with the skill's CL-normalized labels. *Caveats:* network-dependent (fails cleanly
  offline); Census `cell_type` is author-supplied and heterogeneous across contributing
  datasets, so a transferred label is a coarser/noisier hypothesis than one from a single
  curated atlas — corroborating signal, not ground truth.
- **SingleR** (Python `singler` + `celldex`, BiocPy, **no R needed**) —
  correlation against a celldex reference (`hpca`, `mouse_rnaseq`, etc.), per-cell
  with plurality vote per cluster.
  - *Caveat:* ships cp313 wheels (not cp314); densify the test matrix before scoring
    (a sparse transpose mis-annotates).
- **Hand-rolled mini-consensus** — run 2–3 light methods and flag where they disagree
  for a certainty / "needs-review" signal. (A full consensus engine like popV is overkill
  for this skill's single-cross-check design — see "Skip" below.)
- **`scanpy.tl.ingest`** — zero-dependency kNN baseline transfer when you have a
  reference `AnnData`.

## GPU / large-memory tier (profiles `local_gpu` / `scheduler`)

Unlocked once the environment is detected and confirmed (`reference/compute_environments.md`).
Each is still a HYPOTHESIS, held to the same bar as the CPU cross-check.

- **scANVI reference model + scArches surgery** — the core GPU win, and the canonical
  "train/refine a reference model, then map the query onto it" workflow.
  `cta reference train-model` trains scVI→scANVI on a labeled atlas (semi-supervised:
  learns a batch-corrected latent space *and* a label classifier);
  `cta reference map-scarches` then maps the query on by **freezing the reference net and
  training a few query-specific weights** — so the query lands in the reference latent space
  and gets per-cell predictions **with calibrated uncertainty**, without retraining and
  without needing the reference's cells. Reference model can be one you trained, or a
  **shipped model pulled from the scvi-tools HuggingFace hub** (`scvi.hub`) when a
  tissue-matched one exists — preferred when available (no training). Writes the same
  `refmap_<level>_<col>.csv` schema, so the cards/report fold it in like any cross-check.
  scvi-tools also runs on Apple-Silicon **MPS** — so a Mac can do this slowly as a deeper
  local option (offer it; don't default to it).
- **GPU kNN reference mapping** — `cta crosscheck reference-map --knn-method rapids`
  scales the existing CellMapper cross-check to **seconds over millions of cells**. Same
  conceptual transfer, GPU backend (needs `cuml-cu12` from the `gpu` pixi env — CellMapper's
  rapids backend calls cuML directly, *not* rapids-singlecell). The `faiss-gpu` backend needs
  a working faiss-gpu build (the conda one; the pip wheel is often ABI-broken).
- **Large reference atlases / Census slices** — with the memory headroom, lift the
  `cta reference fetch-census --max-cells` cap and prefer a full curated atlas (or a shipped
  hub model) over the laptop-scale slice (`reference/knowledge_sources.md`).
- **Foundation models — OPTIONAL, additional cross-check only.** 2026 benchmarks repeatedly
  show single-cell foundation models (scGPT, Geneformer, UCE, scFoundation) **rarely beat
  scANVI or even simple baselines for annotation**, and degrade out-of-distribution on small
  targeted panels — so they are never the primary annotator here. The two that fit a local,
  data-stays-put design and emit a usable per-cell label are **SCimilarity**
  (`cta crosscheck scimilarity`; zero-shot kNN against a large reference, needs ~64 GB RAM +
  a multi-GB bundle) and **scPRINT** (CPU-capable, emits ontology IDs) — wire either in only
  as one more independent opinion to reconcile.

## Skip regardless of profile

- **scGPT / Geneformer / UCE / scFoundation as primary annotators** — embedding-or-fine-tune,
  research-grade, and don't earn their keep over scANVI for annotation per the benchmarks
  above. Reach for them only if fine-tuning on a labeled reference is explicitly on the table.
- **popV** — a 10-method consensus; overkill for this skill's single-cross-check design.
- **External LLM annotators** (e.g. GPTCelltype, mLLMCelltype) — redundant here,
  because **Claude is the annotator**. Their *conventions* are worth adopting, though:
  - annotate per-cluster;
  - pass species + tissue context into the reasoning;
  - harmonize labels across samples;
  - use a controlled vocabulary;
  - request an explicit confidence field.
  Likewise, automatic color/ordering beautification of an `obs` annotation is something
  Claude does natively — no external LLM dependency needed.

## On hosted LLM annotators (e.g. CyteType) — not used

A hosted annotation service is not used as a component: it is closed-source, sends the
expression matrix off-machine, and has no local path — all of which conflict with this
skill's on-machine, data-stays-put design, and it is redundant with "Claude annotates."
Two of its workflow patterns are nonetheless worth adopting:

- 5-role decomposition: contextualize → generate-and-test *competing* hypotheses per
  cluster → review/flag weak evidence → literature-ground → summarize.
- self-generated 0–1 confidence scores that flag low-trust clusters → fold into the
  evidence cards.

## References for transfer

The strongest reference is usually the user's own curated atlas for their
tissue/context — when one exists — rather than a generic frozen database. Compute
markers directly off it with `rank_genes_groups`. See `knowledge_sources.md`.
