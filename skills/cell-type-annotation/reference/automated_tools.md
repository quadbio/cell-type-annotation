# Reference: Automated classifiers, label transfer & reference mapping

CPU / Apple-Silicon constraints: no GPU, no from-scratch training, no multi-GB atlas
downloads. Lightweight pretrained classifiers and small reference-mapping are fine.

**Core principle: every automated call is a HYPOTHESIS, not a verdict.** (This is the
single-cell best-practices consensus.) Claude's marker reasoning is the arbiter.
When an automated method disagrees with the marker evidence, **flag the
disagreement** — do not silently overwrite. Disagreement is signal: it often marks
ambiguous progenitor / transitional states that are biologically real and important.

## Worth including (CPU-local)

- **CellTypist** (`pip install celltypist`) — logistic-regression classifier,
  per-cell with optional `majority_voting` over clusters. ~60 pretrained models, each a
  few MB. **Tissue-gate it — don't download blind:**
  - The model index (name + `details` + `No_celltypes` + `source`) is available WITHOUT
    downloading via `celltypist.models.get_models_index(force_update=True)` (the JSON at
    `celltypist.cog.sanger.ac.uk/models/models.json`). Use `run_celltypist.py list
    [--contains <tissue>]` to read it and decide whether any model matches.
    (⚠️ `models.get_all_models()` lists only *already-downloaded* models, and
    `models_description(on_the_fly=True)` is unreliable across versions — use the index.)
  - `celltypist.annotate(model=...)` needs the `.pkl` **downloaded first** (Model.load
    does not auto-fetch); `run_celltypist.py predict` downloads just the chosen model.
  - Real, valid tissue-matched models exist (e.g. `Developing_Human_Brain` = 129 cell
    types, first-trimester human brain; `Adult_Human_PrefrontalCortex`;
    `Developing_Human_Hippocampus`). **If none matches the tissue, OMIT CellTypist** —
    don't force an ill-fitting model; use the reference mapping below instead.
  - `model.extract_top_markers()` mines markers **without predicting** (priors).
  - *Caveat:* on a limited/targeted panel most model genes are missing → accuracy
    degrades; the output is a hypothesis to reconcile, never an overwrite.
- **CellMapper** (`pip install cellmapper`) — kNN label / expression / embedding
  transfer; the **preferred reference cross-check**, via `scripts/reference_map.py`.
  It **builds the joint embedding itself** from the shared genes —
  `compute_joint_pca()` (same-modality) or `compute_fast_cca()` (cross-modality, e.g. a
  dissociated scRNA reference onto a targeted spatial query) — so it does **not** need a
  pre-existing joint embedding. CPU kNN via **pynndescent** (scales to millions); gives
  per-cell predictions **with confidence** and `compute_presence_score()`. (Earlier
  guidance to "skip if there's no shared embedding" was wrong for this tool.)
- **CELLxGENE Census as an on-demand reference** (`cellxgene-census`, optional extra) —
  when the user has **no atlas of their own and no CellTypist model fits**, this is the
  fallback that keeps the reference cross-check alive. `scripts/fetch_census_reference.py`
  opens the remote TileDB-SOMA store (no atlas download), reads only the matching obs to
  pick a **capped, stratified, fixed-seed** subset, fetches just those cells, and writes a
  small reference `.h5ad` that feeds `reference_map.py`. CL/UBERON-filtered, so it lines
  up with the skill's CL-normalized labels. *Caveats:* network-dependent (fails cleanly
  offline); Census `cell_type` is author-supplied and heterogeneous across contributing
  datasets, so a transferred label is a coarser/noisier hypothesis than one from a single
  curated atlas — corroborating signal, not ground truth.
- **SingleR** (Python `singler` + `celldex`, BiocPy, **no R needed**) —
  correlation against a celldex reference (`hpca`, `mouse_rnaseq`, etc.), per-cell
  with plurality vote per cluster.
  - *Caveat:* ships cp313 wheels (not cp314); densify the test matrix before scoring
    (a sparse transpose mis-annotates).
- **popV** (`fast` mode, ~5 min / 50k cells on CPU) **or** a hand-rolled
  mini-consensus (run 2–3 light methods, flag where they disagree) for a
  certainty / "needs-review" signal. Bundled references are adult Tabula Sapiens,
  not developmental.
- **`scanpy.tl.ingest`** — zero-dependency kNN baseline transfer when you have a
  reference `AnnData`.

## Skip for a CPU-local skill

- **Foundation models** (scGPT, Geneformer, UCE, scFoundation; scPRINT is the only
  CPU-honest one but still slow) — GPU/cloud-bound, and out-of-distribution on a
  small targeted panel. Not worth it here.
- **scANVI / scArches onto large atlases** — GPU-leaning and large downloads.
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
skill's CPU-local, data-stays-put design, and it is redundant with "Claude annotates."
Two of its workflow patterns are nonetheless worth adopting:

- 5-role decomposition: contextualize → generate-and-test *competing* hypotheses per
  cluster → review/flag weak evidence → literature-ground → summarize.
- self-generated 0–1 confidence scores that flag low-trust clusters → fold into the
  evidence cards.

## References for transfer

The strongest reference is usually the user's own curated atlas for their
tissue/context — when one exists — rather than a generic frozen database. Compute
markers directly off it with `rank_genes_groups`. See `knowledge_sources.md`.
