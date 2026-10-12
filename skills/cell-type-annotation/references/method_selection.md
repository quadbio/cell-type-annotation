# Method selection

You and the marker evidence are the annotator on every path. A cross-check is **one hypothesis to reconcile against
the markers**, never a verdict and never a silent overwrite. Run exactly one primary cross-check per level; a second
only to break a tie. Disagreement with the markers is signal: it often marks the transitional states that matter
most.

## Choosing

The ranking and every gate live in `cell_type_annotation.routing`, as data. Describe the query, record the result in
`reports/plan.md`, and carry the primary route's caveat into the evidence cards:

```python
from cell_type_annotation.routing import evaluate

for r in evaluate(species="human", stage="adult", state="normal", technology="dissociated", profile="cpu", atlas=True):
    print(r["rank"], r["method"], r["status"], r["reason"])
```

It does not decide whether to cross-check (always), whether the answer is right (the markers, then the critics), the
vocabulary (the Stage 1 contract), or a cluster's `state` (QC; a cross-check only corroborates one).

**Stage** and **state** have no data probe and gate the pan-body human models: establish them with the user in
Stage 1. `unknown` disqualifies those models rather than guessing.

The block below is generated from the code (`routing.update_page`); a test fails when they drift. Change the
routing in the code, never here.

<!-- BEGIN GENERATED -->
### Axes

| Axis | Values | How you determine it |
| --- | --- | --- |
| **species** | `human` · `mouse` · `other` · `unknown` | Stage 1 intake; corroborated by the gene-ID convention. |
| **stage** | `adult` · `developmental` · `organoid` · `unknown` | Stage 1 intake. There is no data probe for this: ask, do not infer from tissue. |
| **state** | `normal` · `disease` · `tumor` · `unknown` | Stage 1 intake for the design; QC deviation per cluster for quality and cycling. |
| **technology** | `dissociated` · `spatial_wt` · `spatial_panel` | A few hundred genes in `var` means a targeted panel; confirm the platform with the user. |
| **profile** | `cpu` · `gpu` | Where the cross-check will run, including a GPU job submitted through the scheduler. |
| **references** (flags) | `obs_pred` · `atlas` · `scanvi_model` · `celltypist_model` | `obs` listing; `celltypist.models.get_models_index(force_update=True)` filtered to the tissue. |

### Ranking

Evaluated top-down. The first route whose preconditions hold and whose disqualifiers are all
false is the primary cross-check. A disqualifier beats rank.

| # | Method | Use when | Why here |
| --- | --- | --- | --- |
| 1 | `obs_pred`: existing `obs` prediction | `obs` already carries a prediction that fits this level | Free and already reconciled with the object: no recompute, no new assumptions. |
| 2 | `scarches`: scANVI / scArches mapping | a GPU AND a well-matched scANVI reference model (trained or from the hub) | The strongest option when it fits: labels with calibrated per-cell uncertainty, in a reference latent space built for this tissue. A well-matched, curated reference beats any generic model. |
| 3 | `reference_map`: CellMapper on your reference | a labelled reference AnnData of your own (the user's atlas) | The user's own curated atlas for their tissue and context is usually the strongest reference available, and CellMapper builds the joint embedding itself, including cross-modality CCA for a targeted panel. |
| 4 | `celltypist`: CellTypist | a tissue- AND stage-matched pretrained model exists | A genuinely matched model is a strong, nearly free cross-check trained on exactly this tissue and stage, so it outranks a pan-body model. The gate is the whole point: stage matters as much as tissue. |
| 5 | `azimuth`: pan-human Azimuth | human, adult/postnatal, non-tumour, whole-transcriptome, and no matched reference or model above | One supervised model spanning 23 human tissues, so a single reference covers a broad or multi-tissue query that no tissue-specific model does. It is also the only cross-check here that returns a Cell Ontology id per cluster, a calibrated confidence, and trained `Unassigned` / `Doublet like cell` classes that feed the `state` axis. CPU-only and cheap (~50-60k cells/min, 84 MB model). |
| 6 | `scimilarity`: SCimilarity | broad / multi-tissue human adult query with no matched reference or model | An option for broad / multi-tissue human queries where no tissue-matched model exists. Ranked below Azimuth because it costs far more (9 GB bundle, 16-27 GB RAM) and returns a flat label space with no ontology ids, no calibrated confidence and no state signals. |
| 7 | `census`: Census slice, then CellMapper | nothing above fits: build a reference instead of going without one | Keeps a reference cross-check alive for any species and tissue, so no level goes unchecked. |

### Disqualifiers

| Method | Refused when |
| --- | --- |
| `obs_pred` | the column's provenance is unknown, or the method that produced it is itself disqualified here (species / stage / tissue / panel) |
| `scarches` | no CUDA GPU reachable |
| `reference_map` | fewer than 30 genes shared with the query after symbol alignment |
| `celltypist` | no tissue- and stage-matched model: omit CellTypist rather than force an ill-fitting one (a mismatched model collapses, it does not degrade gracefully); targeted imaging panel (most model genes missing) |
| `azimuth` | species not human, or not established (the model has a fixed human gene space); developmental / organoid query (trained on adult tissue; not designed for developmental stages); tumour expected (malignant cells were excluded from training); stage or state not established (no data probe for either: ask before routing); targeted imaging panel (missing panel genes are zero-filled, producing confident garbage) |
| `scimilarity` | species not human, or not established (the model has a fixed human gene space); developmental / organoid query (trained on adult tissue; not designed for developmental stages); tumour expected (malignant cells were excluded from training); stage or state not established (no data probe for either: ask before routing); targeted imaging panel (missing panel genes are zero-filled, producing confident garbage); under ~27 GB of usable RAM |
| `census` | offline (Census is a remote TileDB store) |

### Caveats that travel with the result

- **`obs_pred`**: It summarises someone else's prediction and inherits that method's disqualifiers. Establish what produced the column before trusting it; if you cannot, treat this route as unavailable.
- **`scarches`**: Stochastic: seed it, and still report predictions as a hypothesis.
- **`reference_map`**: On a targeted spatial panel this is the preferred route (cross-modality CCA).
- **`celltypist`**: Scale-sensitive: feed it log1p of counts normalized to 10,000 per cell.
- **`azimuth`**: Two structural limits. (1) It cannot be constrained to a tissue: it predicts over the full pan-body space, so on a single-organ query it can return a type from another organ; check the winning label against the tissue. (2) For lymph node and spleen its training labels came from the CellTypist organ atlases, so on immune tissue it is not independent of CellTypist. Its CL ids, calibrated confidence and state signals are useful however you weigh its labels, which also makes it a good tie-break below a matched model.
- **`scimilarity`**: Zero-shot foundation model: an additional opinion, never a primary annotator.
- **`census`**: Census `cell_type` is author-supplied and heterogeneous across datasets, so a transferred label is a coarser, noisier hypothesis than one from a curated atlas.
<!-- END GENERATED -->

## Why the gates are where they are

The disqualifiers come from the tool authors' stated scope. Pan-human Azimuth "was not designed to capture
developmental stages, disease-specific states, or cancer programs" (non-cancer disease usually maps onto the nearest
healthy identity) and is human-only; that sets the species, stage and tumour gates. The panel gate is structural:
pan-transcriptome models zero-fill missing genes, so a 300-gene panel covering ~5% of their input still returns
confident labels, with no visible symptom. SCimilarity's human-only gate follows from its fixed human gene order. The
ranking encodes a preference (a curated, tissue-matched reference over a generic one), cost, and what each tool
returns beyond a label; it is not a benchmark result.

## Recipes

Each summarizes per cluster with `prediction_summary(clusters, labels, confidence)` into `results/crosscheck_<level>.csv`
(per-cell predictions, if kept, go to `outputs/`). `mean_conf` means something different for every method: compare
it within a method, never across. Heavy methods run as jobs (the environment's own skill); dependencies go in the
analysis repo's `pixi.toml`, exotic ones in a task-local one.

**`obs_pred`.** `prediction_summary(obs[cluster_key], obs[pred_col], obs.get(conf_col))`. Establish what produced the
column first; it inherits that method's disqualifiers.

**`reference_map`** (CellMapper):
```python
from cellmapper import CellMapper
cmap = CellMapper(query, ref)                         # both on shared symbols, log1p of CP10k
cmap.compute_fast_cca(n_comps=50, key_added="X_joint")   # compute_joint_pca for same-modality
cmap.compute_neighbors(use_rep="X_joint", knn_method="pynndescent", random_state=0)  # "rapids" on a GPU (cuml)
cmap.compute_mapping_matrix()
cmap.map_obs(label_col, prediction_postfix="_pred", confidence_postfix="_conf")
```
- Read a large reference backed and subsample before loading: `ref.obs.groupby(label_col).sample(...)` capped per
  type, fixed seed, then `ref[idx].to_memory()`.
- Normalize a counts reference (`normalize_total(1e4)`, `log1p`) to the query's space.
- Align gene ids first: an Ensembl reference with `feature_name` gets `var_names = var["feature_name"]`,
  `var_names_make_unique()`. Census slices index `var` by integer position, so never treat those names as symbols.
  Without a symbol column, map the marker-relevant genes via the knowledgebase MCP (`bc_get_ensembl_id_from_gene_symbol`).
  About a quarter of full-transcriptome genes have no symbol; that is expected.
- Refuse below 30 shared genes.

**`scarches`** (scvi-tools; a tissue-matched hub model when one exists, else train scVI then scANVI on the atlas):
```python
model = HubModel.pull_from_huggingface_hub(repo_name=repo, cache_dir=cache).model  # or a trained model's path
setup = model.registry_["setup_args"]
if setup.get("batch_key") and setup["batch_key"] not in query.obs:
    query.obs[setup["batch_key"]] = "query"           # a new batch for surgery to absorb
if setup.get("labels_key"):
    query.obs[setup["labels_key"]] = setup.get("unlabeled_category", "Unknown")
scvi.model.SCANVI.prepare_query_anndata(query, model)
q = scvi.model.SCANVI.load_query_data(query, model)
q.train(max_epochs=100, devices=1, plan_kwargs={"weight_decay": 0.0}, check_val_every_n_epoch=10)
labels, conf = q.predict(), q.predict(soft=True).max(axis=1)
```
- Raw counts in the layer the model registered; a model trained with `gene_likelihood="normal"` wants log-norm instead.
- Seed (`scvi.settings.seed = 0`); `torch.set_float32_matmul_precision("high")` on Tensor-Core GPUs; leave DataLoader
  workers at 0. Pulling hub metadata needs network.

**`azimuth`** (`panhumanpy`):
- Input is raw counts, or exact log1p of CP10k with `normalization_override=True`. Anything else is silently
  mis-scaled, because panhumanpy infers normalization from integer-ness.
- `var_names` must be symbols. With Ensembl ids it zero-fills every gene and returns confident nonsense.
- Refuse below 20% overlap with the model's 5,055-gene panel (`panhumanpy._tools.v1/inference_feature_panel`).
- `AzimuthNN(adata, model_version="v1", normalization_override=..., output_mode="minimal", refine=True).cells_meta`
  gives `azimuth_broad/medium/fine` and `final_level_confidence`. A cell whose heads disagree carries the string
  `"False"`: replace it with `final_level_labels` and count it as inconsistent.
- CL ids come from the shipped `cell_ontology_map.csv`: match the label, then its leaf after the last `|`, and drop
  `CL:0000000`.
- Per cluster, keep the fractions `Unassigned`, `Doublet like cell` and inconsistent as state evidence
  (methodology §6). Compare labels at your hierarchy's granularity: its fine labels are often finer than the
  annotation, which reads as disagreement but is refinement.
- Install: it pins TensorFlow 2.17 and scikit-learn 1.6 exactly, which conflicts with numpy 2. The pins are not
  load-bearing (labels identical on TF 2.21 / numpy 2.4), so in a separate pixi feature use
  `pypi-options.dependency-overrides = { tensorflow = ">=2.17,<3", scikit-learn = ">=1.6" }` and `numpy = ">=2,<2.5"`.

**`celltypist`:**
- List models without downloading with `celltypist.models.get_models_index(force_update=True)["models"]`.
  `get_all_models()` lists only downloaded ones.
- Choose a model that is tissue- **and** stage-matched, or none: a mismatched model collapses onto a few types.
- `models.download_models(model=[f"{name}.pkl"])`, then
  `celltypist.annotate(adata, model=f"{name}.pkl", majority_voting=True, over_clustering=cluster_key)`.
- Input `X` must be log1p of counts normalized to 1e4.
- Summarize the raw `predicted_labels`, not `majority_voting`, which is one label per cluster by construction.
- `Model.load(...).extract_top_markers(type, n)` gives marker priors without predicting.

**`scimilarity`:**
- `CellAnnotation(model_path=".../model_v1.1")`, then `align_dataset(query, ca.gene_order)` with raw counts in
  `layers["counts"]`, then `lognorm_counts`, then `ca.get_embeddings(query.X)`, then
  `ca.get_predictions_knn(emb, weighting=True)`.
- Needs 16-27 GB RAM and a ~9 GB annotation bundle; CPU is fine.
- Install `hnswlib` from conda-forge. The PyPI wheel is built with AVX-512 and crashes (SIGILL) on AVX2-only CPUs.

**`census`:**
- `cellxgene_census.open_soma(census_version=...)`. Probe `soma_joinid` alone first, since a broad filter matches
  millions of cells. Cap the cell count and subsample stratified by `cell_type` (fixed seed), then
  `get_anndata(..., obs_coords=ids)`.
- `X` is raw counts. Record the resolved build, not `stable`, in the task README.

**Tie-break only:**
- SingleR (`singler` + `celldex`): densify the matrix first, because a sparse transpose mis-annotates.
- `sc.tl.ingest` against a reference.

## Skipped, and why

- **Foundation models (scGPT, Geneformer, UCE, scFoundation) as primary annotators.** They are a fallback for when
  no matched model exists, never the default.
- **popV.** A ten-method consensus contradicts the one-cross-check design.
- **External or hosted LLM annotators (GPTCelltype, CyteType).** Claude annotates here, and the data stays on the
  machine. Two of their conventions are adopted anyway:
  - competing hypotheses per cluster;
  - an explicit confidence per call.

## Out-of-distribution signals

Azimuth's diagnostics are a more reliable guide than any axis you set by hand:
- **Inconsistent-hierarchy fraction.** It rises sharply out of distribution.
- **`Unassigned`.** A trained class for low-quality and mixture profiles.
- **Calibrated confidence.** Low confidence also signals a query the model was not trained for.

Gene coverage is not the tell: orthologs keep the panel overlap high on mouse data while the biology goes
unrecognized.
