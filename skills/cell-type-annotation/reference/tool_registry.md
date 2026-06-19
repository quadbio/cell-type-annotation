# Reference: Tool registry — which tool for which step, and what it needs

**The single source of truth for tool selection.** One row per tool, in workflow order:
what step it serves, whether it needs a GPU, which compute profile it belongs to, and the
method behind it. When you add, remove, or re-scope a script, **update this table** — other
docs point here rather than restate it (SKILL.md's Scripts table stays a terse index;
`compute_environments.md` owns the profiles themselves). Profiles are defined in
`reference/compute_environments.md`; the hypothesis-not-verdict bar in
`reference/automated_tools.md`.

### GPU column legend

| Value | Meaning |
| --- | --- |
| **none** | CPU only — no GPU involved. |
| **optional** | Runs on CPU; uses a GPU automatically if one is present (faster, same result kind). |
| **preferred** | GPU strongly recommended. Runs on CPU / Apple-Silicon MPS too, but slowly. |
| **required** | Needs an NVIDIA CUDA GPU; will not run otherwise (Linux + CUDA). |

A separate **RAM** note flags steps whose real cost is memory, not GPU.

## Registry

| Step (stage) | Tool / script | GPU | Profile | Method / notes |
| --- | --- | --- | --- | --- |
| Detect environment (1b) | `detect_env.py` | none | all | stdlib probe of `torch`/`nvidia-smi`/`psutil`/scheduler. Detection only. |
| Bootstrap venv (1c) | `setup_env.sh` | none | all | uv venv; `--profile gpu` also installs `requirements-gpu.txt`. |
| Inspect object (1e) | `inspect_adata.py` | none | all | keys/layers/norm-fingerprint/panel flag → `context.md`. |
| Slim cache (1e, opt.) | `prep_slim.py` | none | all | cache embedding+X+counts+obs to avoid reloading huge files. |
| Seed markers (3.1, opt.) | `assemble_markers.py` | none | all | seed `markers_<level>.json` from a CellTypist model / reference DE. |
| Score — bottom-up (3.3) | `compute_markers.py` | none | all | `rank_genes_groups` (wilcoxon) + COSG (CPU). GPU acceleration would need rapids-singlecell (full RAPIDS+dask, not in the default extras) — niche, since the main object isn't re-preprocessed. |
| Score — top-down (3.3) | `profile_markers.py` | none | all | score the curated panel across clusters → pct/mean + dotplot. |
| Score — signatures (3.3) | `score_signatures.py` | none | all | decoupler `aucell` + pyUCell + DE-overlap → `crosscheck`. |
| Cross-check — in-object (3.4) | `crosscheck_obs_pred.py` | none | all | summarize an existing `obs` prediction column. Preferred when one fits. |
| Cross-check — kNN refmap (3.4) | `reference_map.py` | optional | all | CellMapper kNN transfer. CPU `pynndescent`/`sklearn`/`faiss-cpu`; GPU `--knn-method rapids` (needs cuml-cu12) scales to millions. (`faiss-gpu` needs a conda build; pip wheel is ABI-broken.) |
| Cross-check — CellTypist (3.4) | `run_celltypist.py` | none | all | tissue-gated pretrained LR classifier; `list` (no download) then `predict` only if a model fits. |
| Cross-check — reference model (3.4) | `map_query_scarches.py` | preferred | local_gpu / scheduler | scArches surgery onto a scANVI model (trained, or a scvi-hub shipped model). Labels + per-cell uncertainty. |
| Train reference model (3.4 prep) | `train_reference_model.py` | preferred | local_gpu / scheduler | scVI→scANVI on a labeled atlas (scvi-tools). Reference side of train-then-map. |
| Cross-check — foundation model (3.4, opt.) | `run_scimilarity.py` | none · **RAM ~64 GB** | large-memory | SCimilarity zero-shot kNN. Additional hypothesis only, never primary. |
| Fetch reference (3.4 fallback) | `fetch_census_reference.py` | none | all (larger slices on large-mem) | capped, tissue-matched CELLxGENE Census slice → feeds `reference_map.py`. Needs network. |
| Investigate before Unknown (3.7) | `profile_clusters.py` | none | all | QC medians + categorical crosstabs + optional spatial. |
| Sub-cluster (3.7) | `subcluster.py` | none | all | Leiden on the anchor embedding + per-sub summary. |
| Confidence calibration (3.7) | `cluster_confidence.py` | none | all | COSG-margin + agreement proxy (advisory). |
| Evidence cards (3.6) | `scaffold_cards.py` | none | all | render cards from the labels CSV; `--evidence` embeds the numbers. |
| Report + gate (3.9) | `build_report.py` | none | all | per-level HTML; `--strict` = level-completeness gate. |
| Write-back (final) | `write_back.py` | none | all | write labels into `obs` (gated; `--dry-run` first). |

## How to read it

- **Most of the pipeline is `none`** — the skill's core (inspect → score → reason →
  report) is CPU work and unchanged across environments. GPU only enters at the
  reference-mapping cross-check and (optionally) large-query scoring.
- **`preferred`/`required` rows are the only ones that care about the profile.** On
  `local_cpu` they are skipped or run slowly (CPU/MPS); on `local_gpu` they run inline;
  on `scheduler` they are written into the plan to run via the user's own job-submission
  mechanism (never submitted by this skill — see `compute_environments.md`).
- **`run_scimilarity.py` needs RAM, not a GPU** — it sits on the large-memory axis, which
  is independent of the GPU axis.
