# Reference: Marker-based scoring & manual annotation tools

CPU / Apple-Silicon, scVerse-first. These are the tools that turn marker gene sets
into per-cluster or per-cell scores, plus the scanpy idioms for manual annotation.
Every score is *evidence for a hypothesis*, never a verdict — Claude's reasoning
over the evidence is the arbiter.

## Default marker-annotation stack (CPU)

1. **scanpy** — the substrate. Clustering (`sc.tl.leiden(flavor="igraph")`,
   multi-resolution), `sc.tl.rank_genes_groups`, and visualization
   (`dotplot`/`matrixplot` with `standard_scale="var"` and ~p99 value caps).
2. **decoupler (scverse core, 2.x)** — `dc.mt.ora` **per-cluster** on DE results for
   marker-overlap calls; `dc.mt.aucell` **per-cell** for robust continuous scores.
3. **snapseed** (`pip install hnoca`; the standalone repo is archived) — declarative
   YAML marker hierarchy for multi-level annotation; mean/fraction scoring (JAX),
   less robust → use as a first-pass and verify ambiguous nodes by hand.
4. **pyUCell** (scverse ecosystem, 2026) — robust rank-based per-cell scoring with
   **positive + negative** gene sets; the principled upgrade over
   `sc.tl.score_genes`.

## Critical API corrections (these will bite if you use old tutorials)

- **decoupler moved to scverse core with a 2.0 API rewrite.** Use `dc.mt.*` /
  `dc.pp.*`, **not** the old `dc.run_*`. The relevant methods:
  - **`ora`** — over-representation (Fisher/hypergeometric) on the top genes of a
    cluster; the best choice for plain per-cluster marker overlap.
  - **`aucell`** — rank-AUC per cell; the best choice for robust continuous scores.
    ⚠️ **On targeted/spatial panels, compare AUCell scores RELATIVELY, never absolutely
    across signatures.** A broadly-expressed marker set scores high in *every* cluster
    and can argmax the same broad type everywhere (incl. obvious neurons). Read the
    per-signature **z-score across clusters** (`score_signatures.py` writes
    `scores_aucell_z`) and the consolidated `crosscheck` table, and let **COSG
    specificity** + the reference cross-check be primary — not raw AUCell.
  - `ulm` / `mlm` / `viper` are for **weighted TF/pathway regulons, NOT bare marker
    lists** — do not use them to score curated cell-type markers.
- **`sc.tl.score_genes` is composition-dependent and bin-sensitive.** Its control
  bins are computed over the whole dataset, so the same cell scores differently
  depending on what else is loaded. Prefer pyUCell (rank-based) or `dc.mt.aucell`
  for anything you will annotate on or compare across runs.
- **snapseed scores by mean-expression / fraction, NOT AUCell** (a common
  misconception). Fast but uncalibrated — no probability output. Treat its labels
  as preliminary.

## Other tools — borrow ideas, mostly don't depend

- **COSG** (`pip install cosg`, `cosg.cosg`) — cosine-similarity marker selection
  optimizing **specificity**; spatial/panel-aware, fast on CPU, gives a score (not a
  p-value). Excellent for picking clean markers on targeted panels. (Also covered in
  `differential_expression.md`.)
- **ScType / sc-type-py** — specificity-weighted z-score with **negative markers**
  and per-cluster argmax. Good conceptual model; the R original is dormant.
- **MACA** — dormant.
- **ANS** — fixes `score_genes` bias; niche.
- **HiCAT** — 3-level hierarchical, GSA-scored.
- **CellO** — ontology classifier, but needs trained models + whole-transcriptome →
  skip for targeted panels.

## Manual annotation idioms (scanpy-native)

- **Dotplot of a marker dictionary:**
  `sc.pl.dotplot(adata, markers_dict, groupby=cluster_key, standard_scale="var")`
  — dot size = fraction expressing, color = mean expression. A `{type: [genes]}`
  dict draws labeled blocks. Cap extremes with `vmax`/p99.
- **Negative markers:** scanpy has no native support → use pyUCell (pos/neg sets) or
  a signed decoupler network.
- **Cluster → label mapping:**
  `adata.obs["ct"] = adata.obs[cluster_key].map(mapping_dict).astype("category")`.
- **Sub-clustering in place:**
  `sc.tl.leiden(adata, restrict_to=(cluster_key, ["3"]), key_added=...)` splits one
  cluster. Resolve mutually-exclusive marker co-expression by sub-clustering, then
  scoring the competing pos/neg sets (aucell / pyUCell) and taking the argmax — and
  check for doublets before concluding it is a real mixed state.

## Most underrated

- **pyUCell** — robust rank scorer with negative sets; the right default for
  per-cell signature scores.
- **decoupler `mt.ora`** — overlooked for plain cluster→type calls because decoupler
  is famous for TF activity; it is exactly the right tool for marker-overlap.
- **snapseed's YAML hierarchy** — adopt the *format* as a version-controllable
  marker source-of-truth even if you don't run the engine.
