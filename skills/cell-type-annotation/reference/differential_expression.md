# Reference: Differential expression — two distinct jobs

There are **two** DE jobs and they must never be swapped:

> **Markers (Job A)** = per-cell Wilcoxon (+ COSG), descriptive, panel-robust.
> **Conditions (Job B)** = pseudobulk + PyDESeq2, inferential, replicate-dependent.

Job A is the dominant need for annotation. Job B is secondary (e.g. comparing two
conditions within a cell type) and has a hard replication requirement.

---

## Job A — cluster marker identification (the annotation input)

- **Default call:**
  ```python
  sc.tl.rank_genes_groups(
      adata, groupby=cluster_key,
      method="wilcoxon", tie_correct=True, pts=True,
      use_raw=False,   # and point at the log1p layer explicitly
  )
  ```
  ⚠️ scanpy's **code default is `t-test`, not `wilcoxon`** — set it explicitly.
  `tie_correct=True` matters for sparse / limited-panel (e.g. MERSCOPE) counts.
  Run on **log-normalized** data (not raw, not scaled).
- **Benchmark backing** (Pullin & McCarthy, *Genome Biology* 2024, 59 methods):
  **simple methods win** (Wilcoxon, logreg, t-test, limma). Fancy methods give no
  downstream-annotation advantage — don't reach for them.
- **Add COSG** (`pip install cosg`, `cosg.cosg`) — cosine-similarity selection that
  optimizes **specificity** (a more annotation-relevant objective than the per-gene
  p-value), spatial/panel-aware, fast on CPU. It returns a score, not a p-value →
  use it to rank/select clean markers. Pattern: **Wilcoxon for the statistics, COSG
  for specific panels.**
- **Extract results:**
  `sc.get.rank_genes_groups_df(adata, group=...)` → names / scores / logfoldchanges /
  pvals_adj / pct. `pts=True` adds the fraction-expressing columns.
  - Conventional filter: `pvals_adj < 0.05`, `logfoldchanges > 1` (log2),
    `pct_nz_group > 0.25`, `max_out_group_fraction ≲ 0.5`. Use
    `sc.tl.filter_rank_genes_groups` or filter the dataframe directly.
- **`logreg`** = a multivariate cross-check for subtle / closely-related states (keep
  regularization on).
- **Panel size does NOT break one-vs-rest tests** — they are per-gene and the
  multiple-testing burden is gentler on a small panel. Real hazards on targeted
  spatial data:
  - zero-heavy ties → `tie_correct=True`;
  - a "marker absent" may simply mean **not on the panel** — NEVER infer cell-type
    absence from a panel-absent gene;
  - normalization / segmentation spillover creates false *shared* markers → COSG's
    specificity objective helps cut these.

---

## Job B — condition / contrast DE (secondary)

- **Non-negotiable: pseudobulk, not per-cell.** Per-cell tests on conditions inflate
  p-values via pseudoreplication — cells from one sample are not independent; the
  replication unit is the **sample**. scanpy's own docs warn against
  `rank_genes_groups` for between-condition DE.
- **Default:** pseudobulk + **PyDESeq2** (scverse core since Dec 2025) via
  `decoupler.pp.pseudobulk(adata, sample_col=..., groups_col=..., mode="sum")` per
  sample × cell-type → PyDESeq2 with `design ~ condition`. Pure CPU,
  Apple-Silicon-clean. (⚠️ decoupler 2.x uses `dc.pp.pseudobulk`; the old
  `dc.get_pseudobulk` was removed.)
- **pertpy** DE module for complex designs (covariates / contrasts, formula
  interface, PyDESeq2 backend).
- **Milo** (`pertpy.tl.Milo`) when the phenotype is **compositional** (cell-state
  abundance shifts) rather than within-type expression — common when a perturbation
  shifts cell-state composition.
- **memento** (`memento-de`, scanpy-compatible, CPU) — tests mean **+ variability +
  gene-correlation**; a complement when those matter (younger software).
- **Avoid:** per-cell Wilcoxon for conditions; scVI-DE as the primary on CPU (slow,
  same pseudoreplication caveat); **diffxpy entirely** (archived Oct 2025, redirects
  to PyDESeq2); MAST / edgeR unless you accept an R dependency.
- **Hard requirement:** valid Job-B inference needs ≳3 biological replicates per
  condition. With n=1 per condition (e.g. a single-timepoint design), describe effect
  sizes but state plainly that **no valid test is possible**.
