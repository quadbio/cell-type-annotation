# Marker tools

Recipes for the per-level evidence, called directly from task scripts. Every score is evidence for a hypothesis,
never a verdict.

## Between-cluster DE

Marker identification compares clusters, per cell, descriptively. It is not condition DE (comparing conditions
within a type needs pseudobulk and replicates), which is out of scope here.

```python
adata.var["de_genes"] = sc.pp.filter_genes(adata, min_cells=5, inplace=False)[0]  # a mask: the panel stays whole
sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon", pts=True, use_raw=False, mask_var="de_genes")
sc.tl.filter_rank_genes_groups(adata)    # min_in_group_fraction=0.25, max_out_group_fraction=0.5, min_fold_change=1
specific = sc.get.rank_genes_groups_df(adata, group=None, key="rank_genes_groups_filtered").dropna(subset=["names"])
```
- Set `method="wilcoxon"`: scanpy's default is a t-test. Run on log-normalized data, never raw or scaled.
- Keep the gene floor absolute. A relative one (0.1% of 500k cells = 500) deletes the markers of a 27-cell cluster.
  Mask rather than subset: a measured but rare gene removed from the object would read as off-panel downstream.
- Simple methods win for markers (Wilcoxon, t-test, logreg; Pullin & McCarthy, *Genome Biology* 2024). Use
  `method="logreg"` as a multivariate cross-check for subtle states.
- **Within the parent.** At L2 and below, rank a parent group's clusters against each other (subset to the parent),
  not against the whole object.
- **Pairwise.** To separate two candidate siblings, subset to those two clusters and rank there
  (`groups=[a], reference=b`): `filter_rank_genes_groups` recomputes its fractions against every other cell in the
  object, not against the reference.
- **Sibling margins.** `hierarchy_markers(adata, [parent_key, cluster_key], genes=panel)` scores each cluster's
  detection against its siblings within the parent, a fraction-of-cells statistic independent of cluster size.
  `FOXG1` is a poor telencephalic marker against everything, but an excellent one at the telencephalic branch point.
  Write it to `results/sibling_markers_<level>.csv`.
- On sparse targeted panels, consider `tie_correct=True`. Spillover creates false shared markers, which the
  out-group fraction filter helps cut.
- At atlas scale on a GPU, `rapids_singlecell.tl.rank_genes_groups` (same Wilcoxon, same `uns` output) is a drop-in.
  Its CUDA-13 kernels need `cuda-cudart = "13.*"` and `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` in a CUDA-12 pixi env, or it
  silently falls back to CPU.

## The curated panel

`results/markers.csv` holds the curated markers: `level, node, gene, sign, source, pmid, note`, one row per gene.
Read it with `read_markers`.
- `pos`: genes that define the type.
- `neg`: genes whose presence argues against it (`DCX` for an NPC, `MS4A1` for a T cell). You curate negatives;
  sources yield positives only.

Per level:
```python
gate = panel_gate(markers, adata.var_names)        # off-panel genes, and uncallable candidates (< 2 measured)
genes = sorted(set(markers.gene) & set(adata.var_names))
agg = sc.get.aggregate(adata[:, genes], by=cluster_key, func=["count_nonzero", "mean"])
pct = pd.DataFrame(agg.layers["count_nonzero"], index=agg.obs_names, columns=genes).div(
    agg.obs["n_obs_aggregated"].to_numpy(), axis=0
)
neg = negative_marker_flags(pct, markers)          # violation: a negative detected in > 20% of the cluster
```
An off-panel negative is unmeasured, never confirmed absent. Melt `pct` to `cluster, gene, pct`, sorted, for the
cards.

## Scores

- **decoupler 2.x:** `dc.mt.ora` per cluster on the top DE genes for plain marker overlap; `dc.mt.aucell` per cell.
  The old `dc.run_*` API is gone. `ulm`, `mlm` and `viper` are for weighted regulons, not marker lists.
- **AUCell on panels is relative.** A broadly expressed set scores high everywhere and can win every cluster. Read
  each signature's z-score across clusters, and let filtered markers and the cross-check lead.
- **pyUCell:** rank-based, with positive and negative sets (`gene-` suffix). The principled replacement for
  `sc.tl.score_genes`, which depends on what else is in the object.
- **snapseed** scores by mean and fraction, uncalibrated: use it as a first pass at most. Its YAML hierarchy is a
  good format for a marker source of truth.
- A consensus table per cluster (top type by overlap, AUCell z, UCell, and how many agree) feeds
  `confidence_proxy(..., agree=...)`.

## QC and cell cycle

When `obs` lacks them, compute in memory, never into the user's object, with symbol `var_names` (SKILL step 5;
on Ensembl ids both lookups silently find nothing):
```python
adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, layer=counts_layer, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
s, g2m = cell_cycle_genes(adata.var_names)         # Tirosh lists, case-insensitive, so mouse resolves too
if len(s) >= 2 and len(g2m) >= 2:                  # on the log-normalized X; .raw is often counts
    sc.tl.score_genes_cell_cycle(adata, s_genes=s, g2m_genes=g2m, use_raw=False)
```
On a targeted panel the cell-cycle genes may be off-panel. Say so: the absence of a score is not "not cycling".
`qc_deviation(obs.groupby(cluster_key)[metrics].median())` flags clusters off the population, from three clusters
on.

## Figures

Intermediate figures are for reading, not for the paper: plain scanpy into `figures/`, e.g.
`sc.pl.dotplot(adata, panel, groupby=cluster_key, standard_scale="var", use_raw=False)` (scanpy otherwise plots
`.raw`, often counts or still Ensembl-indexed), `sc.pl.umap(adata, color=...)`. Reuse `uns["<key>_colors"]` where
present.

The one paper figure of a run is the annotation hierarchy, made at the end with the sci-figures skill:
```python
shown, flags = display_markers(scores, hierarchy, markers, adata.var_names, level_keys=level_keys)
layout = pl.annotation_hierarchy(adata, hierarchy=hierarchy, markers=shown, flags=flags, badge_level=leaf_key)
```
- `scores` is `hierarchy_markers(..., include_genes=<measured positive markers>)`.
- A node shows up to two of its own signed-off canonical markers. Each must clear a 10-point margin over its pooled
  siblings, with no single sibling detecting it more widely.
- A node with none gets a flag: `†` canonical markers off-panel, `−` none separates it, `∅` unassigned.
- Write a `−` on a confident call into the critics' brief.

## Borrow ideas, don't depend

ScType (negative markers, specificity weighting), COSG (cosine specificity), HiCAT (hierarchical scoring) and CellO
(ontology classifier; whole-transcriptome only).
