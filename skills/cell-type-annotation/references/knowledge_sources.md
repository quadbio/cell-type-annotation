# Knowledge sources

Prefer MCP tools; use a Python loader only where no MCP covers the source.

## Requirements

- **`knowledgebase` MCP** (BioContextAI, `github.com/biocontext-ai/knowledgebase-mcp`): required. If it is not
  configured, stop and ask the user to set it up; do not degrade to web-only evidence, which is the
  confident-but-unfounded annotation this skill exists to prevent.
- **`zotero` MCP**: strongly preferred, optional. If missing, say so and use the knowledgebase's Europe PMC tools.
- Check that whichever are present actually respond before starting.

## Cell Ontology

Normalize every final label to a Cell Ontology id (CL), or PCL, the provisional cell ontology, for fine neuronal
subtypes. Before write-back, call `bc_get_term_details` on every `cl_id`: the term exists, is not obsolete, and its
name fits the label. Reconcile mismatches with the user. `validate_cl_ids` checks syntax only.

## What the knowledgebase MCP covers

- **PanglaoDB** (`bc_get_panglaodb_marker_genes`, `bc_get_panglaodb_options`): human and mouse, broad but shallow,
  frozen around 2020. A coarse prior, never ground truth.
- **Cell Ontology via OLS4** (`bc_get_cell_ontology_terms`, `bc_search_ontology_terms`,
  `bc_get_term_hierarchical_children`, `bc_get_term_details`): the label → CL normalizer; `ontologies="pcl"` for fine
  neuronal subtypes.
- **Human Protein Atlas** (`bc_get_human_protein_atlas_info`): per gene, tissue specificity and localization.
- **Europe PMC** (`bc_get_europepmc_articles`, `bc_get_europepmc_fulltext`): abstracts, open-access full text and
  preprints. Route bioRxiv topic searches here; the bioRxiv tools search by date only.
- Symbol ↔ Ensembl (`bc_get_ensembl_id_from_gene_symbol`), GO, KEGG, Reactome, STRING, UniProt, Open Targets: marker
  validation and pathway context.

## Zotero first

When `zotero` is configured, search it (`zotero_semantic_search`) before anything else: the user's library likely
holds the key papers for their tissue, it is the highest-trust source, and its full texts are searchable. Cite what
you use in `results/markers.csv`.

## Loaders without an MCP

- **CellTypist model markers**, when a tissue- and stage-matched model exists: `extract_top_markers()` gives priors
  without predicting (`method_selection.md`).
- **CellMarker 2.0**: the broadest curated, PMID-backed table (CC-BY-4.0 xlsx; the host is flaky, so cache it once).
- **ScTypeDB** (`ScTypeDB_full.xlsx`): a rare source of negative markers.
- **CELLxGENE Census**: markers for "type X in tissue Y" from a capped slice with `rank_genes_groups`.

## Reference atlases

A curated atlas matching the tissue and context beats any frozen database. Use it twice: compute markers off it
(`rank_genes_groups` on its labels) as priors, and map the query onto it as the cross-check. Record its path, label
column(s), whether its `X` is counts, and its gene-id convention at Stage 1. For spatial panels, a platform-matched
spatial reference is especially useful. Without one, a Census slice keeps a reference cross-check alive, with a
caveat: Census labels are author-supplied and mixed in quality.

## Order to consult

1. Normalize candidate labels via the Cell Ontology tools.
2. Coarse markers from PanglaoDB.
3. Per-gene checks in HPA.
4. Literature: Zotero, then Europe PMC.
5. Data-driven markers from a matched CellTypist model, if one fits.
6. Breadth from CellMarker, negatives from ScTypeDB.
7. The user's own atlas: markers, and the mapping cross-check.
