# Reference: Knowledge sources — databases, ontologies, atlases, literature

Prefer **MCP tools** (zero setup, in-session) for knowledge lookups; drop to a Python
loader only where there is no MCP equivalent. **Insist on proper tooling, two tiers:**
the **`knowledgebase`** MCP is a *hard requirement* — if it is not configured, STOP and
ask the user to set it up rather than silently degrading to web-only evidence. This is
BioContextAI's **Knowledgebase MCP** (`github.com/biocontext-ai/knowledgebase-mcp`;
registry at `biocontext.ai`) — point the user there for setup. The
**`zotero`** MCP is *strongly preferred but optional* — if it is missing, say so and
fall back to the Europe PMC literature tools below, don't stop the run.

**Always normalize final labels to Cell Ontology (CL) IDs** (for fine subtypes where a
transcriptomic taxonomy exists, e.g. PCL — the provisional cell ontology — for
neuronal subtypes).

**Validate every `cl_id` before write-back — semantic check is yours (MCP), not the CLI's.**
`cta report write-back` only checks `cl_id` *syntax* (`CL:0000679` / `PCL:0000123`; it blocks a
malformed id). Whether the term actually **exists**, is **not obsolete**, and its official name
is **consistent with the label** you assigned needs an ontology lookup, which the standalone
`cta` commands can't do. So before the final write-back (SKILL.md Stage 3.6), call
`bc_get_term_details` on each `cl_id` and reconcile any mismatch with the user — this is the same
MCP you used to normalize the label in the first place, closing the loop on it.

## Already covered by `knowledgebase-mcp` (verified working)

- **PanglaoDB** — `bc_get_panglaodb_marker_genes` / `bc_get_panglaodb_options`
  (human + mouse; organ / cell_type / specificity filters). Coverage is broad but
  shallow — finely-resolved subtypes are thin. ⚠️ The database content is
  **frozen (~2020)** → use as a coarse prior only, never as ground truth.
- **Cell Ontology (CL) via OLS4** — `bc_get_cell_ontology_terms`,
  `bc_get_term_hierarchical_children`, `bc_search_ontology_terms`,
  `bc_get_term_details`. Search + hierarchy traversal across diverse lineages. This
  is your canonical **label → CL normalizer**. Try `ontologies="pcl"` (the
  provisional cell ontology) for fine neuronal subtypes.
- **Human Protein Atlas (HPA)** — `bc_get_human_protein_atlas_info`, per-gene (tissue
  specificity, subcellular localization). Gene → info only; the reverse
  (cell-type → genes) needs a bulk download.
- **Europe PMC** — `bc_get_europepmc_articles` / `bc_get_europepmc_fulltext`
  (abstracts + OA full text + preprints). Your primary literature tool. **Route
  bioRxiv topic search here** — the dedicated bioRxiv MCP is recent/date-range only
  (no keyword search).
- Plus gene-symbol ↔ Ensembl, GO, KEGG, Reactome, STRING, UniProt, Open Targets —
  for marker validation and pathway context.

## Zotero MCP — check FIRST (if configured)

When the `zotero` MCP is available, use `zotero_semantic_search` before anything else
for marker evidence: the user's curated library very likely already contains the key
papers for their tissue. This is the highest-trust, lowest-cost source. Cite what you
find in the evidence cards. If `zotero` is not configured, skip this step and rely on
the `knowledgebase` Europe PMC tools for literature — don't block on it.

## Need a small Python loader (no MCP equivalent)

- **CellTypist model markers** (top pick for data-driven markers when a model covers
  your tissue): tissue-gate it with `cta crosscheck celltypist list [--contains <tissue>]`
  (reads the remote index, no downloads; ~60 models, e.g. `Developing_Human_Brain`,
  `Cells_Intestinal_Tract`, `Immune_All_Low`), and if one matches use `extract_top_markers()`
  for priors. Full tissue-gating mechanics + caveats: `reference/automated_tools.md`.
- **CellMarker 2.0** — broadest curated, PMID-backed (CC-BY-4.0 xlsx; the host is
  flaky → cache the file locally once).
- **ScTypeDB** (`ScTypeDB_full.xlsx`) — a rare source of **negative markers**.
- **CellxGene Census** (`cellxgene-census`, optional extra) — a remote TileDB-SOMA
  connection over S3: opening it does **not** download the atlas; you fetch only the
  capped, tissue/organism-filtered slice your query selects (laptop-scale, CPU). Here it
  serves as a **marker source**: derive markers on demand for "cell type X in tissue Y"
  via `get_anndata` + `rank_genes_groups`. (Its other use — `cta reference fetch-census` as
  the no-atlas reference-mapping fallback, with the author-supplied-label caveat — lives in
  `reference/automated_tools.md`.)

## Reference atlases (compute markers off them — AND map onto them as a cross-check)

Whenever the user has a curated reference atlas matching their tissue/context, it beats
generic frozen databases. Use it two ways: (1) compute markers directly with
`rank_genes_groups` for priors (`cta markers assemble --reference ...`); and (2) **map the
query onto it** as an automated cross-check with `cta crosscheck reference-map` (CellMapper —
builds its own joint embedding, transfers labels with confidence; no pre-existing joint
rep needed). Capture reference path(s) + label column(s) at Stage 1. For spatial panels a
platform-matched spatial reference is especially useful.

**No atlas of your own?** When the user has no reference and no CellTypist model fits the
tissue, fall back to `cta reference fetch-census` to pull a capped, tissue-matched
subset from CELLxGENE Census and map onto *that* with `cta crosscheck reference-map` — so every
tissue still gets an automated reference cross-check (caveats: `reference/automated_tools.md`).

**On a GPU / large-memory profile** (`reference/compute_environments.md`) two reference
upgrades open up: lift the `cta reference fetch-census --max-cells` cap (or use a full
curated atlas, since the RAM is there to hold it), and prefer a scANVI reference model
mapped on via scArches surgery — a shipped scvi-tools HuggingFace-hub model
(`cta reference map-scarches --hf-repo …`), or one trained on the user's atlas with
`cta reference train-model`. Mechanics and the hypothesis-not-verdict bar:
`reference/automated_tools.md`.

## Default knowledge stack (the order to consult)

1. **Ontology-normalize** candidate labels via the CL / OLS MCP.
2. **Quick coarse markers** via the PanglaoDB MCP.
3. **Per-gene validation** via the HPA MCP.
4. **Literature** via Zotero (first, if configured), then Europe PMC.
5. **Data-driven markers** via tissue-matched CellTypist models (tissue-gated; skip if none fit).
6. **Curated breadth** via CellMarker / ScTypeDB.
7. **Ground truth** via the user's own atlases — markers computed directly, and as a
   label-transfer cross-check via `cta crosscheck reference-map` (CellMapper).
