# Methodology

The reasoning behind the per-level loop in `SKILL.md`: the gates, the evidence cards, the sub-cluster decision,
the two label axes, the critics and the panel rules. Read it once before the first level.

## 1. Stance

You are the annotator. Tools produce evidence; you weigh it. Every call must be legible and contestable: a human or
a critic can open the task directory and see why a cluster got its label, what was considered and what was
rejected.

Annotation runs top-down through a label hierarchy and recursively within parent groups, because identity resolves
coarse to fine: a memory CD8 T cell cannot be called before the T cell. Descend into a group only once its parent
label is assigned. Each level strictly refines the one above (`build_hierarchy` refuses anything else); if a
regional split cuts across a parent (ventral telencephalic NPC vs neuron), qualify the label by its parent rather
than reuse one label under two parents.

## 2. The two gates

### Gate A: expose assumptions for rebuke

`reports/assumptions.md` is a living ledger that turns silent assumptions, the usual source of confidently wrong
annotation, into reviewed decisions. One block per level, appended as you descend, in plain language: the reader
approves what the labels will mean, not a config. Each block has four parts:

```markdown
## L1: compartment (signed off 2026-06-15 by the user)

### Markers I'll rely on
| Cell type | Markers           | Why / source                 | What this means for your results                       |
| --------- | ----------------- | ---------------------------- | ------------------------------------------------------ |
| T cell    | CD3D, CD3E, TRAC  | canonical pan-T (PanglaoDB)  | Anything CD3+ is grouped as T; NK splits at L2.        |
| B cell    | CD79A, MS4A1      | canonical pan-B              | Plasma cells fold in here, separated at L2.            |

### Priors I'm importing
- Healthy adult PBMC: the usual lymphoid/myeloid split, no stromal compartments.

### Panel caveats (targeted panels only: absence is not negative evidence)
| Identity        | Canonical markers | On panel? | Consequence                                   |
| --------------- | ----------------- | --------- | --------------------------------------------- |
| Plasmacytoid DC | CLEC4C, IL3RA     | both off  | Uncallable on this panel: contributes to Unknown. |

### Label vocabulary
L1 = compartment, L2 = lineage, L3 = type, L4 = state; labels normalized to CL ids.
```

The marker table mirrors `results/markers.csv`, and the ledger mirrors what you render in chat, row for row.

- The **first** sign-off blocks. Before any label exists, render the ledger and ask with AskUserQuestion: Approve,
  Revise (which mapping), or Add a constraint ("treat marker X as ambiguous"). Say in one line why the gate exists:
  every label at this level rests on these assumptions. Never infer sign-off from silence. Record it in the block
  heading.
- At later levels, append the new assumptions and re-prompt only for a **materially new** one (a new marker→type
  mapping, prior or panel caveat that changes a call). Otherwise tell the user what you appended and continue.
- Invariant: never assign a label on an assumption the user has not had the chance to see.

**Non-interactive runs** (benchmarks) are opt-in, never the default. They apply only when the user has asked for an
unattended run **and** supplied the context each gate would elicit: keys, biology, vocabulary, marker→type
assumptions (a filled `plan.md`/`assumptions.md`, or an unambiguous brief). Then:
- auto-approve and record: write every ledger block and report as usual, with "signed off: pre-authorized,
  non-interactive" in the heading;
- gaps still stop: a materially new assumption without a basis, a level over half `Unknown`, an irreconcilable
  critic conflict;
- say in the summary which gates were auto-approved against which supplied context.

### Gate B: honest `Unknown`

When the evidence is irreconcilable, the label is `Unknown`, documented and flagged, not a forced guess. A justified
`Unknown` is a success: it shows where the data or the panel cannot support a call. Below the top level, qualify it by
its parent (`Unknown T cell`, `Unknown B cell`) so the hierarchy still nests; keep the `Unknown` prefix, which is what
the reports and the figure recognize.

## 3. Termination

- Level recursion ends at the last level of the agreed hierarchy.
- Sub-clustering is bounded: at most **one split** per cluster (§5). If it does not resolve the conflict, the
  cluster is `Unknown`.
- A label refuted by the critics with no surviving alternative becomes `Unknown`; `refuted` is a transient status.
- If a step errors, surface it; never skip a cluster silently.
- If a level lands more than about half its clusters in `Unknown`, stop and consult the user: that pattern usually
  means a wrong panel assumption, a bad embedding or a normalization problem upstream.

## 4. Where the reasoning lives

`results/labels_<level>.csv` is the single source of truth, one row per cluster:
`cluster, label, state, cl_id, confidence, status, rationale, rejected` (`LABEL_COLUMNS`). Filling it *is* the
assignment. `status` is lowercase, one of `assigned`, `unknown`, `needs_subcluster`, `refuted`; cluster ids stay
verbatim. `render_level` turns the table plus the evidence tables into `reports/<level>.md`, one card per cluster,
so re-rendering never loses reasoning, and `check_level(labels, clusters)` reports what still keeps the level from
being complete, including a cluster with no row or with two.

A card shows:
- cluster, size, parent;
- the specific markers (`filter_rank_genes_groups` survivors, with log2FC and fraction expressing) and the curated
  panel's detection, including off-panel markers per candidate;
- negative markers expressed where they should be absent;
- the cross-check's call and whether it agrees;
- the label, normalized to a CL id, and the state;
- a **confidence** anchored to objective signals: the top-1 vs top-2 margin, the share of the winner's markers on the
  panel, cross-check agreement. Calibrate it against `confidence_proxy`; a high confidence on a thin margin is a
  flag, since a label can be wrong and feel certain;
- the rationale, and what was rejected and why.

## 5. The sub-cluster decision

A cluster co-expressing markers of mutually exclusive types is a real mixed or transitional state, a doublet, or
under-clustered. Before concluding, sub-cluster it on the agreed embedding and re-score the competing positive and
negative sets per sub-cluster. `restrict_to` reuses an existing graph, so build that graph from the embedding first:
`sc.pp.neighbors(adata, use_rep=embedding_key, key_added="annot")`, then
`sc.tl.leiden(adata, restrict_to=(key, [cluster]), neighbors_key="annot", key_added=...)`. A purity measure (e.g. ROGUE) can corroborate a split. Split at most once, and only into sub-clusters
large enough to test.

**Doublets.** Doublet *detection* (Scrublet, scDblFinder) stays out of scope. Here a doublet means a cluster mixing
two or more types, and its handling is sub-clustering: a true doublet usually splits into its single types. An
existing doublet score in `obs` corroborates. If a split does not separate the identities and a doublet cannot be
ruled out, do not promote the cluster to a novel type: set `state=doublet`, `label=Unknown`, or ask the user for a
doublet score.

## 6. Two axes: type and state

A cluster has a **type** (lineage identity) and a **state** (quality, cycle, artifact). Forcing both into one label
causes the commonest failures: a cycling cluster mislabelled because cell-cycle genes dominate its markers, or a
low-quality cluster given a type it cannot support. `state` is a separate column, exempt from CL ids and nesting.
Vocabulary: `normal`, `low-quality`, `cycling`, `doublet`, extended as the tissue needs (`stressed`, `activated`).

- **Cycling:** a proliferating cluster almost always has a real lineage. Keep the type and set `state=cycling`.
- **Low-quality:** low depth or genes, high mitochondrial fraction, off-population in `qc_deviation`. When that makes
  the type unsupportable: `state=low-quality`, `label=Unknown`.
- **Doublet:** sub-cluster first (§5); `state=doublet` with `label=Unknown` only when a split fails.

State evidence comes from the QC metrics and cell-cycle scores computed in Stage 1, per cluster. When Azimuth ran
(human adult), its trained classes are a second, independent source:

| per-cluster fraction | corroborates | note |
| --- | --- | --- |
| `Unassigned` | low-quality **or** a mixture | a trained class (deliberate negatives: empty droplets, debris, simulated mixtures); read it against QC deviation vs co-expression |
| `Doublet like cell` | the sub-cluster-first route | a second opinion without bringing detection into scope |
| inconsistent hierarchy | Gate B; out-of-distribution | the model's heads disagree on a path; high means do not let it drive a fine call |
| mean confidence | your `confidence` | temperature-scaled, so an actual probability, unlike other methods' |

These corroborate a state; they never set one. A cluster is low-quality because its QC says so and you judged the
type unsupportable, not because a classifier said `Unassigned`.

**When `Unknown`:** no candidate above noise; two identities equally supported after one split; the only fitting
candidate is uncallable (fewer than two on-panel markers, `panel_gate`); or a co-expressing cluster that cannot be
told from a doublet.

**Investigate first.** A large or odd cluster that resists a clean call is often a real early or transitional
state. Look at its QC medians, its composition over the covariates the user named in Stage 1 (e.g. a timepoint),
and, for spatial data, where it sits. Good-quality cells concentrated in one stage with coherent, if non-canonical,
markers are a type to name.

## 7. Critics

After a level's labels are assigned, spawn independent critic subagents whose job is to **refute**: alternative
types that fit the same markers, ignored negative markers, panel artifacts, doublet explanations.

The critic and the annotator are the same model, and a card is the annotator's own argument. To make the challenge
real:
- the critic reads the raw tables in `results/` (markers, sibling scores, panel detection, negative flags, panel
  gate) and re-derives the top markers itself, treating the card as the claim under test;
- it fetches at least one competing hypothesis from an external source (the knowledgebase MCP, plus Zotero), and
  checks markers there: STRING (`bc_get_string_interactions`), GO (`bc_get_go_terms_by_gene`), HPA
  (`bc_get_human_protein_atlas_info`), Open Targets;
- it has no access to the in-memory AnnData or the chat.

**Hierarchy.** Structural nesting is enforced in code (`build_hierarchy`). The critic checks meaning: a maturation
mismatch (an "IPC" or "neuron" under an "NPC" parent), a regional contradiction (a "thalamic" L3 under a
"telencephalic" L2), a finer label that is not a subtype of its parent. A node `display_markers` flags with `−`
while confidently called means its signed-off markers do not separate it from its siblings: the critic must explain
it or challenge the call.

Reconcile before finalizing: the label survives, or is revised or downgraded to `Unknown`. A conflict you cannot
resolve from the evidence goes to the user before the level is finalized.

**Scaling.** Batch related clusters per critic (those competing for one parent, or 5-10 at a time) so the critic can
contrast neighbours. Critic every flagged cluster individually (`Unknown`, low confidence, thin margin, uncallable
winner, hierarchy mismatch) and batch the clean ones more coarsely. If you sample the confident tier, say so in the
report. A substantive challenge from a batched critic still gets an individual follow-up.

## 8. Panels and spatial data

- An off-panel gene is unmeasured, never negative evidence. Gate every marker set against the panel.
- Segmentation spillover creates false shared markers between adjacent types: lean on specificity (the
  `filter_rank_genes_groups` out-group filter, sibling margins) and spatial coherence.
- Spatial context (neighbourhood, niche) corroborates marker evidence; it does not replace it.
- Detect and alert: surface signs of unaddressed normalization, batch or ambient problems; never fix them silently.
