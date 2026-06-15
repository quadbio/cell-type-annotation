# Reference: Annotation methodology

This is the *how* — the algorithm Stage 3 (Execute) runs. See `marker_tools.md`,
`differential_expression.md`, `automated_tools.md`, and `knowledge_sources.md` for
the tool details; this file is the procedure that orchestrates them.

## Table of contents

1. Core stance
2. The two hard gates
3. Top-down recursive algorithm
4. Evidence cards
5. The sub-cluster decision
6. Honest "Unknown"
7. Critic subagents
8. Panel-aware rules (spatial / targeted)

---

## 1. Core stance

You (Claude) are the annotator. Tools produce *evidence*; you weigh it. The whole
point of this skill over a black-box classifier is that every call is **legible and
contestable**: a human (or a critic subagent) can open the working directory and see
exactly why a cluster got its label, what was considered, and what was rejected.

Annotation proceeds **top-down through a label hierarchy** and **recursively within
parent groups**, because identity resolves coarse-to-fine: you cannot sensibly call a
fine subtype (a memory CD8 T cell, a dopaminergic neuron) before you have established
the parent class (a T cell, a neuron).

## 2. The two hard gates

These are load-bearing. Do not skip them to "save time" — they are what makes the
output trustworthy.

### Gate A — Expose assumptions for rebuke (a living ledger)

The assumptions ledger (`assumptions.md`) is a **living document**, not a one-time
form. It converts silent assumptions — the usual source of confidently-wrong
annotation — into reviewed, recorded decisions. For each level it states, explicitly:

- the marker → type mappings you intend to use ("for type X I will rely on markers
  A, B, C");
- biological priors you are importing ("for this perturbation I expect a shift toward
  cell type / state X");
- panel caveats ("genes G, H are not on the panel — I will NOT treat their absence as
  negative evidence");
- the label-vocabulary contract (what each hierarchy level means; whether to reuse a
  specific paper's labels or a controlled ontology).

**How sign-off works — and how it scales down the hierarchy:**

- The **first** sign-off is **blocking**. Before producing a single label, render the
  current ledger (as a table, in chat — consistent with "stay in chat") and use
  **AskUserQuestion** to get an explicit decision (*Approve / Revise specific entries /
  Add a constraint*). Do **not** infer sign-off from silence. Only on an affirmative
  answer do you record the level in `run_state.json` (`gates.signed_off_levels`) and
  proceed to assign labels.
- At each **subsequent** level you *append* that level's new assumptions to the ledger
  and **re-prompt only when you introduce a materially new assumption** (a new
  marker→type mapping, a new biological prior, a new panel caveat that changes a call).
  If a level adds nothing beyond what the user already approved in spirit, note what
  you appended and continue without a blocking pause.
- The invariant, regardless of depth: **never assign a label on an assumption the user
  has not had the chance to see.** This keeps deep (L3/L4) hierarchies from nagging the
  user at every recursion while preserving the gate's purpose.

### Gate B — Honest "Unknown" is a first-class label

When the evidence is genuinely irreconcilable (e.g. mutually exclusive lineage
markers co-expressed with no doublet explanation, or no marker set scoring above
noise), the correct label is **`Unknown`**, documented and flagged — not a forced
guess. A well-justified `Unknown` is a *success*, not a failure: it tells the user
exactly where the data or the panel can't support a call.

## 3. Top-down recursive algorithm

For each level of the hierarchy, and within each parent group, repeat:

1. **Assemble priors.** Pull candidate types for this level/context and their markers
   from the knowledge stack (consult it in the order owned by `knowledge_sources.md`).
2. **Panel-gate the markers.** Intersect every marker set with `adata.var_names`.
   **Record the dropped genes** in the evidence card. A candidate left with fewer than
   ~2 on-panel markers is **uncallable** on this panel (`score_signatures.py` flags it)
   — it contributes to `Unknown`, never to a confident call (§6).
3. **Score.** Run the scoring tools on the panel-gated sets:
   - per-cluster overlap: a transparent intersection of each cluster's top-DE genes
     with each marker set (this is a raw overlap, **not** `decoupler.mt.ora`);
   - per-cell robust scores: `decoupler.mt.aucell` and/or pyUCell (with negative
     markers where available).
   - Compute cluster markers with `rank_genes_groups(method="wilcoxon",
     tie_correct=True, pts=True)` + COSG for specificity (see
     `differential_expression.md`).
4. **Cross-check** against ≥1 automated hypothesis (CellTypist / SingleR /
   CellMapper) — as a *hypothesis*, flagging agreement/disagreement, never
   overwriting your marker reasoning.
5. **Gate A** (§2): extend the assumptions ledger; block for sign-off the first time,
   thereafter re-prompt only on a materially new assumption.
6. **Write an evidence card** per cluster (see §4).
7. **Assign** the label, or descend / sub-cluster / mark Unknown per §5–6.
8. **Critic pass** (§7): independent subagents try to refute the assigned labels.
9. **Reconcile**, then **build the HTML report** for this level and **checkpoint**
   (`run_state.json`).
10. **Descend** into each assigned parent group and recurse at the next level.

**Termination & failure handling.**

- *Level recursion* terminates when `current_level` is the last entry in `hierarchy`
  (descending further is undefined — stop).
- *Sub-cluster recursion* (§5) is bounded: attempt **at most one split** per cluster;
  if the split does not resolve the conflict, the cluster is `Unknown`, not split
  again. This is the base case that prevents an unbounded re-clustering loop.
- A cluster whose label is **refuted** by the critic with no surviving alternative
  becomes `Unknown` (Gate B) — `refuted` is a transient status, never a final label.
- If a scoring script errors mid-level, checkpoint `run_state.json` and surface the
  failure to the user rather than silently skipping the cluster.
- If a level lands >~50% of its clusters in `Unknown`, pause and consult the user
  before descending into mostly-empty parent groups — that pattern usually signals a
  wrong panel assumption, a bad embedding, or a normalization problem upstream.

## 4. Evidence cards

One per cluster per level, written as a durable artifact (so a critic subagent that
cannot see the chat or the `AnnData` can review it). A card records:

- cluster id, n cells, parent assignment, hierarchy level;
- top DE / COSG markers (with pct-expressing and effect sizes);
- the candidate types considered, each with its panel-gated marker set and the score
  it received;
- **dropped (off-panel) markers** for each candidate;
- automated-method predictions and whether they agree;
- the assigned label (or `Unknown`), normalized to a **CL ID**;
- a **confidence 0–1** anchored to *objective* signals — the score margin between the
  top-1 and top-2 candidate, the fraction of the winner's defining markers that are
  on-panel, and whether the automated cross-check agrees — **not** a free-form feeling.
  A label can be wrong *and* feel certain, so a high confidence with a thin top1–top2
  margin must be flagged as low-margin (the report highlights these);
- a one-line rationale;
- explicitly: what was rejected and why.

## 5. The sub-cluster decision

If a cluster co-expresses markers of mutually-exclusive types, it is either (a) a
real mixed/transitional state, (b) a doublet artifact, or (c) under-clustered. Before
concluding, **sub-cluster it on the provided embedding**
(`sc.tl.leiden(restrict_to=...)`) and re-score the competing pos/neg sets per
sub-cluster. An orthogonal cluster-purity check (e.g. ROGUE) corroborates whether a
split is warranted.

**Bound the recursion (base case):** attempt **at most one split** per cluster, and
only split into sub-clusters with enough cells to test (don't chase sub-clusters
below a usable size). If one split does not cleanly separate the competing identities,
**stop and mark `Unknown`** — do not keep re-splitting.

**Doublet caveat — do not invent a "novel type".** Distinguishing a real mixed state
(a) from a doublet artifact (b) *requires* a doublet signal, but doublet detection is
**out of scope** (the skill alerts, it does not compute QC). So:

- If a doublet score is already present in `obs` (Scrublet / scDblFinder / DoubletFinder
  — `inspect_adata.py` surfaces such columns), use it: high-doublet sub-clusters are
  artifacts, not types.
- If **no** doublet score exists, you **cannot** rule out a doublet. Do not promote a
  mutually-exclusive co-expressing cluster to a confident "novel/transitional type" —
  either mark it `Unknown` with the doublet ambiguity stated, or ask the user to
  provide a doublet score (Scrublet is CPU-cheap) before making the call.

Where a split is justified and identities separate cleanly, recurse the algorithm into
the sub-clusters (subject to the one-split bound above).

## 6. Honest "Unknown"

Reach for `Unknown` when: no candidate scores above the noise floor; or two
incompatible identities are equally supported and one split does not separate them
(§5); or the only fitting candidate is **uncallable** because its defining markers are
off-panel (`<2` on-panel markers — `score_signatures.py` flags this), making the call
unfalsifiable; or a co-expressing cluster cannot be distinguished from a doublet for
lack of a doublet score (§5). Document the specific conflict in the evidence card and
flag it in the report.

**Investigate before defaulting to `Unknown`.** A large or odd cluster that resists a
clean marker call is often a real early/transitional state, not noise. Run
`scripts/profile_clusters.py` first: QC medians (vs global), composition over any
user-named categorical `obs` (e.g. a timepoint or batch — supplied at Stage 1, never
assumed), and optional spatial spread. Good-quality cells concentrated in one
developmental stage, with coherent (if non-canonical) markers, are a *type to name*,
not an `Unknown`. Only after this check fails to resolve the identity is `Unknown`
the honest call.

## 7. Critic subagents

After labels are assigned for a level, spawn **independent critic subagents** whose
job is to **refute**, not rubber-stamp. Default the critic to skepticism, and argue the
strongest case *against* each label: alternative types that fit the same markers,
ignored negative markers, panel artifacts, doublet explanations.

**Guard against the shared-blind-spot trap.** The critic and the annotator are the same
model, and the evidence card is the annotator's *own argument* — a critic handed only
the card inherits its framing and will tend to rubber-stamp. To make the challenge real:

- The critic reads the **raw exported tables** (`markers_<level>.csv`,
  `cosg_<level>.csv`, `scores_*.csv`, `panel_gating_<level>.csv`) — not just the
  evidence card — and **re-derives** the top markers itself, treating the card as the
  claim under test rather than as input.
- The critic must fetch **at least one competing hypothesis from an external source**
  (the `knowledgebase` MCP, plus `zotero` if configured) so its prior is not purely the
  base model's recall — this is what makes it *independent* rather than a second opinion
  from the same memory. For **per-marker corroboration**, the `knowledgebase` MCP exposes
  free, on-machine validation tools the critic should use: STRING (`bc_get_string_interactions`
  / `bc_get_string_similarity_scores` — do the claimed markers co-function?), GO
  (`bc_get_go_terms_by_gene` — does a marker's annotation fit the claimed identity?),
  Open Targets and HPA (`bc_get_human_protein_atlas_info` — tissue/expression grounding).
- The critic gets **read-only knowledge-MCP access** to check a label's ontology/marker
  basis, but **no** access to the in-memory `AnnData` or the chat history.

**Critique the label's place in the hierarchy, not only its biology.** A requested
hierarchy must nest, so the critic also checks each label against its **parent path**
(L1→…→L_k):

- *Structural* nesting (every finer label has exactly one parent) is a deterministic
  invariant enforced in **code** — `cta_io.check_hierarchy_nesting`, run by
  `write_back.py` (and any time via `--dry-run`). Don't rely on an LLM to never miss it;
  the critic need not re-derive it.
- *Semantic* path-consistency is the critic's job — flag a path that **passes** the
  structural check but is biologically incoherent: a **maturation** mismatch (an L3
  "IPC"/"neuron" under an L1 "NPC/RG" parent, or an L1 "Neuron" whose L3 is a progenitor),
  a **regional** contradiction (an L3 "thalamic" under an L2 "telencephalic"), or a finer
  label that is not actually a subtype of its parent. The structural check cannot catch
  these — they need judgment.

Reconcile objections before finalizing: either the label survives a real challenge, or
it is revised / downgraded to `Unknown`. When a critic raises a substantive conflict you
**cannot** resolve from the exported evidence (e.g. two identities equally supported),
surface the competing interpretations to the user *before* finalizing the level — don't
silently pick a side or silently downgrade.

## 8. Panel-aware rules (spatial / targeted panels)

- **Absence of a gene is not absence of expression** if the gene is off-panel.
  Panel-gate everything and never use an off-panel gene as negative evidence.
- Expect **segmentation spillover** on spatial data → false shared markers between
  spatially adjacent types; lean on COSG specificity and spatial coherence.
- Spatial context (neighborhood composition, niche) is corroborating evidence, not a
  substitute for marker evidence.
- Detect-and-alert only: if you see signs of unaddressed normalization, batch, or
  doublet/ambient problems, surface them — the skill does not silently fix QC.
