# Reference: Annotation methodology

This is the *why and how-deep* behind Stage 3 (Execute) — the rationale, the two gates,
the evidence-card and critic design, and the cross-cutting rules. The **executable
step-by-step loop itself lives in SKILL.md Stage 3** (with the exact `cta` commands); read
this file for the reasoning the steps reference. See `marker_tools.md`,
`differential_expression.md`, `automated_tools.md`, and `knowledge_sources.md` for the
tool details.

## Table of contents

1. Core stance
2. The two hard gates
3. Top-down recursive algorithm — shape & termination
4. Evidence cards
5. The sub-cluster decision
6. Honest "Unknown" — and the parallel `state` axis
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

**Running non-interactively (opt-in — for benchmarks / automated runs).** The gates are
driven by *you*, the annotator, via `AskUserQuestion`; the `cta` tools are already
non-interactive. So "non-interactive mode" is a behavioural contract, not a flag, and it is
**opt-in only** — never the default, never a quiet reduction of the normal flow. It applies
when the user has **both** (a) explicitly asked for an unattended/benchmark run and (b)
supplied up front the context each gate would otherwise elicit: the data keys/layers, the
biology, the label-vocabulary contract, and the marker→type assumptions (a pre-filled
`context.md` / `assumptions.md` / `plan.md`, or an unambiguous brief). Under that contract:

- **Auto-approve-and-record, don't skip.** Still write `assumptions.md` for every level and
  still build every level report; record the sign-off in `run_state.json`
  `gates.signed_off_levels`, tagged so the audit trail shows it was *pre-authorized* (e.g.
  add `"auto_approved": "non_interactive"` alongside the level). The artifacts a reviewer
  reads must be indistinguishable from an interactive run — the only difference is that no
  blocking prompt was rendered.
- **Gaps still stop.** If the supplied context does not answer a gate — a materially new
  assumption with no provided basis, a `>~50%`-`Unknown` level, an irreconcilable
  critic conflict (§7) — **halt and surface it**, exactly as interactively. Non-interactive
  removes the *prompt for things already answered*, not the gate itself.
- **Be explicit in the summary.** State that the run was non-interactive and list which
  gates were auto-approved against which supplied context, so nothing reads as silently
  decided.

### Gate B — Honest "Unknown" is a first-class label

When the evidence is genuinely irreconcilable (e.g. mutually exclusive lineage
markers co-expressed with no doublet explanation, or no marker set scoring above
noise), the correct label is **`Unknown`**, documented and flagged — not a forced
guess. A well-justified `Unknown` is a *success*, not a failure: it tells the user
exactly where the data or the panel can't support a call.

## 3. Top-down recursive algorithm — shape & termination

The **executable loop** — assemble priors → panel-gate → score → cross-check → Gate A →
fill the labels CSV (= assign, or sub-cluster / `Unknown`) → critic+reconcile →
report+gate+descend — is **SKILL.md Stage 3**, with the exact `cta` commands and the
per-step detail. This section is only what that loop's steps don't spell out: the
recursion's shape, and how it ends.

Annotation runs **top-down through the hierarchy and recursively within each parent
group** (§1): resolve a parent class before its subtypes, and descend into a group only
once its parent label is assigned. The numeric gates the steps enforce
(`MIN_ON_PANEL_MARKERS`, etc.) live in one place, `src/cta/constants.py` — tune them there
for an unusually small or rich panel.

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
cannot see the chat or the `AnnData` can review it).

**Where the reasoning lives — the labels CSV, not the markdown.** The per-level labels
table `labels/annotation_<level>.csv` is the single source of truth: it carries
`label,state,cl_id,confidence,rationale,rejected` columns (`state` = the orthogonal cell-state
axis, §6), and `cta report scaffold-cards` *renders* the
cards from it. This is a deliberate design choice — authoring rationale directly into N
markdown files is token-heavy, trivially skipped, and destroyed by a re-scaffold, which is
exactly how a run ends up "finished" with empty TODO cards. Filling the CSV columns instead
makes the reasoning (a) durable across re-scaffolds, (b) recoverable on resume from one
file, and (c) machine-checkable: `cta report build --strict` blocks descent until every
cluster has a label and a non-blank rationale (§ the level-completeness gate). Fill the CSV,
re-scaffold, then build the report with `--strict`.

A card records:

- cluster id, n cells, parent assignment, hierarchy level;
- top DE / cluster-specific (filter_rank_genes_groups) markers (with pct-expressing and effect sizes);
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
(a) from a doublet artifact (b) *requires* a doublet signal. The skill computes standard
QC metrics + a cell-cycle signal (`cta data qc`), but **doublet detection**
(Scrublet / scDblFinder) stays **out of scope** — by "doublet" here we mean a cluster
mixing two or more cell types, whose canonical handling is **sub-clustering**, not a
detector score. So:

- The first move on a mutually-exclusive co-expressing cluster is to **sub-cluster and
  re-score** (above): a true doublet usually splits into its constituent single types.
- If a doublet score is already present in `obs` (Scrublet / scDblFinder / DoubletFinder
  — `cta data inspect` surfaces such columns), use it as corroboration: high-doublet
  sub-clusters are artifacts, not types.
- If a split does **not** separate the identities and you cannot rule out a doublet, do
  **not** promote the cluster to a confident "novel/transitional type": record it on the
  **state axis** as `state=doublet` with `label=Unknown` (§6), stating the ambiguity — or
  ask the user for a doublet score (Scrublet is CPU-cheap) before making any positive call.

Where a split is justified and identities separate cleanly, recurse the algorithm into
the sub-clusters (subject to the one-split bound above).

## 6. Honest "Unknown" — and the parallel `state` axis

**Two axes, not one.** A cluster has a cell-**type** (its lineage identity) *and* a cell-**state**
(its quality / cycle / artifact status). Forcing both onto a single label is what produced the
dominant blind-run failures: a cycling cluster mislabeled because cell-cycle genes dominate its
markers, or a low-quality cluster given a confident type it can't support. So the labels CSV
carries a `label` **and** a `state` column, written to obs side by side
(`<prefix>_<level>` and `<prefix>_<level>_state` by `cta report write-back`):

- **`state` is orthogonal to `label`** and **exempt from CL-ID normalization and the
  hierarchy-nesting check** — it is not a cell type. Recommended vocabulary:
  `normal` / `low-quality` / `cycling` / `doublet` (extend as the tissue needs, e.g.
  `stressed`, `activated`). Leave it blank/`normal` for ordinary clusters.
- **Cycling** is the key win: a proliferating cluster almost always still has a real lineage.
  Keep the type call and set `state=cycling` — do **not** downgrade it to `Unknown` just
  because `S_score`/`G2M_score` are high. The cross-check (reference mapping) and
  cell-cycle-robust markers usually still recover the type.
- **Low-quality** (low `total_counts`/`n_genes`, high `pct_counts_mt`, off-population on
  `qc_deviation_<level>.csv`): when the depth/mito profile makes the type unsupportable, set
  `state=low-quality` and `label=Unknown` — the honest outcome, now *named* rather than guessed.
- **Doublet**: handled by sub-clustering first (§5); `state=doublet` + `label=Unknown` is the
  terminal call only when a split can't separate the constituent types.

The evidence for `state` comes from `cta data qc` (read-only QC + cell-cycle written into the
canonical working copy, never the user's original) surfaced per-cluster by `cta cluster profile`.
This is detection, not correction — the skill still never normalizes, filters, or removes doublets.

Reach for `Unknown` (the **type** axis) when: no candidate scores above the noise floor; or two
incompatible identities are equally supported and one split does not separate them
(§5); or the only fitting candidate is **uncallable** because its defining markers are
off-panel (`<2` on-panel markers — `cta markers score` flags this), making the call
unfalsifiable; or a co-expressing cluster cannot be distinguished from a doublet for
lack of a doublet score (§5). Document the specific conflict in the evidence card and
flag it in the report.

**Investigate before defaulting to `Unknown`.** A large or odd cluster that resists a
clean marker call is often a real early/transitional state, not noise. Run
`cta cluster profile` first: QC medians (vs global), composition over any
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
  `markers_filtered_<level>.csv`, `scores_*.csv`, `panel_gating_<level>.csv`) — not just the
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
  `cta report write-back` (and any time via `--dry-run`). Don't rely on an LLM to never miss it;
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

**Scaling the critic to many clusters (batch + prioritize).** One subagent per cluster is
right for a handful of clusters, but a level with dozens makes that a token/latency blowout.
Scale the fan-out without dropping rigour:

- **Batch.** Hand each critic subagent a *group* of related clusters (e.g. all the clusters
  competing for one parent identity, or 5–10 at a time) rather than one each — the critic
  re-derives markers across the batch and can directly contrast neighbours (the most common
  confusions are between adjacent clusters anyway).
- **Prioritize, don't uniformly sample.** Spend the most scrutiny where a wrong call is most
  likely: critic **every** flagged cluster individually — `Unknown`, low confidence, thin
  top1–top2 margin, an uncallable winner, a hierarchy/maturation mismatch — and batch the
  clean, high-margin, cross-check-agreeing ones more coarsely. If you cap or sample the
  confident tier to bound cost, **say so** in the report (which clusters got a full vs. a
  batched review) — never let a sampled pass read as exhaustive.
- The reconciliation bar is unchanged: a label that a batched critic challenges
  substantively still gets the individual follow-up before it is finalized.

## 8. Panel-aware rules (spatial / targeted panels)

- **Absence of a gene is not absence of expression** if the gene is off-panel.
  Panel-gate everything and never use an off-panel gene as negative evidence.
- Expect **segmentation spillover** on spatial data → false shared markers between
  spatially adjacent types; lean on filter_rank_genes_groups specificity (the out-group-
  fraction filter) and spatial coherence.
- Spatial context (neighborhood composition, niche) is corroborating evidence, not a
  substitute for marker evidence.
- Detect-and-alert only: if you see signs of unaddressed normalization, batch, or
  doublet/ambient problems, surface them — the skill does not silently fix QC. (It *does*
  compute read-only QC metrics + a cell-cycle signal via `cta data qc`, into the working
  copy only, so low-quality/cycling clusters are detectable — computing a metric is not
  fixing the data, and doublet detection/normalization/filtering remain out of scope.)
