"""Tunable science / methodology constants — the one place to find and adjust the
numeric knobs the `cta` tools enforce.

These are the panel-/organism-/dataset-dependent values that were previously buried as
magic numbers across the toolkit. They live here so a user (or the lab) can discover and
tune them in ONE file. Because `cta` is installed editable into the pixi env, editing a
value here takes effect on the next run — no reinstall.

Scope — what is and isn't here:
  * Per-run knobs that ALREADY have CLI flags are not duplicated here; set those per
    invocation (`cta <command> --help`). Index of the main ones, with their command:
      --n-genes            top markers per cluster / per type  (markers compute, crosscheck celltypist markers)
      --resolution / --n-neighbors  subcluster Leiden          (cluster subcluster)
      --strong-thresh      log2FC "strong marker" cutoff        (cluster confidence)
      --min-targets        decoupler signature min on-panel size (markers score)
      --max-cells / --max-ref  reference subsample caps          (reference fetch-census / crosscheck reference-map)
      --n-comps            joint-embedding components            (crosscheck reference-map)
      --n-latent / --max-epochs* / --n-samples-per-label  scVI/scANVI training (reference train-model / map-scarches)
  * Reasoning-level thresholds that Claude APPLIES while annotating (the conventional DE
    filters, the ">~50% Unknown -> pause" rule, the SCimilarity RAM guide) live in the
    reference docs Claude reads, NOT here — see reference/differential_expression.md and
    reference/methodology.md. This file holds only what the Python code enforces.
"""

from __future__ import annotations

import re

# --- Callability gate (panel-dependent) ---------------------------------------------
# A candidate type needs at least this many of its positive markers ON the panel to be
# scored at all; below it the type is UNCALLABLE on this panel (a contributor to Unknown,
# not a zero score that misreads as "absent"). Safe to raise on rich panels; lower only on
# very small targeted panels. Enforced by: `cta markers score` (reference/methodology.md §6).
MIN_ON_PANEL_MARKERS = 2

# --- Upfront gene filter (canonicalization) ------------------------------------------
# `cta data canonicalize` drops genes detected in fewer than this many cells when it builds the
# canonical slim, so every downstream step sees one clean, consistent gene set. A small ABSOLUTE
# floor (the scanpy tutorial uses 3, best-practices 20) that removes only genes in ≤4 cells
# (unambiguous ambient/technical noise). It is absolute ON PURPOSE — a relative-to-n_obs threshold
# (e.g. 0.1% = 500 cells on a 500k atlas) would filter out the specific markers of RARE clusters: a
# gene that perfectly marks a 27-cell microglia cluster (P2RY12) is expressed in ≤27 cells and must
# survive. A global per-gene cell-count filter is bounded by the SMALLEST cluster you want to
# annotate, which is small — so keep it low and absolute. Enforced by `cta data canonicalize`.
GENE_FILTER_MIN_CELLS = 5

# --- Negative-marker violation threshold (panel-dependent) --------------------------
# A curated NEGATIVE marker is one whose expression is EXPECTED to be absent in a candidate
# type (e.g. DCX for an NPC, MS4A1 for a T cell). `cta markers profile` flags a violation when
# a negative marker is detected in more than this fraction of a cluster's cells — i.e. it is
# "on" where the call says it should be off, which is evidence AGAINST that call for the critic
# pass to weigh. An off-panel negative marker is UNMEASURED, never "confirmed absent", so it is
# reported, not scored here. Enforced by: `cta markers profile` (writes panel_neg_flags_<level>.csv).
#
# Calibrated on 5 real atlases (Immune Cell Atlas, GTEx v9, mouse pancreas, HypoMap, HNOCA — via
# celltype-anno-bench): genuine positive markers are expressed in ~50–68% of their home type's
# cells (10th pct 0.52), while a truly-absent marker's cross-type background (ambient + shared
# developmental programs) reaches ~10–20% in the noisiest data. So the two are well separated and
# the threshold has headroom: 0.20 clears ZERO genuine positives on all 5 datasets while cutting
# the false-violation rate on true negatives from ~18% (at 0.10) to ~5% mean — and, crucially for
# developmental/organoid data, from 43% to 10% on HNOCA. 0.10 cried wolf; 0.20 is a trustworthy
# flag. Raise toward 0.25 for very continuous/developmental data; lower only on clean, discrete,
# low-ambient data where you want maximum sensitivity.
NEG_MARKER_MAX_PCT = 0.20

# --- Reference-mapping gate (panel-dependent) ---------------------------------------
# Minimum genes shared between query and reference before a label transfer is trusted;
# below it `cta crosscheck reference-map` refuses (transfer over a handful of genes is noise).
MIN_SHARED_GENES = 30

# --- Normalization target (standard; change only with reason) -----------------------
# Target sum for sc.pp.normalize_total when the toolkit must (re)normalize a reference to
# the query's library-size convention. 1e4 ("CP10K") is the scverse / CellTypist / scvi-tools
# default and the cross-check assumes the query was normalized the same way.
NORMALIZE_TARGET_SUM = 1e4

# --- Report low-confidence flag (reporting housekeeping, not a science gate) ---------
# Clusters with confidence below this are flagged "LOW CONF" for review in the per-level
# HTML report. Enforced by: `cta report build`. (The report's own legend text also states
# this value — keep them in step if you change it.)
REPORT_LOW_CONF_THRESHOLD = 0.6

# --- Confidence proxy banding (advisory only) ---------------------------------------
# `cta cluster confidence` suggests a 0-1 band from the per-cluster specificity of the markers
# that survive `sc.tl.filter_rank_genes_groups` (which replaced COSG), as a sanity check on
# hand-set confidences (it never writes labels). The signal is now log2 fold-change: each band is
# (min lfc_top1, min n_strong, suggested_conf), where lfc_top1 is the strongest surviving marker's
# log2FC and n_strong counts surviving markers with log2FC >= --strong-thresh. Bands are tried in
# order; CONFIDENCE_BASE applies if none match. Each extra concurring cross-check method adds
# CONFIDENCE_AGREEMENT_BOOST, capped at CONFIDENCE_CAP. (Recalibrated from COSG's 0–1 cosine scale
# to the log2FC scale of filter_rank_genes_groups — adjust if your data's effect sizes differ.)
CONFIDENCE_BANDS = ((3.0, 5, 0.9), (2.0, 3, 0.8), (1.0, 1, 0.7), (0.5, 0, 0.6))
CONFIDENCE_BASE = 0.45
CONFIDENCE_AGREEMENT_BOOST = 0.05
CONFIDENCE_CAP = 0.95

# --- Cell Ontology id syntax (write-back gate) --------------------------------------
# `cta report write-back` SYNTACTICALLY validates any non-blank `cl_id` in the labels CSV
# before mutating the user's AnnData: it must be a real ontology accession of the form
# PREFIX:0000000. We accept the Cell Ontology (CL) and the Provisional Cell Ontology (PCL,
# used for fine neuronal subtypes that CL does not yet carry) — both use 7-digit numeric ids,
# matching the Cell Annotation Schema's `cell_ontology_term_id` rule. This is a CHEAP,
# OFFLINE, deterministic check that catches typos (`CL:000681`), wrong casing (`cl:0000681`),
# and free-text leaking into the id column. It does NOT confirm the term EXISTS, is not
# obsolete, or matches its label — that SEMANTIC check is done in-chat via the knowledgebase
# MCP (`bc_get_term_details`), because `cta` commands have no MCP access. See SKILL.md
# Stage 3.6 and reference/knowledge_sources.md. Extend the prefixes for other CL-extending
# ontologies (e.g. FBbt) if a run needs them.
CL_ID_ALLOWED_PREFIXES = ("CL", "PCL")
CL_ID_PATTERN = re.compile(r"^(?:" + "|".join(CL_ID_ALLOWED_PREFIXES) + r"):\d{7}$")
