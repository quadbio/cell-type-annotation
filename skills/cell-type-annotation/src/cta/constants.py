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
      --strong-thresh      COSG "strong marker" cutoff          (cluster confidence)
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

# --- Callability gate (panel-dependent) ---------------------------------------------
# A candidate type needs at least this many of its positive markers ON the panel to be
# scored at all; below it the type is UNCALLABLE on this panel (a contributor to Unknown,
# not a zero score that misreads as "absent"). Safe to raise on rich panels; lower only on
# very small targeted panels. Enforced by: `cta markers score` (reference/methodology.md §6).
MIN_ON_PANEL_MARKERS = 2

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
# `cta cluster confidence` suggests a 0-1 band from COSG specificity as a sanity check on
# hand-set confidences (it never writes labels). Each band is (min cosg_top1, min n_strong,
# suggested_conf), tried in order; CONFIDENCE_BASE applies if none match. Each extra
# concurring cross-check method adds CONFIDENCE_AGREEMENT_BOOST, capped at CONFIDENCE_CAP.
CONFIDENCE_BANDS = ((0.05, 3, 0.9), (0.03, 2, 0.8), (0.02, 0, 0.7), (0.01, 0, 0.6))
CONFIDENCE_BASE = 0.45
CONFIDENCE_AGREEMENT_BOOST = 0.05
CONFIDENCE_CAP = 0.95
