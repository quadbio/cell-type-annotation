"""Routing, evidence checks, label hierarchies, reports and the hierarchy figure for cell-type annotation."""

from importlib.metadata import version

from cell_type_annotation import pl, routing
from cell_type_annotation.checks import (
    cell_cycle_genes,
    confidence_proxy,
    negative_marker_flags,
    normalization_fingerprint,
    panel_gate,
    qc_deviation,
    validate_cl_ids,
)
from cell_type_annotation.hierarchy import LabelHierarchy, build_hierarchy, hierarchy_markers
from cell_type_annotation.markers import display_markers, read_markers
from cell_type_annotation.report import LABEL_COLUMNS, check_level, prediction_summary, render_level

__all__ = [
    "LABEL_COLUMNS",
    "LabelHierarchy",
    "build_hierarchy",
    "cell_cycle_genes",
    "check_level",
    "confidence_proxy",
    "display_markers",
    "hierarchy_markers",
    "negative_marker_flags",
    "normalization_fingerprint",
    "panel_gate",
    "pl",
    "prediction_summary",
    "qc_deviation",
    "read_markers",
    "render_level",
    "routing",
    "validate_cl_ids",
]
__version__ = version("cell-type-annotation")
