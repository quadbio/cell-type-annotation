"""Unit tests for the scArches query-vs-registry sanitizer (cta.map_query_scarches).

Pure / no-network / no-scvi: validates the reconciliation contract from issue #10 — shipped
hub models register a batch_key/labels_key the real query atlas does not satisfy, so before
`load_query_data` we inject the batch_key (constant, when absent) and force the labels_key to
the model's unlabeled_category. The module imports cleanly without scvi-tools because the
`import scvi` lives inside the Typer command body.
"""

from __future__ import annotations

import anndata as ad
import numpy as np

from cta.map_query_scarches import sanitize_query_against_registry


def _query(obs=None):
    a = ad.AnnData(np.zeros((4, 3), dtype="float32"))
    if obs:
        for k, v in obs.items():
            a.obs[k] = v
    return a


def test_absent_batch_key_injected_as_constant():
    q = _query()
    notes = sanitize_query_against_registry(q, {"batch_key": "donor_assay"})
    assert list(q.obs["donor_assay"]) == ["query"] * 4
    assert any("donor_assay" in n and "injected" in n for n in notes)


def test_present_batch_key_preserved():
    q = _query({"donor_assay": ["a", "a", "b", "b"]})
    notes = sanitize_query_against_registry(q, {"batch_key": "donor_assay"})
    assert list(q.obs["donor_assay"]) == ["a", "a", "b", "b"]
    assert any("preserved" in n for n in notes)


def test_out_of_vocab_labels_forced_to_unlabeled():
    # the failing case from the issue: labels_key present with categories the model never saw
    q = _query({"cell_type": ["Tcell", "Bcell", "weird", "novel"]})
    notes = sanitize_query_against_registry(
        q, {"labels_key": "cell_type", "unlabeled_category": "Unknown"}
    )
    assert list(q.obs["cell_type"]) == ["Unknown"] * 4
    assert any("cell_type" in n and "Unknown" in n for n in notes)


def test_absent_labels_key_created_as_unlabeled():
    q = _query()
    sanitize_query_against_registry(q, {"labels_key": "cell_type", "unlabeled_category": "Unknown"})
    assert "cell_type" in q.obs.columns
    assert list(q.obs["cell_type"]) == ["Unknown"] * 4


def test_labels_key_without_unlabeled_falls_back_to_unknown():
    q = _query()
    sanitize_query_against_registry(q, {"labels_key": "cell_type"})
    assert list(q.obs["cell_type"]) == ["Unknown"] * 4


def test_no_registered_keys_is_noop():
    q = _query({"leiden": ["0", "0", "1", "1"]})
    before = list(q.obs.columns)
    notes = sanitize_query_against_registry(q, {})
    assert notes == []
    assert list(q.obs.columns) == before


def test_both_keys_applied_together():
    q = _query({"cell_type": ["x", "y", "z", "w"]})
    sanitize_query_against_registry(
        q, {"batch_key": "donor_assay", "labels_key": "cell_type", "unlabeled_category": "Unknown"}
    )
    assert list(q.obs["donor_assay"]) == ["query"] * 4
    assert list(q.obs["cell_type"]) == ["Unknown"] * 4
