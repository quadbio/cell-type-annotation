"""Unit tests for the shared stratified subsampler (cta.io.stratified_subsample_positions).

Pure / no-network: validates the allocation contract and proves the refactor preserves the
exact selection of the two legacy callers (reference_map positions, fetch_census joinids).
"""
from __future__ import annotations

import numpy as np

from cta.io import stratified_subsample_positions as subsample


def _groups(n_rare=5, n_common=95):
    # one rare category and one common one, to test the floor-1 guarantee
    return np.array(["rare"] * n_rare + ["common"] * n_common)


def test_short_circuit_returns_all():
    out = subsample(10, 50)
    assert np.array_equal(out, np.arange(10))


def test_determinism_same_seed():
    g = _groups()
    assert np.array_equal(subsample(100, 20, g), subsample(100, 20, g))


def test_unstratified_cap_and_sorted():
    out = subsample(100, 20)
    assert out.size == 20
    assert np.array_equal(out, np.sort(out))
    assert len(set(out.tolist())) == 20  # no duplicates


def test_stratified_rare_group_survives_and_respects_cap():
    g = _groups(n_rare=3, n_common=997)  # rare would round to 0 without the floor
    out = subsample(1000, 20, g)
    assert out.size <= 20
    # at least one rare-category cell is included (floor of 1)
    assert any(g[i] == "rare" for i in out)


def _legacy_positions(groups, max_n, seed=0):
    """The pre-refactor reference_map._subsample_indices stratified logic (positions)."""
    rng = np.random.default_rng(seed)
    cats, counts = np.unique(groups, return_counts=True)
    alloc = np.maximum(1, np.round(counts / counts.sum() * max_n).astype(int))
    picks = []
    for cat, n_take in zip(cats, alloc):
        idx = np.flatnonzero(groups == cat)
        picks.append(rng.choice(idx, size=min(int(n_take), idx.size), replace=False))
    sel = np.concatenate(picks)
    if sel.size > max_n:
        sel = rng.choice(sel, size=max_n, replace=False)
    return np.sort(sel)


def _legacy_census_ids(ids, groups, max_cells, seed=0):
    """The pre-refactor fetch_census._choose_joinids stratified logic (id values)."""
    rng = np.random.default_rng(seed)
    cats, counts = np.unique(groups, return_counts=True)
    alloc = np.maximum(1, np.round(counts / counts.sum() * max_cells).astype(int))
    picks = []
    for cat, n_take in zip(cats, alloc):
        cat_ids = ids[groups == cat]
        picks.append(rng.choice(cat_ids, size=min(int(n_take), cat_ids.size), replace=False))
    sel = np.concatenate(picks)
    if sel.size > max_cells:
        sel = rng.choice(sel, size=max_cells, replace=False)
    return np.sort(sel)


def test_equivalent_to_legacy_reference_map_positions():
    g = _groups(n_rare=12, n_common=288)
    assert np.array_equal(subsample(300, 40, g), _legacy_positions(g, 40))


def test_equivalent_to_legacy_census_joinids():
    # non-contiguous id domain (like soma_joinids) to exercise the position->id mapping
    ids = np.arange(300) * 7 + 3
    g = _groups(n_rare=12, n_common=288)
    new = np.sort(ids[subsample(300, 40, g)])
    assert np.array_equal(new, _legacy_census_ids(ids, g, 40))
