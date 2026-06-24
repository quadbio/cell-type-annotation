"""Unit tests for reference gene-ID alignment (issue #24).

The canonical query is symbol-indexed (`cta data canonicalize`), so the reference's
var_names are aligned to symbols before the var_names intersection (`reference-map`) or
DE (`markers assemble --reference`). These tests exercise the alignment helper and the
shared-gene recovery directly on synthetic AnnData — no scanpy / console-script cold start,
so they stay fast (the reasoning that matters is pure-Python).
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from cta.canonicalize import align_var_names_to_symbols
from cta.reference_map import _shared_genes

# A reference whose var index is Ensembl with a `feature_name` symbol column — the common
# CELLxGENE shape. The first 40 genes have symbols; the last 5 are symbol-less (kept Ensembl).
N_SYMBOLED = 40
N_UNSYMBOLED = 5


def _ensembl_reference():
    rng = np.random.default_rng(0)
    n_vars = N_SYMBOLED + N_UNSYMBOLED
    a = ad.AnnData(rng.random((30, n_vars)).astype("float32"))
    a.var_names = [f"ENSG{i:011d}" for i in range(n_vars)]
    symbols = [f"GENE{i}" for i in range(N_SYMBOLED)] + [""] * N_UNSYMBOLED
    a.var["feature_name"] = symbols
    return a, symbols


def test_align_ensembl_reference_with_symbol_column():
    """Ensembl var_names + feature_name -> symbol var_names; Ensembl ids preserved; the
    symbol-less tail keeps its Ensembl id (nothing dropped)."""
    ref, symbols = _ensembl_reference()
    info = align_var_names_to_symbols(ref)

    assert info["convention"] == "ensembl->symbol"
    assert info["renamed"] == N_SYMBOLED
    assert info["unmapped"] == N_UNSYMBOLED
    assert list(ref.var_names[:N_SYMBOLED]) == symbols[:N_SYMBOLED]
    # symbol-less genes stay Ensembl
    assert all(str(n).startswith("ENSG") for n in ref.var_names[N_SYMBOLED:])
    # original ids round-trippable
    assert "ensembl_id" in ref.var.columns
    assert ref.var["ensembl_id"].iloc[0].startswith("ENSG")


def test_align_recovers_shared_genes_with_symbol_query():
    """A symbol-canonical query vs an Ensembl reference shares ~0 genes verbatim, but the
    symbol intersection after alignment clears MIN_SHARED_GENES."""
    ref, symbols = _ensembl_reference()
    query = ad.AnnData(np.zeros((5, N_SYMBOLED), dtype="float32"))
    query.var_names = symbols[:N_SYMBOLED]  # query is symbol-indexed

    # Before alignment the Ensembl reference shares nothing with the symbol query.
    with pytest.raises(SystemExit):
        _shared_genes(query, ref)

    align_var_names_to_symbols(ref)
    shared = _shared_genes(query, ref)  # would raise if < MIN_SHARED_GENES
    assert set(shared) == set(symbols[:N_SYMBOLED])


def test_align_with_symbol_map(tmp_path):
    """No symbol column -> a --ref-symbol-map TSV renames only the mapped Ensembl genes."""
    rng = np.random.default_rng(1)
    n_vars = 4
    a = ad.AnnData(rng.random((10, n_vars)).astype("float32"))
    a.var_names = [f"ENSG{i:011d}" for i in range(n_vars)]
    mapping = tmp_path / "ref_symbol_map.tsv"
    mapping.write_text(f"{a.var_names[0]}\tAQP4\n{a.var_names[1]}\tGFAP\n")

    info = align_var_names_to_symbols(a, symbol_map_path=str(mapping))
    assert info["renamed"] == 2 and info["unmapped"] == 2
    assert list(a.var_names) == ["AQP4", "GFAP", f"ENSG{2:011d}", f"ENSG{3:011d}"]


def test_align_already_symbols_is_noop():
    """A symbol-indexed reference is left untouched (convention == 'already symbols')."""
    a = ad.AnnData(np.zeros((6, 3), dtype="float32"))
    a.var_names = ["AQP4", "GFAP", "CD3D"]
    info = align_var_names_to_symbols(a)
    assert info["convention"] == "already symbols"
    assert list(a.var_names) == ["AQP4", "GFAP", "CD3D"]


def test_align_no_symbol_source_errors():
    """Ensembl var_names with no symbol column and no --ref-symbol-map is a hard error —
    the same contract as `cta data canonicalize`, naming the --ref- flags."""
    a = ad.AnnData(np.zeros((6, 3), dtype="float32"))
    a.var_names = [f"ENSG{i:011d}" for i in range(3)]
    with pytest.raises(SystemExit) as exc:
        align_var_names_to_symbols(a)
    assert "--ref-symbol-map" in str(exc.value)


def test_align_integer_varnames_with_symbol_column():
    """`cta reference fetch-census` writes positional integer var_names ('0','1',…) with
    real symbols in `var['feature_name']`. Auto-detection must NOT mistake the integer ids
    for "already symbols" — it must rename from feature_name (regression for the
    fetch-census -> reference-map "0 shared genes" bug)."""
    rng = np.random.default_rng(2)
    n_vars = N_SYMBOLED + N_UNSYMBOLED
    a = ad.AnnData(rng.random((20, n_vars)).astype("float32"))
    a.var_names = [str(i) for i in range(n_vars)]  # positional integers, as Census writes
    symbols = [f"GENE{i}" for i in range(N_SYMBOLED)] + [""] * N_UNSYMBOLED
    a.var["feature_name"] = symbols

    info = align_var_names_to_symbols(a)
    assert info["convention"] != "already symbols"
    assert info["renamed"] == N_SYMBOLED
    assert list(a.var_names[:N_SYMBOLED]) == symbols[:N_SYMBOLED]
    # symbol-less tail keeps its original positional id (nothing dropped)
    assert list(a.var_names[N_SYMBOLED:]) == [str(i) for i in range(N_SYMBOLED, n_vars)]


def test_align_integer_varnames_explicit_symbol_col():
    """An explicit `--ref-symbol-col` is honored UNCONDITIONALLY — even past auto-detection —
    renaming integer var_names from the named column."""
    rng = np.random.default_rng(3)
    a = ad.AnnData(rng.random((8, 3)).astype("float32"))
    a.var_names = ["0", "1", "2"]
    a.var["feature_name"] = ["AQP4", "GFAP", "CD3D"]
    info = align_var_names_to_symbols(a, symbol_col="feature_name")
    assert info["renamed"] == 3
    assert list(a.var_names) == ["AQP4", "GFAP", "CD3D"]


def test_align_integer_varnames_recovers_shared_genes():
    """End-to-end regression: an integer-indexed Census reference shares 0 genes verbatim
    with a symbol query, but clears MIN_SHARED_GENES once aligned via feature_name."""
    rng = np.random.default_rng(4)
    n_vars = N_SYMBOLED + N_UNSYMBOLED
    ref = ad.AnnData(rng.random((20, n_vars)).astype("float32"))
    ref.var_names = [str(i) for i in range(n_vars)]
    symbols = [f"GENE{i}" for i in range(N_SYMBOLED)] + [""] * N_UNSYMBOLED
    ref.var["feature_name"] = symbols

    query = ad.AnnData(np.zeros((5, N_SYMBOLED), dtype="float32"))
    query.var_names = symbols[:N_SYMBOLED]

    with pytest.raises(SystemExit):  # integer ref vs symbol query: nothing shared
        _shared_genes(query, ref)

    align_var_names_to_symbols(ref)
    shared = _shared_genes(query, ref)  # would raise if < MIN_SHARED_GENES
    assert set(shared) == set(symbols[:N_SYMBOLED])


def test_align_explicit_symbol_col_missing_errors():
    """An explicit `--ref-symbol-col` that isn't in var is a hard error naming the flag —
    even when var_names would otherwise auto-classify (here, plain symbols)."""
    a = ad.AnnData(np.zeros((4, 3), dtype="float32"))
    a.var_names = ["AQP4", "GFAP", "CD3D"]
    with pytest.raises(SystemExit) as exc:
        align_var_names_to_symbols(a, symbol_col="feature_name")
    assert "--ref-symbol-col" in str(exc.value)


def test_from_reference_emits_symbol_markers_not_raw_ensembl(tmp_path):
    """`markers assemble --reference` on an Ensembl atlas with an Ensembl `.raw` (CELLxGENE
    shape) must DE on the aligned .X and return SYMBOL marker names — never the Ensembl ids
    that `.raw` carries (rank_genes_groups defaults to use_raw=True, which we override)."""
    import scanpy as sc

    from cta.assemble_markers import from_reference

    rng = np.random.default_rng(0)
    n_genes = 12
    symbols = [f"SYM{i}" for i in range(n_genes)]
    # two cleanly separable groups: each over-expresses its own gene block
    blocks, labels = [], []
    for ti in range(2):
        base = np.full(n_genes, 1.0)
        base[ti * 6:(ti + 1) * 6] = 30.0
        blocks.append(rng.poisson(np.tile(base, (15, 1))).astype("float32"))
        labels += [f"type{ti}"] * 15
    counts = np.vstack(blocks)
    a = ad.AnnData(counts.copy())
    a.var_names = [f"ENSG{i:011d}" for i in range(n_genes)]
    a.var["feature_name"] = symbols
    a.obs["cell_type"] = labels
    a.raw = a.copy()              # .raw keeps Ensembl var_names + raw counts (the trap)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)                # .X is now log-norm; .raw stays Ensembl counts
    ref = tmp_path / "ens_ref.h5ad"
    a.write_h5ad(ref)

    markers = from_reference(str(ref), "cell_type", n_genes=5, counts=False)
    all_genes = {g for v in markers.values() for g in v}
    assert all_genes, "no markers returned"
    assert not any(g.startswith("ENSG") for g in all_genes), \
        f"DE leaked Ensembl ids (used .raw?): {sorted(g for g in all_genes if g.startswith('ENSG'))}"
    assert all_genes <= set(symbols)
