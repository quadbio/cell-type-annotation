"""Behavioral tests for `cta data canonicalize` — the single Stage-1 step that writes the
standardized sidecar every downstream command relies on (symbol var_names, counts layer,
log-norm X). Run through the installed `cta` console script, like the other smoke tests.
"""
from __future__ import annotations

import json
import subprocess

import numpy as np


def _run_canonicalize(adata_path, out, *extra):
    return subprocess.run(
        ["cta", "data", "canonicalize", "--adata", str(adata_path), "--out", str(out),
         "--cluster-key", "leiden", "--embedding-key", "X_pca", *extra],
        capture_output=True, text=True,
    )


def _read(path):
    import anndata as ad
    return ad.read_h5ad(path)


def test_canonicalize_ensembl_column_to_symbols(ensembl_h5ad, tmp_path):
    """Ensembl var_names + a `feature_name` column -> symbol var_names, with the Ensembl
    IDs kept in var and a duplicate symbol made unique (none dropped)."""
    src, symbols = ensembl_h5ad
    out = tmp_path / "ens_cta.h5ad"
    r = _run_canonicalize(src, out)
    assert r.returncode == 0, r.stderr
    adata = _read(out)

    # symbols are now the var_names; the duplicate AQP4 was de-duplicated, not dropped
    assert adata.n_vars == len(symbols)
    assert not adata.var_names.duplicated().any()
    assert "AQP4" in set(adata.var_names) and "GFAP" in set(adata.var_names)
    # original Ensembl IDs preserved for round-tripping
    assert "ensembl_id" in adata.var.columns
    assert adata.var["ensembl_id"].iloc[0].startswith("ENSG")

    summary = json.loads((out.with_name("canonicalization.json")).read_text())
    assert summary["var"]["convention"] == "ensembl->symbol"
    assert summary["var"]["deduped"] == 1


def test_canonicalize_counts_from_raw(ensembl_h5ad, tmp_path):
    """Integer counts living in `.raw` are pulled into layers['counts'] (aligned to X)."""
    import scipy.sparse as sp

    src, _ = ensembl_h5ad
    out = tmp_path / "ens_cta.h5ad"
    r = _run_canonicalize(src, out)
    assert r.returncode == 0, r.stderr
    adata = _read(out)

    assert "counts" in adata.layers
    c = adata.layers["counts"]
    c = c.toarray() if sp.issparse(c) else np.asarray(c)
    assert c.shape == adata.shape
    assert np.allclose(c, np.round(c)) and c.min() >= 0  # integer counts
    summary = json.loads((out.with_name("canonicalization.json")).read_text())
    assert summary["counts_source"] == ".raw.X"


def test_canonicalize_symbol_map_partial(tmp_path):
    """With no symbol column, a --symbol-map renames only the mapped Ensembl genes; the
    rest keep their Ensembl id (nothing dropped)."""
    import anndata as ad

    rng = np.random.default_rng(2)
    var_names = [f"ENSG{i:011d}" for i in range(4)]
    adata = ad.AnnData(rng.random((20, 4)).astype("float32"))
    adata.var_names = var_names
    adata.obs["leiden"] = [str(i % 2) for i in range(20)]
    adata.obsm["X_pca"] = rng.standard_normal((20, 3)).astype("float32")
    src = tmp_path / "nocol.h5ad"
    adata.write_h5ad(src)

    # map only the first two genes
    mapping = tmp_path / "symbol_map.tsv"
    mapping.write_text(f"{var_names[0]}\tAQP4\n{var_names[1]}\tGFAP\n")

    out = tmp_path / "nocol_cta.h5ad"
    r = _run_canonicalize(src, out, "--symbol-map", str(mapping))
    assert r.returncode == 0, r.stderr
    res = _read(out)
    assert list(res.var_names) == ["AQP4", "GFAP", var_names[2], var_names[3]]

    summary = json.loads((out.with_name("canonicalization.json")).read_text())
    assert summary["var"]["renamed"] == 2 and summary["var"]["unmapped"] == 2


def test_canonicalize_no_symbol_source_errors(tmp_path):
    """Ensembl var_names with no column and no --symbol-map is a hard error (not a silent
    pass that would break every marker step downstream)."""
    import anndata as ad

    rng = np.random.default_rng(3)
    adata = ad.AnnData(rng.random((10, 3)).astype("float32"))
    adata.var_names = [f"ENSG{i:011d}" for i in range(3)]
    adata.obs["leiden"] = ["0"] * 10
    adata.obsm["X_pca"] = rng.standard_normal((10, 2)).astype("float32")
    src = tmp_path / "bare.h5ad"
    adata.write_h5ad(src)

    r = _run_canonicalize(src, tmp_path / "bare_cta.h5ad")
    assert r.returncode != 0
    assert "symbol" in r.stderr.lower()


def test_canonicalize_idempotent(ensembl_h5ad, tmp_path):
    """Canonicalizing the canonical sidecar again is a no-op: var_names stay symbols and
    counts survive (the second pass must not drop layers['counts'] when there's no .raw)."""
    src, _ = ensembl_h5ad
    out1 = tmp_path / "ens_cta.h5ad"
    assert _run_canonicalize(src, out1).returncode == 0
    first = _read(out1)

    out2 = tmp_path / "ens_cta2.h5ad"
    r = _run_canonicalize(out1, out2)
    assert r.returncode == 0, r.stderr
    second = _read(out2)

    assert list(second.var_names) == list(first.var_names)
    assert "counts" in second.layers
    summary = json.loads((out2.with_name("canonicalization.json")).read_text())
    assert summary["var"]["convention"] == "already symbols"


def test_canonicalize_writes_cp10k_layer(ensembl_h5ad, tmp_path):
    """The canonical sidecar carries a precomputed `lognorm_cp10k` layer = log1p(CP10k), so
    scale-sensitive classifiers (CellTypist) read their exact input with no re-normalization.
    X itself stays the user's own log-norm."""
    import scipy.sparse as sp

    src, _ = ensembl_h5ad
    out = tmp_path / "ens_cta.h5ad"
    assert _run_canonicalize(src, out).returncode == 0
    adata = _read(out)

    assert "lognorm_cp10k" in adata.layers
    cp = adata.layers["lognorm_cp10k"]
    cp = cp.toarray() if sp.issparse(cp) else np.asarray(cp)
    # expm1 of a genuine log1p(CP10k) sums to ~1e4 per cell
    per_cell = np.expm1(cp).sum(axis=1)
    assert np.allclose(per_cell, 1e4, rtol=1e-3), per_cell[:5]
