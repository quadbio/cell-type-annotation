"""Shared fixtures for the behavioral smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

# Each of the 3 synthetic cell types over-expresses a distinct 20-gene block, so markers
# / signatures are cleanly separable — enough to exercise real command bodies end to end.
N_GENES = 60
BLOCK = 20


@pytest.fixture
def tiny_h5ad(tmp_path):
    """Write a tiny log-normalized AnnData with 3 marker-separable clusters; return its path."""
    import anndata as ad
    import scanpy as sc

    rng = np.random.default_rng(0)
    blocks, labels = [], []
    for ti in range(3):
        base = np.full(N_GENES, 0.5)
        base[ti * BLOCK:(ti + 1) * BLOCK] = 5.0  # type-specific high-expression genes
        blocks.append(rng.poisson(np.tile(base, (80, 1))).astype("float32"))
        labels += [str(ti)] * 80
    adata = ad.AnnData(np.vstack(blocks))
    adata.var_names = [f"g{i}" for i in range(N_GENES)]
    adata.obs["leiden"] = labels
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)  # marker/scoring jobs expect log-normalized data in X

    h5 = tmp_path / "q.h5ad"
    adata.write_h5ad(h5)
    return h5


# Symbols for the Ensembl fixture; index 0 and 2 share a symbol on purpose, to exercise
# the duplicate-symbol -> var_names_make_unique path in `cta data canonicalize`.
ENS_SYMBOLS = ["AQP4", "GFAP", "AQP4", "MBP", "PLP1", "CD3D"]


@pytest.fixture
def ensembl_h5ad(tmp_path):
    """Ensembl-indexed AnnData (like a CELLxGENE atlas): a `feature_name` symbol column,
    integer counts hidden in `.raw`, log-norm in X, an embedding + cluster key. Returns
    (path, symbols)."""
    import anndata as ad
    import scanpy as sc

    rng = np.random.default_rng(1)
    n_obs, n_vars = 40, len(ENS_SYMBOLS)
    counts = rng.poisson(3.0, size=(n_obs, n_vars)).astype("float32") + 1.0
    adata = ad.AnnData(counts.copy())
    adata.var_names = [f"ENSG{i:011d}" for i in range(n_vars)]
    adata.var["feature_name"] = ENS_SYMBOLS
    adata.obs["leiden"] = [str(i % 2) for i in range(n_obs)]
    adata.obsm["X_pca"] = rng.standard_normal((n_obs, 5)).astype("float32")
    adata.raw = adata.copy()  # .raw holds the integer counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)  # X is now log-normalized; .raw stays counts

    h5 = tmp_path / "ens.h5ad"
    adata.write_h5ad(h5)
    return h5, list(ENS_SYMBOLS)
