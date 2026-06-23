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
