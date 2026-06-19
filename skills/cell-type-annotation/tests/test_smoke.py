"""Behavioral smoke test: run a real `cta` command end to end on tiny synthetic data.

Complements the registry-sync test (which only checks wiring) by exercising an actual
Typer command body + the shared cta.io helpers through the installed `cta` console script.
"""
from __future__ import annotations

import subprocess

import numpy as np


def test_markers_compute_end_to_end(tmp_path):
    import anndata as ad
    import scanpy as sc

    rng = np.random.default_rng(0)
    n_genes = 60
    blocks, labels = [], []
    for ti in range(3):
        base = np.full(n_genes, 0.5)
        base[ti * 20:(ti + 1) * 20] = 5.0  # type-specific high-expression genes
        blocks.append(rng.poisson(np.tile(base, (80, 1))).astype("float32"))
        labels += [str(ti)] * 80
    adata = ad.AnnData(np.vstack(blocks))
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["leiden"] = labels
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)  # compute-markers expects log-normalized data in X

    h5 = tmp_path / "q.h5ad"
    adata.write_h5ad(h5)
    workdir = tmp_path / "run"

    r = subprocess.run(
        ["cta", "markers", "compute", "--adata", str(h5), "--cluster-key", "leiden",
         "--workdir", str(workdir), "--level", "L1", "--no-cosg"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    import pandas as pd
    out = workdir / "tables" / "markers_L1.csv"
    assert out.exists(), f"markers table not written; stderr:\n{r.stderr}"
    df = pd.read_csv(out)
    assert "cluster" in df.columns and len(df) > 0
    assert set(df["cluster"].astype(str)) == {"0", "1", "2"}
