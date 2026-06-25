"""Behavioral tests for `cta data qc` — Stage-1 QC that makes low-quality / cycling clusters
detectable by writing standard metrics + a cell-cycle signal into the canonical sidecar's obs
(only when absent). Run through the installed `cta` console script, like the other smoke tests.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np

# A handful of real Tirosh S / G2M symbols (subset of the command's defaults) so the
# case-insensitive cell-cycle match has enough genes on the panel to score.
S_GENES = [
    "MCM5",
    "PCNA",
    "TYMS",
    "FEN1",
    "MCM2",
    "MCM4",
    "RRM1",
    "UNG",
    "GINS2",
    "MCM6",
    "CDCA7",
    "DTL",
]
G2M_GENES = [
    "HMGB2",
    "CDK1",
    "NUSAP1",
    "UBE2C",
    "BIRC5",
    "TPX2",
    "TOP2A",
    "NDC80",
    "CKS2",
    "NUF2",
    "CKS1B",
    "MKI67",
]
MT_GENES = ["MT-CO1", "MT-ND1", "MT-CYB"]


def _write_adata(path, *, with_cc=True, mito=True, n_fillers=53):
    """A small log-normalized AnnData with integer counts in layers['counts'], optional MT-
    genes and cell-cycle genes, plus filler genes so the cell-cycle control sampling works."""
    import anndata as ad
    import scanpy as sc

    genes = list(S_GENES + G2M_GENES) if with_cc else []
    if mito:
        genes += MT_GENES
    genes += [f"GENE{i}" for i in range(n_fillers)]

    rng = np.random.default_rng(0)
    n_obs = 60
    counts = rng.poisson(3.0, size=(n_obs, len(genes))).astype("float32") + 1.0
    adata = ad.AnnData(counts.copy())
    adata.var_names = genes
    adata.obs["leiden"] = [str(i % 3) for i in range(n_obs)]
    adata.layers["counts"] = counts.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)  # X log-normalized; layers['counts'] stays integer
    adata.write_h5ad(path)
    return path


def _run_qc(adata_path, *extra):
    return subprocess.run(
        ["cta", "data", "qc", "--adata", str(adata_path), *extra],
        capture_output=True,
        text=True,
    )


def _read(path):
    import anndata as ad

    return ad.read_h5ad(path)


def test_qc_writes_metrics_and_cell_cycle(tmp_path):
    """Depth/mito metrics + a cell-cycle signal land in obs, and a qc.json summary is written."""
    src = _write_adata(tmp_path / "q_cta.h5ad")
    r = _run_qc(src)
    assert r.returncode == 0, r.stderr
    adata = _read(src)

    for col in (
        "n_genes_by_counts",
        "total_counts",
        "pct_counts_mt",
        "S_score",
        "G2M_score",
        "phase",
    ):
        assert col in adata.obs.columns, f"missing {col}"
    # pct_counts_mt is a real percentage in (0, 100) given the MT- genes carry counts
    assert (adata.obs["pct_counts_mt"] > 0).all()

    summary = json.loads((src.with_name("qc.json")).read_text())
    assert "pct_counts_mt" in summary["written"]
    assert summary["cell_cycle"]["scored"] is True
    assert summary["n_mito_genes"] == len(MT_GENES)


def test_qc_does_not_overwrite_existing(tmp_path):
    """A pre-existing QC column is preserved by default; --recompute overwrites it."""
    import pandas as pd

    src = tmp_path / "q_cta.h5ad"
    _write_adata(src)
    # seed a sentinel pct_counts_mt the command must NOT clobber without --recompute
    adata = _read(src)
    adata.obs["pct_counts_mt"] = pd.Series(-1.0, index=adata.obs_names)
    adata.write_h5ad(src)

    assert _run_qc(src).returncode == 0
    kept = _read(src)
    assert (kept.obs["pct_counts_mt"] == -1.0).all()  # untouched
    summary = json.loads((src.with_name("qc.json")).read_text())
    assert "pct_counts_mt" in summary["skipped_existing"]

    # --recompute replaces the sentinel with real values
    assert _run_qc(src, "--recompute").returncode == 0
    recomputed = _read(src)
    assert (recomputed.obs["pct_counts_mt"] > 0).all()


def test_qc_skips_cell_cycle_when_genes_off_panel(tmp_path):
    """With no cell-cycle genes on the panel (targeted-panel case), the score is skipped
    (not faked) while depth/mito metrics are still computed."""
    src = _write_adata(tmp_path / "q_cta.h5ad", with_cc=False)
    r = _run_qc(src)
    assert r.returncode == 0, r.stderr
    adata = _read(src)

    assert "n_genes_by_counts" in adata.obs.columns  # depth metric still computed
    assert "phase" not in adata.obs.columns  # cell-cycle honestly skipped
    summary = json.loads((src.with_name("qc.json")).read_text())
    assert summary["cell_cycle"]["scored"] is False
