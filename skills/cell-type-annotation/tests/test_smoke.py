"""Behavioral smoke tests: run real `cta` commands end to end on tiny synthetic data.

Complements the registry-sync test (which only checks wiring) by exercising actual Typer
command bodies + the shared cta.io helpers through the installed `cta` console script.
"""

from __future__ import annotations

import json
import subprocess


def test_celltypist_input_rebuilds_cp10k():
    """`_celltypist_input` must hand CellTypist log1p(CP10k) regardless of the object's own
    normalization — reading the precomputed canonical layer, and rebuilding from a counts
    layer or a non-CP10k log-norm X when that layer is absent."""
    import anndata as ad
    import numpy as np
    import scanpy as sc

    from cta.io import cp10k_log1p_matrix
    from cta.run_celltypist import _celltypist_input

    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0, size=(40, 30)).astype("float32")
    counts[:, 0] = 50  # ensure non-trivial per-cell totals

    # X deliberately normalized to a NON-CP10k target (5000) → CellTypist would warn.
    a = ad.AnnData(counts.copy())
    a.layers["counts"] = counts.copy()
    sc.pp.normalize_total(a, target_sum=5000)
    sc.pp.log1p(a)

    def per_cell_total(adata):
        return np.expm1(np.asarray(adata.X)).sum(axis=1).ravel()

    # canonical fast path: a precomputed lognorm_cp10k layer is used as-is (no recompute)
    a_canon = a.copy()
    cp, _ = cp10k_log1p_matrix(counts=counts)
    a_canon.layers["lognorm_cp10k"] = cp
    out_canon = _celltypist_input(a_canon, None, None)
    assert np.allclose(per_cell_total(out_canon), 1e4, rtol=1e-3)
    assert np.allclose(np.asarray(out_canon.X), np.asarray(cp))  # read directly, not rebuilt

    # rebuild from the counts layer (auto-detected, no --counts-layer passed, no cp10k layer)
    out_counts = _celltypist_input(a, None, None)
    assert np.allclose(per_cell_total(out_counts), 1e4, rtol=1e-3)

    # counts-absent: re-target the (5000-normalized) log-norm X to CP10k
    a_nolayer = ad.AnnData(a.X.copy())  # X is the 5000-target log-norm, no counts layer
    out_lognorm = _celltypist_input(a_nolayer, None, None)
    assert np.allclose(per_cell_total(out_lognorm), 1e4, rtol=1e-3)


def test_markers_compute_end_to_end(tiny_h5ad, tmp_path):
    import pandas as pd

    workdir = tmp_path / "run"
    r = subprocess.run(
        [
            "cta",
            "markers",
            "compute",
            "--adata",
            str(tiny_h5ad),
            "--cluster-key",
            "leiden",
            "--workdir",
            str(workdir),
            "--level",
            "L1",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr

    out = workdir / "tables" / "markers_L1.csv"
    assert out.exists(), f"markers table not written; stderr:\n{r.stderr}"
    df = pd.read_csv(out)
    assert "cluster" in df.columns and len(df) > 0
    assert set(df["cluster"].astype(str)) == {"0", "1", "2"}

    # filter_rank_genes_groups specificity table (replaces the old COSG table)
    filt = workdir / "tables" / "markers_filtered_L1.csv"
    assert filt.exists(), f"filtered marker table not written; stderr:\n{r.stderr}"
    fdf = pd.read_csv(filt)
    assert {"cluster", "rank", "gene", "logfoldchange"}.issubset(fdf.columns)


def test_inspect_writes_report_to_workdir(tiny_h5ad, tmp_path):
    """`cta data inspect --workdir` drops inspect_report.json (the file Claude authors
    context.md/run_state.json from) — guards the issue-#14 'No such option: --workdir' fix."""
    workdir = tmp_path / "run"
    r = subprocess.run(
        [
            "cta",
            "data",
            "inspect",
            "--adata",
            str(tiny_h5ad),
            "--cluster-key",
            "leiden",
            "--workdir",
            str(workdir),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    report = workdir / "inspect_report.json"
    assert report.exists(), f"inspect_report.json not written; stderr:\n{r.stderr}"
    data = json.loads(report.read_text())
    assert data["n_obs"] == 240 and data["n_vars"] == 60
    assert data["n_clusters"] == 3


def test_markers_score_end_to_end(tiny_h5ad, tmp_path):
    """Exercise `cta markers score` incl. the AUCell z-score path (regressions: that
    branch once referenced an undefined `args.level` and crashed)."""
    import pandas as pd

    workdir = tmp_path / "run"
    # one curated signature per cluster, each with >= MIN_ON_PANEL_MARKERS on-panel genes
    markers = {
        "TypeA": ["g0", "g1", "g2"],
        "TypeB": ["g20", "g21", "g22"],
        "TypeC": ["g40", "g41", "g42"],
    }
    markers_json = tmp_path / "markers_L1.json"
    markers_json.write_text(json.dumps(markers))

    r = subprocess.run(
        [
            "cta",
            "markers",
            "score",
            "--adata",
            str(tiny_h5ad),
            "--cluster-key",
            "leiden",
            "--markers",
            str(markers_json),
            "--workdir",
            str(workdir),
            "--level",
            "L1",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr

    z = workdir / "tables" / "scores_aucell_z_L1.csv"
    assert z.exists(), f"AUCell z-score table not written; stderr:\n{r.stderr}"
    df = pd.read_csv(z)
    assert len(df) > 0
