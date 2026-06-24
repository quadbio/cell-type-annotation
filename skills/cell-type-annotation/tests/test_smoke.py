"""Behavioral smoke tests: run real `cta` commands end to end on tiny synthetic data.

Complements the registry-sync test (which only checks wiring) by exercising actual Typer
command bodies + the shared cta.io helpers through the installed `cta` console script.
"""
from __future__ import annotations

import json
import subprocess


def test_markers_compute_end_to_end(tiny_h5ad, tmp_path):
    import pandas as pd

    workdir = tmp_path / "run"
    r = subprocess.run(
        ["cta", "markers", "compute", "--adata", str(tiny_h5ad), "--cluster-key", "leiden",
         "--workdir", str(workdir), "--level", "L1", "--no-cosg"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    out = workdir / "tables" / "markers_L1.csv"
    assert out.exists(), f"markers table not written; stderr:\n{r.stderr}"
    df = pd.read_csv(out)
    assert "cluster" in df.columns and len(df) > 0
    assert set(df["cluster"].astype(str)) == {"0", "1", "2"}


def test_inspect_writes_report_to_workdir(tiny_h5ad, tmp_path):
    """`cta data inspect --workdir` drops inspect_report.json (the file Claude authors
    context.md/run_state.json from) — guards the issue-#14 'No such option: --workdir' fix."""
    workdir = tmp_path / "run"
    r = subprocess.run(
        ["cta", "data", "inspect", "--adata", str(tiny_h5ad),
         "--cluster-key", "leiden", "--workdir", str(workdir)],
        capture_output=True, text=True,
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
    markers = {"TypeA": ["g0", "g1", "g2"], "TypeB": ["g20", "g21", "g22"],
               "TypeC": ["g40", "g41", "g42"]}
    markers_json = tmp_path / "markers_L1.json"
    markers_json.write_text(json.dumps(markers))

    r = subprocess.run(
        ["cta", "markers", "score", "--adata", str(tiny_h5ad), "--cluster-key", "leiden",
         "--markers", str(markers_json), "--workdir", str(workdir), "--level", "L1"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    z = workdir / "tables" / "scores_aucell_z_L1.csv"
    assert z.exists(), f"AUCell z-score table not written; stderr:\n{r.stderr}"
    df = pd.read_csv(z)
    assert len(df) > 0
