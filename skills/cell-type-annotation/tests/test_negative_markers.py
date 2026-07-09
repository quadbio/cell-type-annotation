"""Negative-marker evidence: sign tracking, violation flagging, off-panel handling, cards.

Covers the first-class negative-marker path added across `cta markers profile` and
`cta report scaffold-cards`: a negative (expected-absent) marker that IS expressed in a
cluster is a violation — concrete evidence against that cluster's candidate call.
"""

from __future__ import annotations

import json
import subprocess


def test_load_panel_tracks_sign(tmp_path):
    """Signed marker JSON must preserve the pos/neg role per gene per group — not silently
    concatenate the two (the pre-existing bug this feature fixes). A flat list is all-positive,
    and a gene may be negative in one group yet positive in another."""
    from cta.profile_markers import _load_panel

    j = tmp_path / "m.json"
    j.write_text(
        json.dumps(
            {
                "T cell": {"pos": ["CD3D", "CD3E"], "neg": ["MS4A1"]},
                "B cell": ["MS4A1", "CD79A"],  # flat form → all positive
            }
        )
    )
    panel, neg = _load_panel(str(j), None, None)
    assert panel["T cell"] == ["CD3D", "CD3E", "MS4A1"]  # positives first, then negatives
    assert neg["T cell"] == {"MS4A1"}
    assert panel["B cell"] == ["MS4A1", "CD79A"]
    assert neg["B cell"] == set()  # flat form carries no negatives


def _write_controlled_adata(path):
    """A 2-cluster object with deterministic expression so pct thresholds are unambiguous:
    MS4A1 is ON in every cluster-0 cell and OFF in every cluster-1 cell."""
    import anndata as ad
    import numpy as np

    var = ["CD3D", "MS4A1", "FOO", "BAR"]
    n = 40  # 20 per cluster
    X = np.zeros((n, len(var)), dtype="float32")
    labels = []
    for i in range(n):
        c0 = i < 20
        labels.append("0" if c0 else "1")
        X[i, 0] = 2.0 if c0 else 0.0  # CD3D: positive marker, on in cluster 0
        X[i, 1] = 1.5 if c0 else 0.0  # MS4A1: negative marker, on in cluster 0 (a VIOLATION there)
        X[i, 2] = 0.7  # FOO everywhere
        X[i, 3] = 0.7 if not c0 else 0.0
    a = ad.AnnData(X)
    a.var_names = var
    a.obs["leiden"] = labels
    a.write_h5ad(path)


def test_profile_flags_negative_marker_violation(tmp_path):
    """`cta markers profile` on a panel with negatives writes panel_sign + panel_neg_flags:
    the negative marker (MS4A1) is flagged a violation where it is expressed (cluster 0) and
    NOT where it is off (cluster 1); an off-panel negative is reported, never scored."""
    import pandas as pd

    h5 = tmp_path / "q.h5ad"
    _write_controlled_adata(h5)
    workdir = tmp_path / "run"
    markers = {"T cell": {"pos": ["CD3D"], "neg": ["MS4A1", "OFFPANEL_X"]}}
    mj = tmp_path / "markers_L1.json"
    mj.write_text(json.dumps(markers))

    r = subprocess.run(
        [
            "cta",
            "markers",
            "profile",
            "--adata",
            str(h5),
            "--cluster-key",
            "leiden",
            "--markers",
            str(mj),
            "--workdir",
            str(workdir),
            "--level",
            "L1",
            "--dotplot",  # exercise the pos/neg-split dot plot path too
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    # the dot plot (with its negative "(−)" block) must render, not silently warn-and-skip
    assert (workdir / "plots" / "dotplot_L1.png").exists(), (
        f"dot plot not written; stderr:\n{r.stderr}"
    )
    assert "dot plot failed" not in r.stderr

    # sign map: MS4A1 negative, CD3D positive
    sign = pd.read_csv(workdir / "tables" / "panel_sign_L1.csv")
    sign_of = dict(zip(sign["gene"], sign["sign"], strict=True))
    assert sign_of["MS4A1"] == "neg" and sign_of["CD3D"] == "pos"

    # violations: cluster 0 (MS4A1 on) True, cluster 1 (MS4A1 off) False
    nf = pd.read_csv(workdir / "tables" / "panel_neg_flags_L1.csv")
    nf["cluster"] = nf["cluster"].astype(str)
    row0 = nf[(nf["cluster"] == "0") & (nf["gene"] == "MS4A1")].iloc[0]
    row1 = nf[(nf["cluster"] == "1") & (nf["gene"] == "MS4A1")].iloc[0]
    assert bool(row0["violation"]) is True and float(row0["pct"]) > 0.5
    assert bool(row1["violation"]) is False and float(row1["pct"]) == 0.0

    # an off-panel negative is UNMEASURED — excluded from the flags, and reported on stderr
    assert "OFFPANEL_X" not in set(nf["gene"])
    assert "off-panel NEGATIVE" in r.stderr


def test_negative_role_preserved_when_gene_is_positive_elsewhere(tmp_path):
    """A gene that is a POSITIVE marker of one type and a NEGATIVE marker of another must keep
    BOTH roles — the common case (a lineage's marker is its sibling's anti-marker). Here CD3D is
    positive for 'Tcell' (listed first) and negative for 'Bcell'; the violation check for Bcell
    must still fire on the cluster expressing CD3D, and panel_sign must record both roles."""
    import pandas as pd

    h5 = tmp_path / "q.h5ad"
    _write_controlled_adata(h5)  # CD3D on in cluster 0, off in cluster 1
    workdir = tmp_path / "run"
    markers = {
        "Tcell": {"pos": ["CD3D"], "neg": []},
        "Bcell": {"pos": ["FOO"], "neg": ["CD3D"]},  # CD3D is Bcell's expected-absent marker
    }
    mj = tmp_path / "markers_L1.json"
    mj.write_text(json.dumps(markers))

    r = subprocess.run(
        [
            "cta",
            "markers",
            "profile",
            "--adata",
            str(h5),
            "--cluster-key",
            "leiden",
            "--markers",
            str(mj),
            "--workdir",
            str(workdir),
            "--level",
            "L1",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr

    # panel_sign records CD3D BOTH as pos for Tcell AND neg for Bcell
    sign = pd.read_csv(workdir / "tables" / "panel_sign_L1.csv")
    roles = {(row.group, row.gene): row.sign for row in sign.itertuples()}
    assert roles[("Tcell", "CD3D")] == "pos"
    assert roles[("Bcell", "CD3D")] == "neg"

    # the Bcell negative check for CD3D fires on cluster 0 (CD3D on) and clears cluster 1 (off) —
    # this is exactly what the old per-gene collapse dropped
    nf = pd.read_csv(workdir / "tables" / "panel_neg_flags_L1.csv")
    nf["cluster"] = nf["cluster"].astype(str)
    bcell = nf[(nf["group"] == "Bcell") & (nf["gene"] == "CD3D")]
    assert not bcell.empty, (
        "Bcell's negative marker CD3D was dropped (per-gene collapse regression)"
    )
    v = dict(zip(bcell["cluster"], bcell["violation"].astype(bool), strict=True))
    assert v["0"] is True and v["1"] is False


def test_negative_marker_section_flags_and_clears():
    """The card helper renders a ⚠ on a violation and the 'confirmed off' footer when clean.
    Tolerates the CSV-round-tripped boolean the table stores ('True'/'False' or bool)."""
    import pandas as pd

    from cta.scaffold_cards import _negative_marker_section

    viol = pd.DataFrame(
        [
            {"cluster": "0", "gene": "MS4A1", "group": "T cell", "pct": 0.8, "violation": "True"},
            {"cluster": "0", "gene": "CD14", "group": "T cell", "pct": 0.02, "violation": "False"},
        ]
    )
    out = _negative_marker_section(viol)
    assert "⚠" in out and "MS4A1" in out and "AGAINST" in out
    assert "confirmed off" not in out  # a violation is present

    clean = pd.DataFrame(
        [{"cluster": "1", "gene": "MS4A1", "group": "T cell", "pct": 0.0, "violation": False}]
    )
    out2 = _negative_marker_section(clean)
    assert "⚠" not in out2 and "confirmed off" in out2
