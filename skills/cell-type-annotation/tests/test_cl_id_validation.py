"""cl_id syntax validation at the write-back gate.

`cta report write-back` must block a malformed ontology accession before it lands in the
user's AnnData, with an explicit --allow-bad-cl-id override — mirroring the existing
--allow-non-nested gate. (Semantic existence/obsolescence is checked in-chat via the
knowledgebase MCP, not here.)
"""

from __future__ import annotations

import subprocess


def test_validate_cl_ids_accept_reject():
    from cta.write_back import validate_cl_ids

    # (level, cluster, label, cl_id)
    good = [
        ("L1", "0", "T cell", "CL:0000084"),
        ("L1", "1", "neuron", "PCL:0000123"),
        ("L1", "2", "Unknown", ""),  # blank is fine — cl_id is optional
        ("L1", "3", "low-q", "NA"),  # blank sentinel, exempt
    ]
    assert validate_cl_ids(good) == []

    bad = [
        ("L1", "0", "T cell", "CL:000084"),  # too few digits
        ("L1", "1", "B cell", "cl:0000236"),  # wrong casing
        ("L1", "2", "mono", "CL:12345678"),  # too many digits
        ("L1", "3", "astro", "just some text"),  # free text
    ]
    viols = validate_cl_ids(bad)
    assert len(viols) == 4
    assert all("well-formed" in v for v in viols)


def _write_labels(workdir, rows):
    labels_dir = workdir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    lines = ["cluster,label,cl_id"] + rows
    (labels_dir / "annotation_L1.csv").write_text("\n".join(lines) + "\n")


def test_write_back_blocks_bad_cl_id_then_passes(tiny_h5ad, tmp_path):
    workdir = tmp_path / "run"
    # cluster 2 carries a malformed accession
    _write_labels(
        workdir,
        ["0,T cell,CL:0000084", "1,B cell,CL:0000236", "2,Monocyte,BADID"],
    )
    base = [
        "cta", "report", "write-back",
        "--adata", str(tiny_h5ad),
        "--workdir", str(workdir),
        "--cluster-key", "leiden",
        "--dry-run",
    ]

    blocked = subprocess.run(base, capture_output=True, text=True)
    assert blocked.returncode != 0, "expected write-back to block on a malformed cl_id"
    assert "cl_id" in blocked.stderr and "BADID" in blocked.stderr

    overridden = subprocess.run(base + ["--allow-bad-cl-id"], capture_output=True, text=True)
    assert overridden.returncode == 0, overridden.stderr
    assert "writing anyway" in overridden.stderr

    # fix the id → passes cleanly without the override
    _write_labels(
        workdir,
        ["0,T cell,CL:0000084", "1,B cell,CL:0000236", "2,Monocyte,CL:0000576"],
    )
    fixed = subprocess.run(base, capture_output=True, text=True)
    assert fixed.returncode == 0, fixed.stderr
    assert "cl_id syntax OK" in fixed.stderr
