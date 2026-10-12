import pandas as pd
import pytest

from cell_type_annotation.report import LABEL_COLUMNS, check_level, prediction_summary, render_level


@pytest.fixture()
def labels():
    return pd.DataFrame(
        [
            ("0", "T cell", "normal", "CL:0000084", 0.9, "assigned", "CD3E/CD3D high.", "NK: no NKG7"),
            ("1", "B cell", "normal", "CL:0000236", 0.8, "assigned", "MS4A1, CD79A.", ""),
            ("2", "Unknown", "low-quality", "", 0.3, "unknown", "Low depth, high mito.", ""),
        ],
        columns=LABEL_COLUMNS,
    )


CLUSTERS = ["0", "1", "2"]
SIZES = {"0": 500, "1": 300, "2": 20}


def test_a_complete_level_has_no_problems(labels):
    assert check_level(labels, CLUSTERS) == []


def test_omitted_and_unexpected_clusters_are_problems(labels):
    problems = check_level(labels.iloc[:2], CLUSTERS + ["3"])
    assert any("without a row: 2, 3" in p for p in problems)
    assert any("not in this level: 2" in p for p in check_level(labels, ["0", "1"]))
    assert check_level(labels.iloc[:0], CLUSTERS)  # an empty table is never complete


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("rationale", "", "without a rationale"),
        ("label", None, "without a label"),
        ("status", "done", "status must be one of"),
        ("confidence", 1.5, "confidence must be"),
        ("cl_id", "CL:12", "not of the form"),
        ("status", "refuted", "refuted call(s) still open"),
    ],
)
def test_check_level_finds_problems(labels, column, value, message):
    labels.loc[0, column] = value
    assert any(message in p for p in check_level(labels, CLUSTERS))


def test_duplicate_clusters_are_problems(labels):
    duplicated = pd.concat([labels, labels.iloc[[0]].assign(label="NK cell")], ignore_index=True)
    assert any("more than once: 0" in p for p in check_level(duplicated, CLUSTERS))


def test_missing_columns(labels):
    assert "missing columns" in check_level(labels.drop(columns="rejected"), CLUSTERS)[0]


def test_render_level(labels):
    evidence = {
        "Top markers": pd.DataFrame(
            {"cluster": ["0", "0", "1"], "gene": ["CD3E", "CD3D", "MS4A1"], "lfc": [3.1, 2.9, 4]}
        )
    }
    text = render_level("L1", labels, evidence=evidence, sizes=SIZES, figures=["umap_L1.png"])

    assert text.startswith("# Level L1")
    assert "820 cells" in text and "1 flagged" in text
    assert text.index("### Cluster 2 (flagged)") < text.index("### Cluster 0")  # flagged first
    assert "| CD3E | 3.1 |" in text and "**Rejected.** NK: no NKG7" in text
    assert "Complete:" in text


def test_prediction_summary():
    summary = prediction_summary(["a", "a", "a", "b"], ["T", "T", "NK", "B"], [0.9, 0.7, 0.2, 1.0])
    row = summary.set_index("cluster").loc["a"]
    assert (row["top1"], row["frac1"], row["top2"], row["frac2"]) == ("T", 0.667, "NK", 0.333)
    assert row["mean_conf"] == 0.6
    assert summary.set_index("cluster").loc["b", "top2"] == ""
    assert prediction_summary(["a"], ["T"])["mean_conf"].isna().all()


def test_render_lists_problems(labels):
    labels.loc[1, "rationale"] = ""
    text = render_level("L1", labels, sizes=SIZES)
    assert "without a rationale: 1" in text
    assert "_missing_" in text
