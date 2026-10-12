import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

#: Leaf -> (level 1, level 2). ``root_c`` is an only child at every level, so the "no siblings"
#: path is exercised.
HIERARCHY_SPEC = {
    "a1x": ("root_a", "mid_a1"),
    "a1y": ("root_a", "mid_a1"),
    "a2x": ("root_a", "mid_a2"),
    "b1x": ("root_b", "mid_b1"),
    "b2x": ("root_b", "mid_b2"),
    "c1x": ("root_c", "mid_c1"),
}

#: Gene -> per-leaf detection. ``mark_a`` separates root_a from its siblings; ``shared_a1_b1`` is
#: high in mid_a1 *and* mid_b1, so it only looks specific against a node's own siblings; ``flat``
#: is uninformative everywhere.
MARKER_SPEC = {
    "mark_a": {"a1x": 1.0, "a1y": 1.0, "a2x": 1.0, "b1x": 0.0, "b2x": 0.0, "c1x": 0.0},
    "mark_a1": {"a1x": 1.0, "a1y": 1.0, "a2x": 0.0, "b1x": 0.0, "b2x": 0.0, "c1x": 0.0},
    "shared_a1_b1": {"a1x": 1.0, "a1y": 1.0, "a2x": 0.0, "b1x": 1.0, "b2x": 0.0, "c1x": 0.0},
    "weak_a": {"a1x": 1.0, "a1y": 0.0, "a2x": 0.0, "b1x": 0.0, "b2x": 0.0, "c1x": 0.0},
    "flat": dict.fromkeys(HIERARCHY_SPEC, 1.0),
}

LEVELS = ["level_1", "level_2", "level_3"]


@pytest.fixture()
def adata_hierarchy():
    """A strictly nested three-level hierarchy with planted, deterministic detection.

    A gene is detected in every cell of a leaf or in none, so detection margins are exact.
    """
    rng = np.random.default_rng(0)
    leaves = [leaf for leaf in HIERARCHY_SPEC for _ in range(int(rng.poisson(20)) + 2)]
    obs = pd.DataFrame(
        {
            "level_1": [HIERARCHY_SPEC[x][0] for x in leaves],
            "level_2": [HIERARCHY_SPEC[x][1] for x in leaves],
            "level_3": leaves,
        }
    )
    obs.index = obs.index.astype(str)
    counts = np.array([[3.0 * MARKER_SPEC[g][leaf] for g in MARKER_SPEC] for leaf in leaves])
    adata = ad.AnnData(X=np.log1p(counts), obs=obs, var=pd.DataFrame(index=list(MARKER_SPEC)))
    adata.layers["counts"] = counts
    for column in LEVELS:
        adata.obs[column] = pd.Categorical(adata.obs[column])
    return adata
