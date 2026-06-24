"""Live integration test for the CELLxGENE Census reference route.

Marked `census` and EXCLUDED from the default suite (it opens the remote Census over the
network). Run on demand: `pixi run test-census`. It exercises the real
`cta reference fetch-census` path end to end — open the remote TileDB-SOMA store, read
matching obs, subsample via the shared stratified sampler, fetch the slice, write an .h5ad —
so it also gives the A4 subsample refactor real-route coverage, not just the unit test.
"""
from __future__ import annotations

import subprocess

import pytest


@pytest.mark.census
def test_fetch_census_end_to_end(tmp_path):
    out = tmp_path / "census_ref.h5ad"
    max_cells = 40
    r = subprocess.run(
        ["cta", "reference", "fetch-census",
         "--organism", "homo_sapiens",
         # narrow filter → small obs read, fast test; a small primary-data tissue
         "--obs-value-filter", "tissue_general=='tongue' and is_primary_data==True",
         "--max-cells", str(max_cells),
         "--out", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"fetch-census failed:\n{r.stderr}"
    assert out.exists(), f"reference .h5ad not written; stderr:\n{r.stderr}"

    import anndata as ad
    ref = ad.read_h5ad(out)
    assert 0 < ref.n_obs <= max_cells, f"expected 1..{max_cells} cells, got {ref.n_obs}"
    assert ref.n_vars > 0
    assert "cell_type" in ref.obs.columns

    # provenance is stamped and the moving 'stable' alias is resolved to a dated build
    prov = ref.uns["census_provenance"]
    assert prov["census_version_requested"] == "stable"
    assert prov["census_version"][:2] == "20", f"expected a dated build, got {prov['census_version']!r}"
    assert prov["n_fetched"] == ref.n_obs
