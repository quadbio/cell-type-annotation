"""Provision the pan-human Azimuth model for `cta crosscheck azimuth` — 84 MB, once.

panhumanpy downloads its weights lazily, with a bare `urllib.request.urlretrieve` fired from
INSIDE the first `AzimuthNN(...)` call. On a firewalled compute node that surfaces as a cryptic
URLError *after* the query h5ad has already been loaded. This command pulls the same file up
front, so the annotation run needs no network at all.

It deliberately does NOT import panhumanpy: the download belongs on an internet-connected
(login) node, where only the lean `default` pixi env is built — panhumanpy lives in the separate
`azimuth` env, which the guidance says to build where the work runs. The constants below
therefore MIRROR panhumanpy's internals; `tests/test_azimuth.py` asserts they still agree
whenever panhumanpy IS importable, so drift is caught rather than silently wrong.

Cost: 84 MB over the wire, 84 MB on disk, in `~/.cache/panhumanpy/<version>/inference_model/`.
That path is hardcoded in panhumanpy with no environment override — worth knowing on an HPC
system with a small home quota.

On a firewalled compute node ensure outbound internet first (e.g. load your site's proxy
module); an internet-connected node works directly.

Usage:
  cta reference fetch-azimuth              # v1 (default) -> ~/.cache/panhumanpy/v1/...
  cta crosscheck azimuth --query Q.h5ad --cluster-key leiden --workdir DIR
"""

from __future__ import annotations

from pathlib import Path

import typer

from cta.io import download_file, eprint

# --- mirrored from panhumanpy (ANNotate_tools.MODEL_URLS / CACHE_DIR, ANNotate.py) ---------
ZENODO_RECORD = "20401417"
MODEL_URLS = {
    "v0": f"https://zenodo.org/records/{ZENODO_RECORD}/files/panhumanpy_inference_model_v0.keras?download=1",
    "v1": f"https://zenodo.org/records/{ZENODO_RECORD}/files/panhumanpy_inference_model_v1.keras?download=1",
}
CACHE_DIR = Path.home() / ".cache" / "panhumanpy"
MODEL_FILENAME = "inference_model.keras"  # panhumanpy looks for exactly this name
DEFAULT_VERSION = "v1"
# A complete v1 model is ~84 MB; anything far smaller is a truncated/HTML error body, which
# would otherwise fail opaquely inside keras.models.load_model.
_MIN_BYTES = 10 * 1024 * 1024


def cli(
    version: str = typer.Option(
        DEFAULT_VERSION, help="model version to fetch: v1 (calibrated, has `Unassigned`) or v0"
    ),
    force: bool = typer.Option(False, help="re-download even if the cached file looks complete"),
) -> None:
    """Download the pan-human Azimuth model (84 MB) into panhumanpy's cache. Needs network."""
    if version not in MODEL_URLS:
        raise SystemExit(f"ERROR: --version must be one of {sorted(MODEL_URLS)}, got {version!r}")

    dest_dir = CACHE_DIR / version / "inference_model"
    dest = dest_dir / MODEL_FILENAME

    if dest.exists() and dest.stat().st_size >= _MIN_BYTES and not force:
        eprint(f">> already present: {dest} ({dest.stat().st_size / 1e6:.0f} MB). Nothing to do.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    eprint(f">> downloading pan-human Azimuth {version} (~84 MB, resumable) from Zenodo")
    download_file(MODEL_URLS[version], dest, what=f"pan-human Azimuth {version}")

    size = dest.stat().st_size if dest.exists() else 0
    if size < _MIN_BYTES:
        raise SystemExit(
            f"ERROR: downloaded file is only {size / 1e6:.1f} MB (expected ~84 MB) — probably a "
            f"truncated transfer or an error page. Delete {dest} and re-run."
        )
    eprint(f">> done. {dest} ({size / 1e6:.0f} MB)")
    eprint(">> panhumanpy will now find it without network access.")
