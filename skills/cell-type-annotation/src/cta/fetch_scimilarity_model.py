"""Fetch the SCimilarity model bundle for `cta crosscheck scimilarity` — annotation-only.

The pretrained model lives on Zenodo (record 10685499) as ONE ~30 GB tarball,
``model_v1.1.tar.gz``. There is no separately-hosted annotation-only artifact, so the
download is unavoidably the full tarball; but the part the annotation cross-check actually
loads is just the encoder + the ``annotation/`` kNN (~9 GB). The 32 GB ``cellsearch/``
TileDB is for cell *search*, never touched by annotation — so by default we extract
EVERYTHING EXCEPT ``cellsearch/`` and the on-disk footprint settles to ~9 GB, not ~41 GB.

Cost, honestly:
  * over the wire: ~30 GB (the whole tarball — Zenodo serves one file).
  * transient disk while downloading: ~30 GB for the tarball (deleted after extraction
    unless --keep-tarball), plus ~9 GB extracted → ~39 GB peak, settling to ~9 GB.
  * with --with-cellsearch: ~41 GB extracted (only if you also want cell SEARCH).

The download is resumable (re-run the command to continue a partial download). On a firewalled
compute node, make sure outbound internet is available first (e.g. load your site's proxy
module); an internet-connected (login) node works directly.

Usage:
  cta reference fetch-scimilarity --dest /path/to/models
  # -> downloads + extracts to /path/to/models/model_v1.1 (annotation-only, ~9 GB)
  # then:
  cta crosscheck scimilarity --query Q.h5ad --cluster-key leiden --workdir DIR \
      --model-path /path/to/models/model_v1.1
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import typer

from cta.io import download_file, eprint, have_tool

ZENODO_RECORD = "10685499"
MODEL_NAME = "model_v1.1"
MODEL_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{MODEL_NAME}.tar.gz"
# The files the annotation path actually loads — their presence marks a complete
# annotation-only extraction (the encoder, the model gene order, and the labelled kNN).
_ANNOTATION_MARKERS = ("encoder.ckpt", "gene_order.tsv", "annotation/labelled_kNN.bin")
_CELLSEARCH = "cellsearch"  # the 32 GB TileDB the annotation path never loads


def _have(tool: str) -> bool:
    return have_tool(tool)


def _download(url: str, tarball: Path) -> None:
    """Resumably download `url` to `tarball` (shared helper; see cta.io.download_file)."""
    eprint(f">> downloading {MODEL_NAME}.tar.gz (~30 GB, resumable) from Zenodo")
    download_file(url, tarball, what="model bundle")


def _extract(tarball: Path, dest: Path, with_cellsearch: bool) -> None:
    """Extract `tarball` into `dest`, skipping the cellsearch TileDB unless asked."""
    if not _have("tar"):
        raise SystemExit("ERROR: `tar` not found on PATH; cannot extract the model bundle.")
    cmd = ["tar", "-xzf", str(tarball), "-C", str(dest)]
    if not with_cellsearch:
        # Two patterns so the dir entry AND its children are skipped on both GNU and BSD
        # tar — keeps the on-disk model to the ~9 GB annotation-only subset.
        cmd += [f"--exclude={MODEL_NAME}/{_CELLSEARCH}", f"--exclude={MODEL_NAME}/{_CELLSEARCH}/*"]
    eprint(
        f">> extracting into {dest}"
        + ("" if with_cellsearch else f" (skipping {_CELLSEARCH}/ — annotation-only)")
    )
    eprint("   $ " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"ERROR: extraction failed (exit {e.returncode}). The tarball may be truncated; "
            "delete it (or re-run to resume the download) and try again."
        ) from e


def _is_complete(model_dir: Path, with_cellsearch: bool) -> bool:
    ok = all((model_dir / m).exists() for m in _ANNOTATION_MARKERS)
    return ok and ((model_dir / _CELLSEARCH).exists() if with_cellsearch else True)


def cli(
    dest: str = typer.Option(
        ...,
        help="directory to download + extract into; the model lands at "
        "<dest>/model_v1.1 (use that as `crosscheck scimilarity --model-path`)",
    ),
    with_cellsearch: bool = typer.Option(
        False,
        help="also extract the 32 GB cellsearch TileDB — only needed for cell "
        "SEARCH, NEVER for annotation; default skips it (~9 GB on disk, not ~41 GB)",
    ),
    keep_tarball: bool = typer.Option(
        False,
        help="keep the downloaded .tar.gz after extracting (default deletes it to "
        "reclaim ~30 GB; keep it if you might re-extract with --with-cellsearch later)",
    ),
    url: str = typer.Option(MODEL_URL, help="override the Zenodo download URL"),
) -> None:
    """Download the SCimilarity model bundle (annotation-only by default, ~9 GB on disk)."""
    dest_dir = Path(dest).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_dir = dest_dir / MODEL_NAME

    if _is_complete(model_dir, with_cellsearch):
        eprint(f">> model already present at {model_dir} — nothing to do.")
        eprint(f">> use:  cta crosscheck scimilarity ... --model-path {model_dir}")
        return

    tarball = dest_dir / f"{MODEL_NAME}.tar.gz"
    if not (keep_tarball and tarball.exists()):
        _download(url, tarball)
    _extract(tarball, dest_dir, with_cellsearch)

    if not _is_complete(model_dir, with_cellsearch):
        raise SystemExit(
            f"ERROR: extraction finished but {model_dir} is missing expected files "
            f"{list(_ANNOTATION_MARKERS)}. The tarball layout may have changed; inspect "
            f"{tarball} (kept) and report it."
        )

    if keep_tarball:
        eprint(f">> kept tarball at {tarball}")
    else:
        tarball.unlink(missing_ok=True)
        eprint(f">> removed {tarball.name} to reclaim space")

    eprint(
        f">> done. model at {model_dir}"
        + ("" if with_cellsearch else " (annotation-only — no cellsearch)")
    )
    eprint(f">> use:  cta crosscheck scimilarity ... --model-path {model_dir}")
