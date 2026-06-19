"""cta — the cell-type-annotation skill toolkit.

Evidence-driven, compute-environment-aware cell-type annotation of pre-clustered
single-cell / spatial transcriptomics data. The individual tools live in submodules
(each owning its own argparse `main()`); `cta.cli` exposes them as the `cta` command.

Kept import-light on purpose: importing `cta` (or running `cta detect-env`) must NOT pull
scanpy/torch/etc. Heavy deps are imported inside the submodule that needs them.
"""

__version__ = "0.1.0"
