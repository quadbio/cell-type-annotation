"""Central scanpy <-> rapids-singlecell routing for the compute-profile split.

Every scVerse step that has a rapids-singlecell GPU equivalent (marker DE, sub-cluster Leiden,
…) routes through `run()` here instead of re-implementing the env check + host<->device transfer
+ CPU fallback. One place to reason about "CPU or GPU", so adding a future GPU-accelerable step
is a two-callable call, not a copy of the plumbing.

Routing rule (no user knob): use the GPU iff a CUDA GPU is reachable in this process AND
rapids-singlecell imports; otherwise scanpy on CPU. The GPU path moves the object's matrices to
the device for the op and ALWAYS restores them to host afterwards (so downstream CPU code and any
fallback see numpy/scipy), and falls back to CPU on ANY failure — the GPU is an accelerator, never
a correctness dependency.

Result equivalence is per-op and the caller declares it. `rank_genes_groups` is bit-identical
CPU/GPU (only float/tie noise); Leiden/UMAP use different backends and genuinely DIVERGE — pass
`equivalent=False` for those so the chosen engine is logged (the run records which one produced
the divergent output). See reference/compute_environments.md.
"""

from __future__ import annotations

from collections.abc import Callable

from cta.io import eprint, resolve_accelerator

_GPU_CACHE: bool | None = None


def gpu_available() -> bool:
    """True iff a CUDA GPU is reachable in this process AND rapids-singlecell imports.

    The single source of truth for the CPU/GPU split. Cached per process — the answer can't
    change mid-run. Importing rapids-singlecell is the real gate: `resolve_accelerator` only
    sees torch, but rapids-singlecell may be absent (CPU env) or fail to load (e.g. a
    CUDA-mismatched compiled kernel), and either way we must use scanpy.
    """
    global _GPU_CACHE
    if _GPU_CACHE is None:
        ok = False
        if resolve_accelerator("auto") == "cuda":
            try:
                import rapids_singlecell  # noqa: F401

                ok = True
            except Exception:  # noqa: BLE001 — any import/load failure -> use CPU
                ok = False
        _GPU_CACHE = ok
    return _GPU_CACHE


def run(
    adata,
    *,
    cpu: Callable,
    gpu: Callable,
    label: str,
    equivalent: bool = True,
) -> str:
    """Run `gpu(adata, rsc)` on a CUDA GPU, else `cpu(adata)`; return the engine used ('gpu'/'cpu').

    On a GPU profile, moves `adata`'s matrices to the device (rapids-singlecell `anndata_to_GPU`),
    runs `gpu`, then ALWAYS restores them to host (`anndata_to_CPU`) — even on failure — so the rest
    of the step and any fallback see CPU arrays. Any exception in the GPU path logs and falls back
    to `cpu`. `equivalent=False` marks ops whose GPU result differs from CPU (clustering/UMAP); it
    only annotates the log line. Both callables mutate `adata` in place; `gpu` additionally receives
    the imported `rapids_singlecell` module so it needn't import it itself.
    """
    if gpu_available():
        try:
            import rapids_singlecell as rsc

            note = "" if equivalent else "  (result differs from CPU — distinct GPU backend)"
            eprint(f">> {label} [GPU / rapids-singlecell]{note}")
            rsc.get.anndata_to_GPU(adata)
            try:
                gpu(adata, rsc)
                return "gpu"
            finally:
                # Restore X to host even on failure, so the CPU path / downstream see numpy/scipy.
                try:
                    rsc.get.anndata_to_CPU(adata)
                except Exception:  # noqa: BLE001
                    pass
        except Exception as e:  # noqa: BLE001 — any GPU failure falls back to scanpy CPU
            eprint(f"   GPU {label} failed ({type(e).__name__}: {e}) — falling back to scanpy CPU")
    eprint(f">> {label} [CPU / scanpy]")
    cpu(adata)
    return "cpu"
