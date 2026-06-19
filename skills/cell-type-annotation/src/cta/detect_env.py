#!/usr/bin/env python
"""Detect the compute environment and suggest a profile for the annotation run.

This is a DETECTION-ONLY probe — it never submits jobs, never installs anything, and
never runs analysis. It reports what is reachable *from where this process runs* so the
skill can write a plan that fits the environment (see reference/compute_environments.md).

It distinguishes three profiles:
  - local_cpu : Apple-Silicon / CPU host (or MPS-only). No CUDA GPU reachable here.
  - local_gpu : an NVIDIA CUDA GPU is reachable IN THIS PROCESS (run heavy steps inline).
  - scheduler : a batch scheduler (e.g. sbatch on PATH) exists but no GPU is reachable
                here — i.e. likely a login/submit node whose GPUs live on compute nodes.
                Heavy GPU steps are written into the plan to be SUBMITTED elsewhere; this
                skill never submits them itself.

The scheduler-vs-cpu distinction can be genuinely ambiguous (a CPU-only host may also
have `sbatch`), so `profile` is a SUGGESTION — the skill confirms it with the user.

Stdlib-only at its core; torch / psutil / cupy are probed if present but never required.

Usage:
  python detect_env.py                 # human summary to stderr, JSON to stdout
  python detect_env.py --json-only     # JSON to stdout only
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys

import typer


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def _bytes_to_gb(n) -> float | None:
    try:
        return round(int(n) / (1024 ** 3), 1)
    except (TypeError, ValueError):
        return None


# --- accelerator -----------------------------------------------------------------------

def _detect_cuda() -> dict:
    """Is an NVIDIA CUDA GPU reachable *in this process*?

    Prefer torch (authoritative for what the analysis stack will actually use); fall back
    to nvidia-smi (catches GPUs when only a CPU-only torch build is installed). Note that
    `CUDA_VISIBLE_DEVICES` can hide/renumber devices for torch but NOT for nvidia-smi.
    """
    info = {"cuda": False, "n_gpus": 0, "gpu_names": [], "vram_gb": None, "source": None}

    try:
        import torch  # noqa
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            n = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(n)]
            vram = max(torch.cuda.get_device_properties(i).total_memory for i in range(n))
            info.update(cuda=True, n_gpus=n, gpu_names=names,
                        vram_gb=_bytes_to_gb(vram), source="torch")
            return info
    except Exception:  # noqa: BLE001 — torch absent / CPU build / driver mismatch
        pass

    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15,
            )
            lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
            if out.returncode == 0 and lines:
                names, vrams = [], []
                for ln in lines:
                    parts = [p.strip() for p in ln.split(",")]
                    names.append(parts[0])
                    if len(parts) > 1:
                        try:
                            vrams.append(int(float(parts[1])))  # MiB
                        except ValueError:
                            pass
                info.update(cuda=True, n_gpus=len(names), gpu_names=names,
                            vram_gb=(round(max(vrams) / 1024, 1) if vrams else None),
                            source="nvidia-smi")
                return info
        except Exception:  # noqa: BLE001
            pass
    return info


def _detect_mps() -> bool:
    """Apple-Silicon Metal (MPS) — usable by scvi-tools, but NOT by the CUDA-only stack
    (rapids-singlecell, faiss-gpu). Gate use on is_available(), not is_built()."""
    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        return False
    try:
        import torch  # noqa
        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        # Hardware is Apple-Silicon; MPS is available once a recent torch is installed.
        return True


# --- memory ----------------------------------------------------------------------------

def _cgroup_limit_bytes() -> int | None:
    """Container/cgroup memory ceiling, if any (cgroup v2 then v1). On HPC this is often
    the real cap, well below the node's physical RAM."""
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
        except OSError:
            continue
        if raw in ("max", ""):
            continue
        try:
            val = int(raw)
        except ValueError:
            continue
        # v1 reports a huge sentinel when unlimited — ignore implausibly large values.
        if 0 < val < (1 << 60):
            return val
    return None


def _slurm_mem_bytes() -> int | None:
    """SLURM allocation memory, if running inside an allocation. Values are in MB."""
    for var in ("SLURM_MEM_PER_NODE",):
        v = os.environ.get(var)
        if v:
            try:
                return int(v) * 1024 * 1024
            except ValueError:
                pass
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    n_cpu = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if per_cpu and n_cpu:
        try:
            return int(per_cpu) * int(n_cpu) * 1024 * 1024
        except ValueError:
            pass
    return None


def _detect_ram() -> dict:
    """Usable RAM = min(physical/available, cgroup cap, SLURM allocation). Reports the
    effective ceiling so the skill doesn't size a job to whole-node RAM it can't use."""
    phys = None
    try:
        import psutil  # noqa
        vm = psutil.virtual_memory()
        phys = int(vm.total)
    except Exception:  # noqa: BLE001
        try:
            phys = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            phys = None

    caps = [c for c in (phys, _cgroup_limit_bytes(), _slurm_mem_bytes()) if c]
    effective = min(caps) if caps else None
    return {
        "ram_gb": _bytes_to_gb(effective),
        "ram_physical_gb": _bytes_to_gb(phys),
        "ram_capped_by_cgroup_or_slurm": bool(
            effective is not None and phys is not None and effective < phys),
    }


# --- scheduler -------------------------------------------------------------------------

def _detect_scheduler() -> dict:
    """A batch scheduler on PATH means heavy GPU work can run on a compute node that this
    process can't see. We detect SLURM, PBS/Torque, LSF, SGE generically. Being INSIDE an
    allocation (e.g. SLURM_JOB_ID set) means we're already on a compute node, not a login
    node — so what's reachable here is authoritative and there's nothing to hand off."""
    submitters = {"slurm": "sbatch", "pbs": "qsub", "lsf": "bsub", "sge": "qsub"}
    found = {name: shutil.which(cmd) is not None for name, cmd in submitters.items()}
    present = sorted(n for n, ok in found.items() if ok)
    inside_job = any(os.environ.get(v) for v in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID",
                                                 "JOB_ID"))
    return {
        "scheduler": present[0] if present else None,
        "schedulers_present": present,
        "inside_allocation": inside_job,
    }


# --- profile suggestion ----------------------------------------------------------------

def _suggest_profile(cuda: dict, mps: bool, sched: dict) -> tuple[str, str]:
    if cuda["cuda"]:
        return "local_gpu", "an NVIDIA CUDA GPU is reachable in this process"
    if sched["scheduler"] and not sched["inside_allocation"]:
        return ("scheduler",
                f"no local GPU, but '{sched['scheduler']}' is on PATH and we are not "
                "inside an allocation — likely a login/submit node with GPUs on compute "
                "nodes (CONFIRM with the user; a CPU-only host can also have a scheduler)")
    if mps:
        return ("local_cpu",
                "Apple-Silicon MPS only — treated as CPU-tier for the CUDA-only stack, "
                "though scvi-tools can use MPS for a deeper local option")
    return "local_cpu", "no CUDA GPU and no batch scheduler reachable"


def cli(
    json_only: bool = typer.Option(
        False, help="print only the JSON object to stdout (no human summary)"),
) -> None:
    """Detect the compute environment and suggest a profile for the annotation run."""
    cuda = _detect_cuda()
    mps = _detect_mps()
    ram = _detect_ram()
    sched = _detect_scheduler()
    profile, why = _suggest_profile(cuda, mps, sched)

    accelerator = "cuda" if cuda["cuda"] else ("mps" if mps else "cpu")
    # execution_mode is a SUGGESTION: inline unless this looks like a submit node.
    execution_mode = "submit" if profile == "scheduler" else "inline"

    report = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "accelerator": accelerator,
        "cuda": cuda["cuda"],
        "n_gpus": cuda["n_gpus"],
        "gpu_names": cuda["gpu_names"],
        "vram_gb": cuda["vram_gb"],
        "cuda_source": cuda["source"],
        "mps": mps,
        **ram,
        **sched,
        "suggested_profile": profile,
        "suggested_execution_mode": execution_mode,
        "reason": why,
    }

    if not json_only:
        eprint(">> compute environment probe (detection only — nothing was run or submitted)")
        eprint(f"   platform     : {report['platform']} / {report['machine']} "
               f"(python {report['python']})")
        if cuda["cuda"]:
            eprint(f"   GPU (CUDA)   : {cuda['n_gpus']}x {', '.join(cuda['gpu_names']) or '?'}"
                   f"  vram~{cuda['vram_gb']}GB  (via {cuda['source']})")
        else:
            eprint(f"   GPU (CUDA)   : none reachable here"
                   f"{'  [MPS available]' if mps else ''}")
        cap = "  (capped by cgroup/SLURM)" if report["ram_capped_by_cgroup_or_slurm"] else ""
        eprint(f"   usable RAM   : ~{report['ram_gb']}GB{cap}")
        eprint(f"   scheduler    : {sched['scheduler'] or 'none'}"
               f"{'  (inside an allocation)' if sched['inside_allocation'] else ''}")
        eprint(f"   => suggested : profile='{profile}', execution_mode='{execution_mode}'")
        eprint(f"      because     {why}")
        eprint("   NOTE: 'scheduler' vs 'local_cpu' is ambiguous — confirm with the user.")

    print(json.dumps(report, indent=2))
