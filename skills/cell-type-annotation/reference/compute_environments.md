# Reference: Compute environments — detect, then plan accordingly

This skill is **compute-environment aware**. It runs unchanged on a laptop, and on a
GPU/large-memory host it additionally downloads reference atlases, trains/refines a
reference model, and maps the query onto it. The environment is **detected early (Stage 1)
and confirmed with the user**, and the plan (Stage 2) and cross-check choices (Stage 3)
branch on it. **Detection only — this skill never submits jobs and never installs the GPU
stack behind the user's back.**

The design that makes this clean already exists: the skill is **artifact-centric**. Every
heavy step is a standalone script that reads/writes durable files and holds no in-memory
state. So the *same* script runs inline (via Bash) on a host that has the hardware, or is
**submitted to run elsewhere** by whatever mechanism the user's environment provides —
with no change to the script. This skill's job is to pick the right tools and write the
plan; *how* a step is executed in a given environment is a separate concern.

## The three profiles

Detected by `cta detect-env` (suggested) and confirmed with the user.

| Profile | What it means | Heavy-step execution | What it unlocks beyond the CPU default |
| --- | --- | --- | --- |
| `local_cpu` | CPU / Apple-Silicon; no CUDA GPU reachable here | inline (Bash) | nothing — **identical to the original skill** |
| `local_gpu` | an NVIDIA CUDA GPU is reachable *in this process* | inline (Bash) on the GPU | reference-model train + scArches mapping; GPU kNN; large atlas / Census; optional foundation-model cross-check |
| `scheduler` | a batch scheduler exists but no GPU is reachable here (e.g. a login/submit node) | **handed off** for submission elsewhere; light steps still inline | same capability set as `local_gpu`, but GPU-heavy scripts are written into the plan to be submitted, not run inline |

Notes:
- **Apple-Silicon (MPS)** is a `local_cpu` host for the CUDA-only tools (cuml/cudf, the GPU
  faiss build don't use MPS). The one exception is **scvi-tools**, which *can* use MPS — so a
  Mac can still train/refine a small reference model and run scArches mapping as a deeper
  local option, just slower than CUDA. Offer it; don't assume it.
- The `local_gpu` vs `scheduler` split is about **where the GPU is reachable from**, not how
  big it is. `ram_gb` is a separate axis (below).

## Where to build the env (the login-node footgun)

The env is built with **pixi** (Stage 1c): `pixi.toml` defines a `default` (CPU) and a `gpu`
environment, and a committed **`pixi.lock`** pins both. The trap on a scheduler cluster: a
login node has **no GPU**, but `pixi install -e gpu` will still download multi-GB CUDA wheels
there — wasting quota on packages that can't run. pixi does **not** auto-detect a GPU, so the
discipline is **build the env where the work runs** — choose the env to match the host (and
always pass **`--frozen`**, next section):

- **`local_cpu` / `local_gpu`** — one host does everything: `pixi install --frozen` (CPU) or
  `pixi install --frozen -e gpu` (GPU) on that host.
- **`scheduler`** — two places:
  1. On the **login node**, `pixi install --frozen` (the `default` CPU env). That is all
     detection and the light inline steps need. Do **not** `pixi install -e gpu` here.
  2. Inside a **GPU job**, `pixi install --frozen -e gpu` to build the GPU env on the GPU node.

On a cluster, keep pixi's envs off quota-limited home and on a **shared scratch** filesystem
(so the login and compute nodes see the same env and the large CUDA env fits):
`pixi config set detached-environments <scratch-path>`. The GPU env pulls cuML/cuDF from the
NVIDIA index (declared in `pixi.toml` `[feature.gpu.pypi-options]`); `pixi.lock` pins
everything for reproducibility.

### Build once, run frozen — the scheduler / HPC reality

The committed **`pixi.lock`** means the env never needs to be *solved* at runtime: always pass
**`--frozen`** (`pixi install --frozen` to build, `pixi run --frozen cta …` to run) so pixi uses
the lockfile as-is and never re-resolves. This matters most on a **scheduler / HPC cluster**,
where plain `pixi run` does *solve + validate + exclusive-lock on one shared prefix* on every
call. Four consequences, all generic to "memory-capped head node + one shared env prefix":

- **Don't solve on a capped head node.** A runtime solve can OOM/SIGSEGV under a head/login
  node's memory cap. With the committed lock, `pixi install --frozen` is a *link*, not a solve
  (peaks at tens of MB, not GB) — safe even on a login node, and `--frozen` never solves later.
- **One shared prefix is locked per `pixi run`.** Each call takes an exclusive lock on the env
  prefix, so concurrent runs (and any stray `pixi install`/`pytest` in the same checkout)
  serialize — even `cta … --help` can queue behind a heavy lock-holder. For concurrent/batch
  use, resolve the prefix once and call the binary directly: `CTA=$(pixi run --frozen which cta)`
  (or `-e gpu`), then `"$CTA" <command> …`.
- **Don't mutate shared env state mid-run.** No `pixi config` changes, no `pixi install`/`pytest`
  against a checkout with live runs — flipping a config key re-points concurrent runs and
  re-triggers solves. Do dependency edits and testing in a separate clone.
- **With `detached-environments`, clear stale local envs.** pixi ignores a leftover local
  `.pixi/envs` but **re-warns every call** — delete it to silence the noise.

## Detection idioms (what `cta detect-env` checks)

- **CUDA GPU:** `torch.cuda.is_available()` + `device_count()` + `get_device_name` +
  `get_device_properties(i).total_memory` (VRAM). Fall back to parsing `nvidia-smi
  --query-gpu=...`. Pitfalls: a CPU-only torch build reports no GPU even when one exists;
  `CUDA_VISIBLE_DEVICES` hides/renumbers devices for torch (but not for nvidia-smi).
- **MPS:** `torch.backends.mps.is_available()` (gate on this, not `is_built()`), and
  `platform.system()=="Darwin" and platform.machine()=="arm64"`.
- **Usable RAM:** `psutil.virtual_memory()` clamped to the cgroup cap
  (`/sys/fs/cgroup/memory.max`) and any scheduler allocation — the node's physical RAM is
  not what you can actually use inside a container or an allocation.
- **Scheduler:** a submitter on `PATH` (`sbatch`/`qsub`/`bsub`) means heavy work can run on
  a compute node this process can't see. Being **inside** an allocation (e.g. a job-id env
  var is set) means you're already on a compute node — so what's reachable here is
  authoritative and there's nothing to hand off.

The decisive ambiguity detection can't resolve on its own: a host with a scheduler on PATH
but no local GPU might be a **login node whose GPUs live on compute nodes**, *or* a plain
CPU host that merely has the client tools installed. `cta detect-env` flags it; the skill
**asks the user** (Stage 1) which it is before committing the plan.

## The memory axis

`ram_gb` gates RAM-hungry options independently of the GPU question:
- **Large reference atlases / large Census slices** — lifting the laptop-scale cap in
  `cta reference fetch-census --max-cells` (and downloading whole curated atlases) only makes
  sense when RAM allows loading them.
- **SCimilarity** (`cta crosscheck scimilarity`) is lighter than once assumed: ~16–27 GB RAM
  (measured), CPU, a few minutes, and only the ~9 GB annotation-only bundle on disk (encoder +
  annotation kNN; the full ~41 GB download incl. the cell-search TileDB is NOT needed). Fits a
  32 GB machine — optional, and HUMAN-only (skip for non-human queries). Provision the bundle
  once with `cta reference fetch-scimilarity --dest DIR` (resumable ~30 GB download, extracts
  annotation-only by default; on a compute node `module load eth_proxy` first).

## Which tool at which step (and what it needs)

The per-tool step / GPU-requirement / profile mapping has **one home**:
`reference/tool_registry.md`. Read it to decide what to run where. In short: most of the
pipeline is CPU (`none`); only the reference-mapping cross-check and (optionally)
large-query scoring touch a GPU. `cta crosscheck scimilarity` is CPU too and needs only moderate
RAM (~16–27 GB), so it runs on any reasonably-sized machine — it is HUMAN-only, not GPU- or
large-memory-gated. The profile decides *how* the `preferred`/`required` rows execute — inline on
`local_gpu`, handed off on `scheduler` (next section).

## The scheduler-handoff convention (stay fully generic)

When the confirmed profile is `scheduler`, the GPU-heavy scripts above run **somewhere this
process can't reach directly**. This skill therefore:

1. **Writes the standalone command** for each heavy step into `plan.md` (and narrates it in
   chat) — the exact `pixi run --frozen -e gpu cta <command> …` invocation, plus its inputs/outputs
   in the working dir — so it can be executed by whatever job-submission mechanism the
   user's environment provides.
2. **Marks those steps "submit, don't run inline"** and pauses for the user to run them via
   their own mechanism, then resumes from the durable artifacts (the run is resumable —
   `run_state.json`).
3. **Never embeds a submission command.** No `sbatch`/`qsub`/`bsub`, no resource flags, no
   assumptions about a specific cluster, scheduler, module system, or companion skill.
   Composing the actual submission for a given environment is out of scope here and is best
   handled by a separate, environment-specific tool the user may have.

Light steps (inspection, marker scoring on moderate data, reading tables) still run inline
on the login node. Only the genuinely GPU/large-memory steps are handed off.

**scvi-tools training hygiene (validated on an RTX 4090, scvi 1.4.3).** The scvi steps
(`cta reference train-model` / `map-scarches`) train as a **single process** by design, and
the scripts encode the matching best practices so a SLURM run stays clean and fast:
- `devices=1` — pin single-device training. On a multi-GPU node this also prevents Lightning
  attempting DDP, which is what would trigger its *"prepend `srun`"* hint; pinning the device
  removes that at the root rather than muting a message.
- `torch.set_float32_matmul_precision('high')` — Tensor-Core GPUs (RTX 40xx / A100) otherwise
  warn and run float32 matmuls at full precision; `'high'` is a real speedup for negligible
  precision loss.
- DataLoader `num_workers` left at scvi's default **0**: maintainers report ~no speedup from
  raising it and high values can deadlock with JAX, so Lightning's *"does not have many
  workers"* hint is a benign false positive (the scripts silence exactly that one message).
- scArches surgery uses `weight_decay=0.0` (keeps the reference latent fixed) and
  `check_val_every_n_epoch=10`, per the scvi reference-mapping tutorial.

## Reproducibility caveats (be honest about GPU non-determinism)

- GPU results are **not bit-identical** to CPU, and deep models (scVI/scANVI) are
  **stochastic** — seed them (`scvi.settings.seed = 0`; the scripts do) but still report
  predictions as a **hypothesis**, never a verdict.
- GPU k-NN (`cta crosscheck reference-map --knn-method rapids`) and any optional GPU clustering/UMAP
  (e.g. rapids-singlecell, not a default extra) are not bit-identical to their CPU
  equivalents. This skill annotates **pre-existing** clusters and does not re-cluster the
  main object, so this mostly doesn't bite — but if a GPU-accelerated step ever feeds
  clustering, treat the clusters as the same kind of input the skill validates, not as
  ground truth.

## The invariant that does not change with the hardware

**Every automated call is a HYPOTHESIS to reconcile against the marker evidence, never a
verdict** — a GPU reference model and a foundation model are held to exactly the same bar
as the CPU CellMapper cross-check (`reference/automated_tools.md`,
`reference/methodology.md` §3–4). A bigger model does not earn more authority; it earns one
more independent opinion to weigh. Data stays on-machine, and the run stays legible and
resumable on any profile.
