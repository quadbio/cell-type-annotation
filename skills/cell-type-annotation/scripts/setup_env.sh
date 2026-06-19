#!/usr/bin/env bash
# Bootstrap the dedicated, reusable uv venv for the cell-type-annotation skill.
#
# Creates a Python 3.12 venv with the scVerse stack at $CTA_VENV
# (default: ~/.cache/cell-type-annotation/venv). This is intentionally SEPARATE
# from the user's project environment — the skill never installs into it.
#
# Profiles (compute-environment aware — see reference/compute_environments.md):
#   cpu  (default)  CPU / Apple-Silicon core: requirements.txt + requirements-extra.txt.
#   gpu             ALSO installs the GPU / large-memory extras (requirements-gpu.txt):
#                   scvi-tools (scANVI/scArches + HuggingFace hub), rapids-singlecell,
#                   faiss-gpu, scimilarity. Use on an NVIDIA CUDA host (or where scvi-tools
#                   can use MPS). The GPU extras are best-effort: any wheel that can't
#                   install on this platform is skipped and never breaks the CPU core.
#
# Idempotent: re-running reuses the existing venv and just ensures deps are present.
# Usage:  bash setup_env.sh                  # cpu profile (default)
#         bash setup_env.sh --profile gpu    # cpu core + GPU/large-memory extras
#         CTA_PROFILE=gpu bash setup_env.sh  # same, via env var
#         CTA_VENV=/path bash setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CTA_VENV:-$HOME/.cache/cell-type-annotation/venv}"
PY="${CTA_PYTHON:-3.12}"
PROFILE="${CTA_PROFILE:-cpu}"

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="${2:-cpu}"; shift 2 ;;
    --profile=*) PROFILE="${1#*=}"; shift ;;
    *) echo "ERROR: unknown argument '$1' (expected --profile cpu|gpu)" >&2; exit 2 ;;
  esac
done
case "$PROFILE" in
  cpu|gpu) ;;
  *) echo "ERROR: --profile must be 'cpu' or 'gpu' (got '$PROFILE')" >&2; exit 2 ;;
esac
echo ">> profile: $PROFILE"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' is not installed or not on PATH. Install it from https://docs.astral.sh/uv/ first." >&2
  exit 1
fi

echo ">> cell-type-annotation venv: $VENV (python $PY)"
mkdir -p "$(dirname "$VENV")"

if [ ! -x "$VENV/bin/python" ]; then
  echo ">> creating venv"
  uv venv --python "$PY" "$VENV"
else
  echo ">> reusing existing venv"
fi

PYBIN="$VENV/bin/python"

echo ">> installing core requirements"
uv pip install --python "$PYBIN" -r "$SCRIPT_DIR/requirements.txt"

# Record the core numpy version so we can prove the extras didn't move it.
NUMPY_BEFORE="$("$PYBIN" -c 'import numpy; print(numpy.__version__)')"
echo ">> core numpy: $NUMPY_BEFORE"

echo ">> installing best-effort extras (failures are non-fatal)"
# Install each extra CONSTRAINED by the core requirements (-c) so no extra can
# silently downgrade a core pin (notably numpy>=2). An extra that can't satisfy the
# constraint fails to install and is skipped — it never corrupts the core stack.
while IFS= read -r line; do
  pkg="$(echo "$line" | sed 's/#.*//' | xargs || true)"
  [ -z "$pkg" ] && continue
  if uv pip install --python "$PYBIN" -c "$SCRIPT_DIR/requirements.txt" "$pkg" >/dev/null 2>&1; then
    echo "   ok: $pkg"
  else
    echo "   skipped (unavailable, or conflicts with a core pin): $pkg" >&2
  fi
done < "$SCRIPT_DIR/requirements-extra.txt"

if [ "$PROFILE" = "gpu" ]; then
  # Build the GPU extras WHERE THE GPU WORK RUNS. On a scheduler/login node with no GPU,
  # installing multi-GB CUDA wheels is wasted (they can't run there) and burns quota — so
  # by default we SKIP the GPU extras when no CUDA GPU is visible on this host, leaving a
  # working CPU core. Override only to pre-stage a SHARED venv for a GPU node (shared FS):
  #   CTA_ALLOW_GPU_BUILD_WITHOUT_GPU=1 bash setup_env.sh --profile gpu
  GPU_VISIBLE=no
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    GPU_VISIBLE=yes
  fi
  if [ "$GPU_VISIBLE" = "no" ] && [ -z "${CTA_ALLOW_GPU_BUILD_WITHOUT_GPU:-}" ]; then
    echo ">> --profile gpu requested but NO CUDA GPU is visible on this host." >&2
    echo "   Skipping the GPU extras — the CPU core above is installed and usable here" >&2
    echo "   (enough for detection + light steps on a login node). The GPU extras (scvi-" >&2
    echo "   tools, rapids-singlecell, faiss-gpu, scimilarity) belong WHERE the GPU runs:" >&2
    echo "   run 'setup_env.sh --profile gpu' INSIDE a GPU job, into this same venv" >&2
    echo "   ($VENV — put it on a shared filesystem so the GPU node sees it)." >&2
    echo "   To pre-stage them here anyway: CTA_ALLOW_GPU_BUILD_WITHOUT_GPU=1 ..." >&2
  else
    [ "$GPU_VISIBLE" = "yes" ] && echo ">> CUDA GPU visible — installing GPU / large-memory extras" \
      || echo ">> no GPU visible, but CTA_ALLOW_GPU_BUILD_WITHOUT_GPU set — pre-staging GPU extras"
    echo "   (best-effort; rapids-singlecell resolves via the NVIDIA pip index)"
    GPU_OK=""; GPU_SKIP=""
    while IFS= read -r line; do
      pkg="$(echo "$line" | sed 's/#.*//' | xargs || true)"
      [ -z "$pkg" ] && continue
      if uv pip install --python "$PYBIN" --extra-index-url https://pypi.nvidia.com \
           -c "$SCRIPT_DIR/requirements.txt" "$pkg" >/dev/null 2>&1; then
        echo "   ok: $pkg"; GPU_OK="$GPU_OK $pkg"
      else
        echo "   skipped (unavailable on this platform, or conflicts with a core pin): $pkg" >&2
        GPU_SKIP="$GPU_SKIP $pkg"
      fi
    done < "$SCRIPT_DIR/requirements-gpu.txt"
    # Make silent skips visible: report which GPU extras actually import.
    "$PYBIN" - <<'PYEOF'
import importlib.util as u
checks = [("scvi-tools", "scvi"), ("rapids-singlecell", "rapids_singlecell"),
          ("faiss-gpu", "faiss"), ("scimilarity", "scimilarity")]
for name, mod in checks:
    print(f"   gpu extra {name:<18}: {'present' if u.find_spec(mod) else 'MISSING'}")
PYEOF
  fi
fi

echo ">> verifying core imports"
NUMPY_BEFORE="$NUMPY_BEFORE" "$PYBIN" - <<'PYEOF'
import importlib.util, os, sys
core = ["scanpy", "anndata", "decoupler", "cosg", "celltypist", "cellmapper",
        "pynndescent", "matplotlib", "jinja2"]
missing = [m for m in core if importlib.util.find_spec(m) is None]
if missing:
    print(f"ERROR: core packages failed to import: {missing}", file=sys.stderr)
    sys.exit(1)
import numpy, scanpy as sc
before = os.environ.get("NUMPY_BEFORE", "")
if before and numpy.__version__ != before:
    print(f"ERROR: an extra moved numpy {before} -> {numpy.__version__}; the core "
          "stack may be on an unintended ABI. Re-run after fixing requirements pins.",
          file=sys.stderr)
    sys.exit(1)
print(f"   scanpy {sc.__version__}  numpy {numpy.__version__}  "
      f"python {sys.version.split()[0]}  OK")
PYEOF

echo ">> done. Run scripts with: $PYBIN <script>.py ..."
echo "$VENV"
