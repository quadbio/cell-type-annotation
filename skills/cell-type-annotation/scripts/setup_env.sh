#!/usr/bin/env bash
# Bootstrap the dedicated, reusable uv venv for the cell-type-annotation skill.
#
# Creates a Python 3.12 venv with the scVerse stack at $CTA_VENV
# (default: ~/.cache/cell-type-annotation/venv). This is intentionally SEPARATE
# from the user's project environment — the skill never installs into it.
#
# Idempotent: re-running reuses the existing venv and just ensures deps are present.
# Usage:  bash setup_env.sh            # create/update the venv
#         CTA_VENV=/path bash setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CTA_VENV:-$HOME/.cache/cell-type-annotation/venv}"
PY="${CTA_PYTHON:-3.12}"

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
