#!/usr/bin/env bash
# Run from any directory: bash /path/to/SPEAR/scripts/bootstrap_cpu.sh
# Creates an isolated local environment and installs a CPU PyTorch wheel before
# resolving the project, avoiding a multi-GB CUDA download on CPU machines.
# Python 3.10+ is required; override SPEAR_PYTHON if your interpreter has another
# name. Windows users may run the corresponding python -m pip commands manually.
set -euo pipefail
SPEAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SPEAR_PYTHON="${SPEAR_PYTHON:-python3}"
"$SPEAR_PYTHON" -m venv "$SPEAR_ROOT/.venv"
SPEAR_ENV_PYTHON="$SPEAR_ROOT/.venv/bin/python"
"$SPEAR_ENV_PYTHON" -m pip install --upgrade pip
"$SPEAR_ENV_PYTHON" -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
"$SPEAR_ENV_PYTHON" -m pip install -e "$SPEAR_ROOT[dev,plots]"
# This script does not download medical images or train models automatically.
"$SPEAR_ENV_PYTHON" -m pytest "$SPEAR_ROOT/tests"
