#!/usr/bin/env bash
# Install the main single-GPU workflow in an isolated environment.
# Usage: bash scripts/bootstrap_gpu.sh [cu128]
# Pick an official wheel channel compatible with your GPU and NVIDIA driver:
# https://pytorch.org/get-started/locally/ . cu128 is a configurable default,
# not an assertion that every machine supports the same CUDA runtime.
# SPEAR_PYTHON, SPEAR_VENV and SPEAR_TORCH_SPEC may override the interpreter,
# environment directory and PyTorch requirement (for example torch==2.11.0).
# CUDA_VISIBLE_DEVICES/SPEAR_DEVICE select the visible GPU for the smoke test.
set -euo pipefail
SPEAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SPEAR_PYTHON="${SPEAR_PYTHON:-python3}"
SPEAR_VENV="${SPEAR_VENV:-$SPEAR_ROOT/.venv-gpu}"
SPEAR_CUDA_CHANNEL="${1:-cu128}"
if [[ ! "$SPEAR_CUDA_CHANNEL" =~ ^cu[0-9]+$ ]] || [ "$#" -gt 1 ]; then
  echo "Usage: bash scripts/bootstrap_gpu.sh [cu128|another official CUDA wheel channel]" >&2
  exit 2
fi
"$SPEAR_PYTHON" -m venv "$SPEAR_VENV"
SPEAR_ENV_PYTHON="$SPEAR_VENV/bin/python"
"$SPEAR_ENV_PYTHON" -m pip install --upgrade pip
# Reinstall explicitly: an existing CPU build must not satisfy this step.
"$SPEAR_ENV_PYTHON" -m pip install --upgrade --force-reinstall \
  "${SPEAR_TORCH_SPEC:-torch>=2.4,<3}" \
  --index-url "https://download.pytorch.org/whl/$SPEAR_CUDA_CHANNEL"
"$SPEAR_ENV_PYTHON" -m pip install -e "$SPEAR_ROOT[dev,datasets]"
"$SPEAR_ENV_PYTHON" - "${SPEAR_DEVICE:-cuda:0}" <<'PY'
import sys
import torch
from spear.device import resolve_device

device = resolve_device(sys.argv[1])
if not device.startswith("cuda:"):
    raise ValueError("bootstrap_gpu.sh requires a CUDA device")
print(f"PyTorch {torch.__version__}; {device}: {torch.cuda.get_device_name(device)}")
PY
# This genuinely executes CUDA forward/backward when a GPU is present. Device
# validation above prevents a misleading successful install followed by a skip.
"$SPEAR_ENV_PYTHON" -m pytest "$SPEAR_ROOT/tests/test_gpu_interface.py" -q
echo "Activate the GPU environment: source $SPEAR_VENV/bin/activate"
