#!/usr/bin/env bash
# Main SPEAR single-GPU training workflow; float32 throughout.
# First prepare IXI/OASIS/IBSR18 with the dataset preprocessing script.
# Usage:
#   bash scripts/run_gpu.sh --manifest /data/ixi/prepared/manifest.json \
#     --output /experiments/spear_ixi --device cuda:0
# A separately trained --prior may be reused. It must record TRAIN/VAL subject
# IDs consistent with this manifest. Without --prior, this script learns one
# using training labels, selects it by validation CE, then freezes it for SPEAR.
# Configure the auxiliary segmenter training budget with --prior-steps.
# Checkpoint resumption is exposed by: spear train ... --resume /path/last.pt
# Evaluation is opt-in (--evaluate). All runtime outputs go below --output;
# these files are not source assets and should not be committed to GitHub.
set -euo pipefail
SPEAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SPEAR_PYTHON="${SPEAR_PYTHON:-python}"
SPEAR_DEVICE="${SPEAR_DEVICE:-cuda:0}"
SPEAR_CONFIG="$SPEAR_ROOT/configs/paper.yaml"
SPEAR_MANIFEST=""
SPEAR_OUTPUT=""
SPEAR_PRIOR=""
SPEAR_PRIOR_STEPS=2000
SPEAR_PRIOR_BATCH=1
SPEAR_PRIOR_CHANNELS=8
SPEAR_TRAIN_STEPS=""
SPEAR_EVALUATE=0

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_gpu.sh --manifest PATH --output DIR [options]
  --device cuda:0          CUDA device; defaults to SPEAR_DEVICE or cuda:0
  --config PATH           Registration YAML; defaults to configs/paper.yaml
  --prior PATH            Reuse an independently pretrained semantic prior
  --prior-steps N         Auxiliary segmenter optimizer steps (default 2000)
  --prior-batch-size N    Auxiliary segmenter batch size (default 1)
  --prior-channels N      Auxiliary segmenter base channels (default 8)
  --steps N               Override registration total optimizer-step budget
  --evaluate              Evaluate the selected checkpoint on the test split
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --evaluate) SPEAR_EVALUATE=1; shift ;;
    --manifest|--output|--device|--config|--prior|--prior-steps|--prior-batch-size|--prior-channels|--steps)
      if [ "$#" -lt 2 ]; then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --manifest) SPEAR_MANIFEST="$2" ;;
        --output) SPEAR_OUTPUT="$2" ;;
        --device) SPEAR_DEVICE="$2" ;;
        --config) SPEAR_CONFIG="$2" ;;
        --prior) SPEAR_PRIOR="$2" ;;
        --prior-steps) SPEAR_PRIOR_STEPS="$2" ;;
        --prior-batch-size) SPEAR_PRIOR_BATCH="$2" ;;
        --prior-channels) SPEAR_PRIOR_CHANNELS="$2" ;;
        --steps) SPEAR_TRAIN_STEPS="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [ -z "$SPEAR_MANIFEST" ] || [ -z "$SPEAR_OUTPUT" ]; then usage >&2; exit 2; fi
if [ ! -f "$SPEAR_MANIFEST" ] || [ ! -f "$SPEAR_CONFIG" ]; then
  echo "Manifest and configuration must be existing files." >&2; exit 2
fi
if [ -e "$SPEAR_OUTPUT/registration/best.pt" ] || [ -e "$SPEAR_OUTPUT/registration/last.pt" ]; then
  echo "Output already has checkpoints. Choose a fresh output or use spear train --resume." >&2
  exit 2
fi
"$SPEAR_PYTHON" - "$SPEAR_DEVICE" <<'PY'
import sys
from spear.device import resolve_device

device = resolve_device(sys.argv[1])
if not device.startswith("cuda:"):
    raise ValueError("run_gpu.sh requires CUDA; use run_cpu_demo.sh for the CPU smoke workflow")
print(f"SPEAR device: {device}; precision: float32")
PY
if [ -z "$SPEAR_PRIOR" ]; then
  SPEAR_PRIOR="$SPEAR_OUTPUT/prior.pt"
  if [ -e "$SPEAR_PRIOR" ]; then
    echo "Prior already exists. Reuse it explicitly with --prior or choose a fresh output." >&2
    exit 2
  fi
  "$SPEAR_PYTHON" -m spear pretrain-segmenter --manifest "$SPEAR_MANIFEST" \
    --output "$SPEAR_PRIOR" --device "$SPEAR_DEVICE" --steps "$SPEAR_PRIOR_STEPS" \
    --batch-size "$SPEAR_PRIOR_BATCH" --channels "$SPEAR_PRIOR_CHANNELS"
fi
SPEAR_STEP_ARGS=()
if [ -n "$SPEAR_TRAIN_STEPS" ]; then SPEAR_STEP_ARGS=(--steps "$SPEAR_TRAIN_STEPS"); fi
"$SPEAR_PYTHON" -m spear train --config "$SPEAR_CONFIG" --manifest "$SPEAR_MANIFEST" \
  --segmenter "$SPEAR_PRIOR" --output "$SPEAR_OUTPUT/registration" \
  --device "$SPEAR_DEVICE" "${SPEAR_STEP_ARGS[@]}"
if [ "$SPEAR_EVALUATE" -eq 1 ]; then
  "$SPEAR_PYTHON" -m spear evaluate --checkpoint "$SPEAR_OUTPUT/registration/best.pt" \
    --manifest "$SPEAR_MANIFEST" --split test --output "$SPEAR_OUTPUT/test" \
    --device "$SPEAR_DEVICE"
fi
