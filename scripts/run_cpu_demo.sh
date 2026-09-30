#!/usr/bin/env bash
# Run the real-MRI CPU development workflow from the project root.
# Install first with scripts/bootstrap_cpu.sh, then activate that environment.
# Override SPEAR_PYTHON if necessary. Data are downloaded separately (CC BY-SA).
# Uses 12 low-resolution IXITiny subjects with binary whole-brain masks.
set -euo pipefail
cd "$(dirname "$0")/.."
SPEAR_PYTHON="${SPEAR_PYTHON:-python}"
SPEAR_RUN_DIR="${SPEAR_RUN_DIR:-runs/ixi_cpu}"
if [ -e "$SPEAR_RUN_DIR/registration/best.pt" ]; then
  echo "Choose a fresh SPEAR_RUN_DIR or resume explicitly with spear train --resume."
  exit 1
fi
"$SPEAR_PYTHON" scripts/download_ixi_tiny.py --output data/raw_ixi_tiny
"$SPEAR_PYTHON" -m spear prepare --manifest data/raw_ixi_tiny/manifest.json \
  --output data/ixi36 --shape 36 36 36
"$SPEAR_PYTHON" -m spear pretrain-segmenter --manifest data/ixi36/manifest.json \
  --output "$SPEAR_RUN_DIR/prior.pt" --steps 120 --batch-size 2 --channels 8 --threads 2 --device cpu
"$SPEAR_PYTHON" -m spear train --config configs/cpu_demo.yaml \
  --manifest data/ixi36/manifest.json --segmenter "$SPEAR_RUN_DIR/prior.pt" \
  --output "$SPEAR_RUN_DIR/registration"
"$SPEAR_PYTHON" -m spear evaluate --checkpoint "$SPEAR_RUN_DIR/registration/best.pt" \
  --manifest data/ixi36/manifest.json --split test --output "$SPEAR_RUN_DIR/test" --device cpu
"$SPEAR_PYTHON" scripts/validate_perturbations.py \
  --checkpoint "$SPEAR_RUN_DIR/registration/best.pt" --manifest data/ixi36/manifest.json \
  --output "$SPEAR_RUN_DIR/controlled_deformations" --device cpu --amplitude 1.0 \
  --cases-per-subject 3 --seed 701
