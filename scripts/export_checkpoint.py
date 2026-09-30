#!/usr/bin/env python3
"""Export a small, portable inference checkpoint without optimizer state.

Example:
    python scripts/export_checkpoint.py runs/registration/best.pt \
        runs/registration/exported.pt

The frozen auxiliary prior is part of model.state_dict and remains included.
This export is for inference; resume training from the original last.pt instead.
"""

import argparse
from pathlib import Path

import torch

from spear.engine import save_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != "spear_registration":
        raise ValueError("Expected a SPEAR registration checkpoint")
    for key in ("optimizer", "scheduler"):
        checkpoint.pop(key, None)
    checkpoint["inference_only"] = True
    # Describe the exported artifact without changing the source checkpoint's
    # scope, dataset, device, or other training provenance. Loading tensors on
    # CPU makes serialization portable; it does not identify the training device.
    checkpoint["export_format"] = "spear_inference"
    save_checkpoint(args.output, checkpoint)
    print(args.output)


if __name__ == "__main__":
    main()
