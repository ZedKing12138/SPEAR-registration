"""Command-line entry points. Run ``spear COMMAND --help`` for every option."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import (
    PairDataset,
    SubjectDataset,
    _read_nifti,
    _resample_to_grid,
    prepare_dataset,
    save_prediction,
)
from .device import resolve_device
from .engine import (
    evaluate,
    load_registration,
    pretrain_segmenter,
    seed_everything,
    train_registration,
)
from .geometry import resize_flow, warp


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


@torch.no_grad()
def register_nifti(
    checkpoint: str, moving: str, fixed: str, output: str, device: str = "cpu", threads: int = 2
) -> dict:
    """Predict on the training grid, export on the canonical native fixed grid.

    The caller supplies anatomically affine-prealigned images. Matching NIfTI
    grids is world-coordinate resampling, NOT affine anatomical registration.
    The exported intensity image uses original moving intensities, not the
    normalized training image. Flow vectors use fixed-grid voxel units.
    """
    device = resolve_device(device)
    seed_everything(17, threads)
    model, metadata = load_registration(checkpoint, device)
    fixed_img = _read_nifti(Path(fixed))
    moving_img = _read_nifti(Path(moving))
    if len(fixed_img.shape) != 3 or len(moving_img.shape) != 3:
        raise ValueError("Registration expects scalar 3D NIfTI volumes")
    with tempfile.TemporaryDirectory(prefix="spear_pair_") as tmp:
        manifest = Path(tmp) / "input.json"
        manifest.write_text(
            json.dumps(
                {
                    "subjects": [
                        {"id": "moving", "image": str(Path(moving).resolve()), "split": "train"},
                        {"id": "fixed", "image": str(Path(fixed).resolve()), "split": "train"},
                    ]
                }
            )
        )
        prepared = prepare_dataset(
            manifest,
            Path(tmp) / "prepared",
            metadata["training_shape"],
            reference=str(Path(fixed).resolve()),
        )
        samples = SubjectDataset(prepared, "train")
        pred = model(samples[0]["image"][None].to(device), samples[1]["image"][None].to(device))
    native_shape = tuple(reversed(fixed_img.shape))
    native_flow = resize_flow(pred["flow"], native_shape).cpu()
    aligned_moving = _resample_to_grid(moving_img, (fixed_img.shape, fixed_img.affine), order=1)
    native_array = aligned_moving.get_fdata(dtype=np.float32)
    if not np.isfinite(native_array).all():
        raise ValueError("Input contains non-finite intensities")
    raw = torch.from_numpy(native_array.transpose(2, 1, 0).copy())[None, None]
    raw_warped = warp(raw, native_flow)
    saved = save_prediction(output, "moving", "fixed", raw_warped, native_flow, fixed_img.affine)
    return {key: str(value) for key, value in saved.items()}


def main(argv: list[str] | None = None) -> None:
    """Explicit workflows keep segmentation pretraining separate from registration."""
    p = argparse.ArgumentParser(
        description="SPEAR 3D registration. Inputs must be affine-prealigned."
    )
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "prepare", help="Resample NIfTIs to a common world grid and normalize intensities"
    )
    s.add_argument(
        "--manifest", required=True, help="JSON subjects with id, image, optional label, split"
    )
    s.add_argument("--output", required=True)
    s.add_argument(
        "--shape", nargs=3, type=_positive, default=[36, 36, 36], metavar=("D", "H", "W")
    )
    s.add_argument("--reference", help="Common NIfTI grid; defaults to the first training image")
    s = sub.add_parser(
        "pretrain-segmenter", help="Train auxiliary U-Net with TRAIN annotations only"
    )
    s.add_argument("--manifest", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--steps", type=_positive, default=200)
    s.add_argument("--batch-size", type=_positive, default=2)
    s.add_argument("--channels", type=_positive, default=8)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device (default cuda:0); use cpu explicitly for CPU",
    )
    s.add_argument("--threads", type=_positive, default=2)
    s.add_argument("--seed", type=int, default=17)
    s = sub.add_parser("train", help="Train image-based registration and select by validation NCC")
    s.add_argument("--config", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--output", required=True)
    s.add_argument(
        "--segmenter", help="Pretrained prior checkpoint; required for semantic guidance"
    )
    s.add_argument("--steps", type=_positive, help="Override training.max_steps")
    s.add_argument("--device", help="Override training.device")
    s.add_argument(
        "--resume", help="Resume model/optimizer; an incomplete epoch restarts its sampling pass"
    )
    s = sub.add_parser("evaluate", help="Report before/after metrics on a subject-disjoint split")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--split", choices=["train", "val", "test"], default="test")
    s.add_argument("--output", required=True)
    s.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device (default cuda:0); use cpu explicitly for CPU",
    )
    s.add_argument("--save-examples", type=int, default=2)
    s.add_argument("--threads", type=_positive, default=2)
    s = sub.add_parser("register", help="Register a NIfTI pair and export a native fixed-grid flow")
    for name in ("checkpoint", "moving", "fixed", "output"):
        s.add_argument("--" + name, required=True)
    s.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device (default cuda:0); use cpu explicitly for CPU",
    )
    s.add_argument("--threads", type=_positive, default=2)
    a = p.parse_args(argv)
    if a.command == "prepare":
        result = {
            "manifest": str(prepare_dataset(a.manifest, a.output, tuple(a.shape), a.reference))
        }
    elif a.command == "pretrain-segmenter":
        result = pretrain_segmenter(
            a.manifest,
            a.output,
            a.steps,
            a.batch_size,
            a.channels,
            a.lr,
            a.device,
            a.seed,
            a.threads,
        )
    elif a.command == "train":
        config = yaml.safe_load(Path(a.config).read_text())
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a YAML mapping")
        config.setdefault("training", {})
        if a.steps:
            config["training"]["max_steps"] = a.steps
        if a.device:
            config["training"]["device"] = a.device
        result = train_registration(config, a.manifest, a.output, a.segmenter, a.resume)
    elif a.command == "evaluate":
        seed_everything(17, a.threads)
        model, info = load_registration(a.checkpoint, a.device)
        dataset = PairDataset(SubjectDataset(a.manifest, a.split))
        result = evaluate(
            model,
            dataset,
            a.device,
            info["config"]["training"].get("ncc_window", 9),
            a.output,
            a.save_examples,
        )["summary"]
    else:
        result = register_nifti(a.checkpoint, a.moving, a.fixed, a.output, a.device, a.threads)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
