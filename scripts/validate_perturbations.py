#!/usr/bin/env python3
"""Test a frozen network on real held-out MRI with controlled synthetic warps.

This same-subject perturbation test uses real held-out IXITiny intensities
and whole-brain masks. Synthetic displacement fields are generated under the
declared seed to evaluate recovery of controlled spatial perturbations.
The exact inverse is used only for reporting endpoint error, never as a model
input, registration loss, training target or test-time optimization objective.

Example (after installing the project and its ``plots`` extra)::

    python scripts/validate_perturbations.py --checkpoint runs/demo/best.pt \
        --manifest data/prepared/manifest.json --output reports/perturbations

IXI-derived image/mask data and rendered image adaptations remain CC BY-SA 3.0.
The output report includes attribution, independently of the software license.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from spear.data import SubjectDataset
from spear.engine import load_registration, seed_everything, write_json
from spear.geometry import compose, warp
from spear.losses import local_ncc_loss
from spear.metrics import dice_score, folding_rate


DATA_ATTRIBUTION = {
    "source": "IXI Dataset; low-resolution IXITiny derivative provided by the TorchIO project",
    "original_source_url": "https://brain-development.org/ixi-dataset/",
    "derivative_provider_url": "https://docs.torchio.org/latest/datasets/",
    "license": "CC-BY-SA-3.0",
    "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
    "adaptations": "Common-grid preprocessing, normalization, synthetic spatial warping and slice visualization",
    "figure_license": "The MRI-derived figure is CC-BY-SA-3.0; the software license is separate.",
}


def smooth_random_deformation(
    shape: Sequence[int], amplitude: float, generator: torch.Generator
) -> Tensor:
    """Generate one deterministic smooth pullback field in CPU voxel units.

    A seeded 3x3x3 random control grid is interpolated to the image shape and
    smoothed twice with a replicate-padded 5x5x5 box filter. Its largest vector
    norm is normalized to one. The translation vector has independently drawn
    components in [-1/sqrt(3), 1/sqrt(3)], so its norm is at most one as well.
    The final field is ``amplitude * (0.75 * smooth + 0.25 * translation)``;
    by the triangle inequality, its maximum vector norm cannot exceed amplitude.
    """
    if len(shape) != 3 or any(int(n) != n or n < 7 for n in shape):
        raise ValueError("shape must contain three integers >= 7 for a border-3 interior")
    if not 0 <= amplitude < float("inf"):
        raise ValueError("amplitude must be a finite nonnegative number")
    if generator.device.type != "cpu":
        raise ValueError("Use a CPU generator for device-independent synthetic fixtures")
    control = torch.rand((1, 3, 3, 3, 3), generator=generator, dtype=torch.float32) * 2 - 1
    smooth = F.interpolate(control, size=tuple(shape), mode="trilinear", align_corners=True)
    for _ in range(2):
        smooth = F.avg_pool3d(F.pad(smooth, (2, 2, 2, 2, 2, 2), mode="replicate"), 5, stride=1)
    smooth = smooth - smooth.mean(dim=(2, 3, 4), keepdim=True)
    smooth = smooth / smooth.square().sum(dim=1).sqrt().amax().clamp_min(1e-8)
    translation = (torch.rand((1, 3, 1, 1, 1), generator=generator) * 2 - 1) / (3**0.5)
    return amplitude * (0.75 * smooth + 0.25 * translation)


def inverse_displacement(deformation: Tensor, iterations: int = 20) -> Tensor:
    """Solve ``f(x) + d(x + f(x)) = 0`` by fixed-point iteration.

    The result is the pullback needed to undo ``moving = warp(fixed, d)``.
    Fixed-point convergence is not assumed for arbitrary large deformations;
    the evaluation protocol checks the composition residual explicitly.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    inverse = -deformation.clone()
    for _ in range(iterations):
        inverse = -warp(deformation, inverse)
    return inverse


def interior_vector_norm(field: Tensor, border: int = 3) -> Tensor:
    """Return per-voxel Euclidean norms after excluding a fixed border."""
    if border < 0 or any(n <= 2 * border for n in field.shape[2:]):
        raise ValueError("border must leave a nonempty three-dimensional interior")
    crop = slice(border, -border) if border else slice(None)
    return field[:, :, crop, crop, crop].square().sum(dim=1).sqrt()


def interior_epe(prediction: Tensor, reference: Tensor, border: int = 3) -> Tensor:
    """Mean endpoint error in voxels on a fixed central interior mask."""
    if prediction.shape != reference.shape:
        raise ValueError("predicted and reference fields must have matching shapes")
    return interior_vector_norm(prediction - reference, border).mean()


def _save_figure(path: Path, example: dict[str, Tensor | str]) -> None:
    """Render exactly the first declared case, avoiding best-case selection."""
    # Plotting is an optional dependency, so importing this script for tests or
    # numerical evaluation does not require matplotlib to be installed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = example["fixed"][0, 0].detach().cpu()
    moving = example["moving"][0, 0].detach().cpu()
    warped = example["warped"][0, 0].detach().cpu()
    middle = fixed.shape[0] // 2
    difference_before, difference_after = (moving - fixed).abs(), (warped - fixed).abs()
    error_max = max(
        float(difference_before[middle].max()), float(difference_after[middle].max()), 1e-6
    )
    arrays = (
        fixed[middle],
        moving[middle],
        warped[middle],
        difference_before[middle],
        difference_after[middle],
    )
    names = (
        "Fixed (real MRI)",
        "Moving (synthetic warp)",
        "Network output",
        "Absolute error: before",
        "Absolute error: after",
    )
    figure, axes = plt.subplots(1, 5, figsize=(14, 3.4), constrained_layout=True)
    for index, (axis, array, title) in enumerate(zip(axes, arrays, names)):
        axis.imshow(
            array.numpy(),
            cmap="gray" if index < 3 else "magma",
            origin="lower",
            vmin=0,
            vmax=1 if index < 3 else error_max,
        )
        axis.set_title(title, fontsize=9)
        axis.set_axis_off()
    figure.suptitle(
        f"Held-out {example['subject_id']} | real MRI + synthetic deformation | not an inter-subject benchmark",
        fontsize=11,
    )
    figure.supxlabel(
        "Middle axial slice; matched error scale. IXI / TorchIO IXITiny image adaptation: CC BY-SA 3.0.",
        fontsize=8,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


@torch.no_grad()
def evaluate_perturbations(
    model: torch.nn.Module,
    subjects: Sequence[dict],
    output: str | Path,
    *,
    device: str = "cpu",
    amplitude: float = 1.0,
    cases_per_subject: int = 3,
    seed: int = 701,
    ncc_window: int = 9,
    training_subject_ids: Sequence[str] = (),
    make_figure: bool = True,
) -> dict:
    """Run a frozen network without labels, inverse fields or adaptation inputs.

    Cases are evaluated in sorted subject-ID order. All generated cases are
    retained, without choosing favorable deformations or tuning hyperparameters
    on outcomes. Invalid deformation/inverse checks raise an error rather than
    silently resampling a more favorable fixture.
    """
    if cases_per_subject < 1 or int(cases_per_subject) != cases_per_subject:
        raise ValueError("cases_per_subject must be a positive integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    ordered = sorted(subjects, key=lambda item: str(item["id"]))
    if not ordered:
        raise ValueError("At least one held-out subject is required")
    subject_ids = [str(item["id"]) for item in ordered]
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Subject IDs must be unique")
    if set(subject_ids) & set(training_subject_ids):
        raise ValueError("Held-out subjects must not overlap the checkpoint's training subjects")
    if any("label" not in item for item in ordered):
        raise ValueError("Whole-brain labels are required for the declared Dice sanity check")

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    model.eval()
    rows = []
    first_example = None
    border, inverse_iterations = 3, 20
    for subject_index, subject in enumerate(ordered):
        fixed = subject["image"].unsqueeze(0).to(device)
        fixed_label = (subject["label"].unsqueeze(0).to(device) > 0).long()
        for case_index in range(cases_per_subject):
            case_seed = seed + subject_index * cases_per_subject + case_index
            generator = torch.Generator(device="cpu").manual_seed(case_seed)
            deformation = smooth_random_deformation(fixed.shape[2:], amplitude, generator).to(
                device
            )
            inverse = inverse_displacement(deformation, inverse_iterations)
            composition_error = interior_vector_norm(compose(deformation, inverse), border)
            reverse_composition_error = interior_vector_norm(compose(inverse, deformation), border)
            source_folding = float(folding_rate(deformation))
            if source_folding > 0 or float(composition_error.max()) > 1e-3:
                raise RuntimeError(
                    f"Synthetic case {case_seed} violates the declared fold-free/inverse-residual protocol; "
                    "no case was discarded or adaptively regenerated."
                )
            moving = warp(fixed, deformation)
            moving_label = warp(fixed_label.float(), deformation, mode="nearest").round().long()

            # The complete inference call: exactly two intensity tensors. Neither
            # the label nor the known inverse is ever passed into the network.
            prediction = model(moving, fixed)
            predicted_flow, registered = prediction["flow"], prediction["warped"]
            registered_label = (
                warp(moving_label.float(), predicted_flow, mode="nearest").round().long()
            )
            row = {
                "subject_id": str(subject["id"]),
                "case_index": case_index,
                "case_seed": case_seed,
                "ncc_before": float(1 - local_ncc_loss(moving, fixed, ncc_window)),
                "ncc_after": float(1 - local_ncc_loss(registered, fixed, ncc_window)),
                "mse_before": float(F.mse_loss(moving, fixed)),
                "mse_after": float(F.mse_loss(registered, fixed)),
                "whole_brain_dice_before": float(dice_score(moving_label, fixed_label, labels=[1])),
                "whole_brain_dice_after": float(
                    dice_score(registered_label, fixed_label, labels=[1])
                ),
                "folding_percent": 100 * float(folding_rate(predicted_flow)),
                "inverse_epe_voxels": float(interior_epe(predicted_flow, inverse, border)),
                "identity_epe_voxels": float(
                    interior_epe(torch.zeros_like(inverse), inverse, border)
                ),
                "inverse_composition_mean_voxels": float(composition_error.mean()),
                "inverse_composition_max_voxels": float(composition_error.max()),
                "reverse_composition_mean_voxels": float(reverse_composition_error.mean()),
                "synthetic_folding_percent": 100 * source_folding,
                "synthetic_max_displacement_voxels": float(
                    deformation.square().sum(1).sqrt().max()
                ),
            }
            rows.append(row)
            if first_example is None:
                first_example = {
                    "subject_id": str(subject["id"]),
                    "fixed": fixed.cpu(),
                    "moving": moving.cpu(),
                    "warped": registered.cpu(),
                }

    keys = [key for key in rows[0] if key not in {"subject_id", "case_index", "case_seed"}]
    summary = {}
    for key in keys:
        values = torch.tensor([row[key] for row in rows], dtype=torch.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    report = {
        "experiment": "Real held-out T1 MRI with controlled synthetic same-subject deformations",
        "is_inter_subject_benchmark": False,
        "subject_ids": subject_ids,
        "subject_count": len(subject_ids),
        "case_count": len(rows),
        "protocol": {
            "split": "test",
            "seed": seed,
            "cases_per_subject": cases_per_subject,
            "case_seed_rule": "seed + sorted_subject_index * cases_per_subject + case_index",
            "amplitude_voxels": amplitude,
            "deformation": "CPU-seeded uniform 3x3x3 control grid; trilinear upsampling; two replicate-padded 5x5x5 box smooths; spatial centering and max-L2 normalization; amplitude*(0.75*smooth + 0.25*bounded translation)",
            "amplitude_definition": "Upper bound on synthetic displacement vector magnitude, in prepared-grid voxels",
            "flow_convention": "Pullback, channels z/y/x, voxel units, align_corners=True, border padding",
            "moving_generation": "moving=warp(fixed,d); moving_label=warp(fixed_label,d,nearest)",
            "inverse_iterations": inverse_iterations,
            "inverse_equation": "f=-warp(d,f); initialized with -d",
            "inverse_max_residual_tolerance_voxels": 0.001,
            "epe_border_excluded_voxels": border,
            "ncc_window": ncc_window,
            "network_inputs": ["moving intensity image", "fixed intensity image"],
            "test_time_optimization": False,
            "ground_truth_inverse_usage": "Metrics only; never training, model input or optimization",
            "hyperparameter_selection_on_test": False,
            "checkpoint_training_overlap_checked": bool(training_subject_ids),
            "figure_selection": "First case of the first lexicographically sorted held-out subject",
        },
        "metric_notes": {
            "ncc": "1 minus squared local NCC loss; jointly constant windows excluded",
            "dice": "Binary whole-brain foreground Dice only, not regional/anatomical Dice",
            "folding": "Percent of predicted-field voxels with Jacobian determinant <=0",
            "epe": "Euclidean displacement endpoint error in prepared-grid voxels; fixed three-voxel border excluded",
            "identity": "Identity-flow baseline against the known numerical inverse on the same interior",
            "inverse_interpolation": "The numerical inverse undoes the continuous sampling map; finite-grid resampling blur and nearest-label aliasing are not exactly invertible",
            "interpretation": "A small controlled-deformation sanity experiment cannot establish inter-subject or clinical registration performance",
        },
        "data_attribution": DATA_ATTRIBUTION,
        "summary": summary,
        "cases": rows,
    }
    write_json(destination / "metrics.json", report)
    if make_figure:
        _save_figure(destination / "example.png", first_example)
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--cases-per-subject", type=int, default=3)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    seed_everything(args.seed, args.threads)
    model, metadata = load_registration(args.checkpoint, args.device)
    subjects = SubjectDataset(args.manifest, "test")
    if "training_subject_ids" not in metadata:
        raise ValueError(
            "Checkpoint must record training_subject_ids to verify subject-disjoint evaluation"
        )
    expected_shape = metadata.get("training_shape")
    if expected_shape and any(
        tuple(subject["image"].shape[1:]) != tuple(expected_shape) for subject in subjects
    ):
        raise ValueError("Prepared subject shape must match the checkpoint's training grid")
    window = metadata.get("config", {}).get("training", {}).get("ncc_window", 9)
    report = evaluate_perturbations(
        model,
        subjects,
        args.output,
        device=args.device,
        amplitude=args.amplitude,
        cases_per_subject=args.cases_per_subject,
        seed=args.seed,
        ncc_window=window,
        training_subject_ids=metadata["training_subject_ids"],
    )
    # Hashes identify exact inputs without leaking local paths or host metadata.
    report["checkpoint_sha256"] = _sha256(args.checkpoint)
    report["prepared_manifest_sha256"] = _sha256(args.manifest)
    write_json(args.output / "metrics.json", report)
    print(
        f"Evaluated {report['case_count']} fixed synthetic perturbations on {report['subject_count']} held-out subjects."
    )
    print("Results: metrics.json and example.png. This is not an inter-subject benchmark.")


if __name__ == "__main__":
    main()
