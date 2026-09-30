"""Reproducible training, checkpointing and evaluation for SPEAR.

No test labels, Dice objective, target displacement or hidden iterative optimizer
are used to train registration. Validation NCC chooses the checkpoint.
"""

from __future__ import annotations

import json
import math
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .data import PairDataset, SubjectDataset, save_prediction
from .device import resolve_device
from .geometry import warp
from .losses import local_ncc_loss, registration_loss
from .metrics import dice_score, folding_rate
from .model import SPEAR, SPEARConfig
from .segmenter import AuxiliarySegmenter, SegmenterConfig, encode_labels


def seed_everything(seed: int = 17, threads: int = 2) -> None:
    """Seed CPU/CUDA RNGs. CUDA grid-sample backward can remain nondeterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_json(path: str | Path, obj: dict | list) -> None:
    """Write human-readable JSON atomically, rejecting NaN/Infinity metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(path: str | Path, payload: dict) -> None:
    """Store only tensors and primitive containers for weights_only loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_prior(path: str | Path, device: str) -> tuple[AuxiliarySegmenter, dict]:
    """Restore a separately pretrained semantic prior, not a random substitute."""
    device = resolve_device(device)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("kind") != "spear_segmenter":
        raise ValueError("Expected a SPEAR auxiliary-segmenter checkpoint")
    model = AuxiliarySegmenter(SegmenterConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device, non_blocking=True).eval(), ckpt


def load_registration(path: str | Path, device: str = "cpu") -> tuple[SPEAR, dict]:
    """A registration checkpoint contains its frozen prior for portable inference."""
    device = resolve_device(device)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("kind") != "spear_registration":
        raise ValueError("Expected a SPEAR registration checkpoint")
    segmenter = None
    if ckpt.get("segmenter_config"):
        segmenter = AuxiliarySegmenter(SegmenterConfig(**ckpt["segmenter_config"]))
    model = SPEAR(SPEARConfig(**ckpt["model_config"]), segmenter=segmenter)
    model.load_state_dict(ckpt["model"])
    return model.to(device, non_blocking=True).eval(), ckpt


def _positive_integer(value, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _common_key_collate(samples: list[dict]) -> dict:
    """Keep optional labels only when every sample in this batch supplies them.

    Registration accepts a mixture of labeled and unlabeled subjects. PyTorch's
    default dictionary collation otherwise depends on the first sample's keys
    and can raise KeyError for such batches. Evaluation uses batch size one,
    retaining every individually available label pair for descriptive Dice.
    """
    common = set.intersection(*(set(sample) for sample in samples))
    return default_collate(
        [{key: sample[key] for key in sample if key in common} for sample in samples]
    )


def _loader(dataset, batch_size: int, shuffle: bool, seed: int, device: str = "cpu"):
    # Pin host batches for asynchronous CPU-to-CUDA transfers. Zero workers
    # keeps the lazy cached dataset portable on Windows and in CPU CI.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
        collate_fn=_common_key_collate,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def evaluate(
    model: SPEAR,
    dataset: PairDataset,
    device: str = "cpu",
    window: int = 9,
    output: str | Path | None = None,
    save_examples: int = 0,
) -> dict:
    """Measure every ordered pair; binary or multi-anatomy labels stay explicit.

    Timing includes the frozen segmenter inside model.forward, after a warmup.
    It excludes file loading, normalization, preprocessing and output writing.
    Folding is reported as a PERCENT, although metrics.folding_rate is a fraction.
    """
    device = resolve_device(device)
    if len(dataset) == 0:
        raise ValueError("Evaluation requires at least two subjects in the split")
    was_training = model.training
    model.eval()
    rows = []
    loader = _loader(dataset, 1, False, 0, device)
    warmup = next(iter(loader))
    model(
        warmup["moving"].to(device, non_blocking=True),
        warmup["fixed"].to(device, non_blocking=True),
    )
    for index, batch in enumerate(loader):
        moving, fixed = (
            batch["moving"].to(device, non_blocking=True),
            batch["fixed"].to(device, non_blocking=True),
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        prediction = model(moving, fixed)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        row = {
            "moving_id": batch["moving_id"][0],
            "fixed_id": batch["fixed_id"][0],
            "ncc_before": float(1 - local_ncc_loss(moving, fixed, window)),
            "ncc_after": float(1 - local_ncc_loss(prediction["warped"], fixed, window)),
            "mse_before": float(F.mse_loss(moving, fixed)),
            "mse_after": float(F.mse_loss(prediction["warped"], fixed)),
            "folding_percent": 100 * float(folding_rate(prediction["flow"])),
            "seconds": elapsed,
        }
        if device.startswith("cuda"):
            row["peak_cuda_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        if "moving_label" in batch and "fixed_label" in batch:
            ml, fl = (
                batch["moving_label"].to(device, non_blocking=True),
                batch["fixed_label"].to(device, non_blocking=True),
            )
            warped_label = warp(ml.float(), prediction["flow"], mode="nearest").round().long()
            row["dice_before"] = float(dice_score(ml, fl))
            row["dice_after"] = float(dice_score(warped_label, fl))
        rows.append(row)
        if output is not None and index < save_examples:
            save_prediction(
                Path(output) / "predictions",
                row["moving_id"],
                row["fixed_id"],
                prediction["warped"].cpu(),
                prediction["flow"].cpu(),
                batch["affine"][0].numpy(),
            )
    summary = {}
    numeric_keys = sorted(
        {key for row in rows for key, value in row.items() if isinstance(value, (int, float))}
    )
    for key in numeric_keys:
        numbers = [row[key] for row in rows if key in row]
        summary[key] = {
            "mean": float(np.mean(numbers)),
            "std": float(np.std(numbers)),
            "min": min(numbers),
            "max": max(numbers),
            "count": len(numbers),
        }
    result = {
        "pair_count": len(rows),
        "summary": summary,
        "pairs": rows,
        "metric_notes": {
            "dice": "Mean foreground-label Dice; absent-in-both classes excluded. IXITiny labels are brain masks only.",
            "ncc": "1 minus local NCC loss, using the configured squared local correlation statistic.",
            "folding": "Percent of voxels with Jacobian determinant <= 0; zero is not a mathematical guarantee.",
            "statistics": "Pairs sharing subjects are dependent; pairwise SD is descriptive, not a confidence interval.",
            "timing": "Warm model forward including its prior, excluding preprocessing and file I/O.",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": device,
            "threads": torch.get_num_threads(),
        },
    }
    if output is not None:
        write_json(Path(output) / "metrics.json", result)
    model.train(was_training)
    return result


def pretrain_segmenter(
    manifest: str,
    output: str,
    steps: int = 200,
    batch_size: int = 2,
    channels: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 17,
    threads: int = 2,
) -> dict:
    """Learn anatomy from TRAIN labels only; use validation CE for checkpoint choice."""
    _positive_integer(steps, "steps")
    _positive_integer(batch_size, "batch_size")
    _positive_integer(threads, "threads")
    _positive_integer(channels, "channels")
    _positive_float(lr, "lr")
    device = resolve_device(device)
    seed_everything(seed, threads)
    train_set, val_set = SubjectDataset(manifest, "train"), SubjectDataset(manifest, "val")
    if not len(train_set) or not len(val_set):
        raise ValueError("Segmenter training requires separate train and val subjects")
    if any("label" not in sample for sample in train_set):
        raise ValueError("Every prior-training subject requires a label volume")
    if any("label" not in sample for sample in val_set):
        raise ValueError("Every prior-validation subject requires a label volume")
    values = sorted({int(v) for sample in train_set for v in sample["label"].unique()})
    cfg = SegmenterConfig(len(values), channels)
    model = AuxiliarySegmenter(cfg).to(device, non_blocking=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Derive class weights ONLY from training voxels, with bounded inverse-frequency weights.
    counts = torch.zeros(len(values))
    for sample in train_set:
        y = encode_labels(sample["label"], values)
        counts += torch.bincount(y.flatten(), minlength=len(values))
    weights = (counts.sum() / counts.clamp_min(1)).sqrt()
    weights = (weights / weights.mean()).to(device, non_blocking=True)
    history, best, step = [], float("inf"), 0
    loader = _loader(train_set, batch_size, True, seed, device)
    while step < steps:
        for batch in loader:
            model.train()
            image = batch["image"].to(device, non_blocking=True)
            target = encode_labels(batch["label"][:, 0].to(device, non_blocking=True), values)
            logits = model(image)
            loss = F.cross_entropy(logits, target, weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite segmenter loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            step += 1
            if step % 20 == 0 or step == steps:
                model.eval()
                with torch.no_grad():
                    val_losses = [
                        float(
                            F.cross_entropy(
                                model(s["image"][None].to(device, non_blocking=True)),
                                encode_labels(
                                    s["label"][None, 0].to(device, non_blocking=True), values
                                ),
                            )
                        )
                        for s in val_set
                    ]
                val_loss = float(np.mean(val_losses))
                if not math.isfinite(val_loss):
                    raise FloatingPointError("Non-finite segmenter validation loss")
                row = {"step": step, "train_ce": float(loss.detach()), "val_ce": val_loss}
                history.append(row)
                print(json.dumps({"prior": row}), flush=True)
                if val_loss < best:
                    best = val_loss
                    save_checkpoint(
                        output,
                        {
                            "kind": "spear_segmenter",
                            "config": asdict(cfg),
                            "model": model.state_dict(),
                            "label_values": values,
                            "step": step,
                            "training_subject_ids": train_set.ids,
                            "validation_subject_ids": val_set.ids,
                            "best_val_ce": best,
                        },
                    )
            if step >= steps:
                break
    write_json(Path(output).with_suffix(".history.json"), history)
    return {"checkpoint": output, "steps": steps, "best_val_ce": best, "label_values": values}


def train_registration(
    config: dict,
    manifest: str,
    output: str,
    prior_path: str | None = None,
    resume: str | None = None,
) -> dict:
    """Train with image NCC + residual smoothness; choose by validation NCC.

    Checkpoints record whether their epoch is complete. Resuming a completed
    epoch starts the next one with an already advanced scheduler; resuming an
    incomplete epoch restarts that epoch's shuffled loader. The latter can
    repeat pairs, so resumption is not promised to match an uninterrupted run
    bit for bit. ``max_steps`` is a total budget, not an additional-step budget.
    Every run validates and saves its actual final model as ``last.pt``.
    """
    tc = config.get("training", {})
    seed = int(tc.get("seed", 17))
    threads = _positive_integer(tc.get("threads", 2), "threads")
    batch_size = _positive_integer(tc.get("batch_size", 1), "batch_size")
    epochs = _positive_integer(tc.get("epochs", 500), "epochs")
    max_steps = _positive_integer(tc.get("max_steps", 0), "max_steps", allow_zero=True)
    validate_every = _positive_integer(tc.get("validate_every", 50), "validate_every")
    lr_step_epochs = _positive_integer(tc.get("lr_step_epochs", 150), "lr_step_epochs")
    window = _positive_integer(tc.get("ncc_window", 9), "ncc_window")
    if window % 2 == 0:
        raise ValueError("ncc_window must be odd")
    lr = _positive_float(tc.get("lr", 1e-4), "lr")
    smooth_weight = _positive_float(tc.get("smooth_weight", 0.05), "smooth_weight", allow_zero=True)
    coarse_weight = _positive_float(tc.get("coarse_weight", 0.0), "coarse_weight", allow_zero=True)
    device = resolve_device(tc.get("device", "cuda:0"))
    seed_everything(seed, threads)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if not resume and any((out / name).exists() for name in ("best.pt", "last.pt")):
        raise ValueError(
            "Output already contains checkpoints; use --resume or a fresh output directory"
        )
    if resume and prior_path:
        raise ValueError(
            "A resume checkpoint already contains its prior; do not also supply --segmenter"
        )
    train_subjects = SubjectDataset(manifest, "train")
    val_subjects = SubjectDataset(manifest, "val")
    train_pairs, val_pairs = PairDataset(train_subjects), PairDataset(val_subjects)
    train_ids = train_subjects.ids
    val_ids = val_subjects.ids
    mc = SPEARConfig(**config.get("model", {}))
    prior, prior_meta = None, {}
    old = {}
    if resume:
        model, old = load_registration(resume, device)
        if old.get("inference_only"):
            raise ValueError("Cannot resume an inference-only export; use the original last.pt")
        if old["model_config"] != asdict(mc):
            raise ValueError("Resume configuration does not match the checkpoint")
        if (
            old.get("training_subject_ids") != train_ids
            or old.get("validation_subject_ids") != val_ids
        ):
            raise ValueError("Resume TRAIN or VAL split differs from the checkpoint")
        prior_meta = {
            "config": old.get("segmenter_config"),
            "label_values": old.get("segmenter_label_values"),
            "training_subject_ids": old.get("segmenter_training_subject_ids", []),
            "validation_subject_ids": old.get("segmenter_validation_subject_ids", []),
        }
    else:
        if mc.guidance == "semantic":
            if not prior_path:
                raise ValueError(
                    "Semantic SPEAR requires --segmenter; choose intensity explicitly for an ablation"
                )
            prior, prior_meta = load_prior(prior_path, device)
        model = SPEAR(mc, segmenter=prior).to(device, non_blocking=True)
    if mc.guidance == "semantic":
        prior_train = set(prior_meta.get("training_subject_ids", []))
        prior_val = set(prior_meta.get("validation_subject_ids", []))
        if not prior_train or not prior_train.issubset(set(train_ids)):
            raise ValueError(
                "Prior training IDs must be a nonempty subset of registration TRAIN IDs"
            )
        if not prior_val or not prior_val.issubset(set(val_ids)):
            raise ValueError(
                "Prior validation IDs must be a nonempty subset of registration VAL IDs; test-label selection is forbidden"
            )
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_epochs, gamma=0.5)
    if old:
        optimizer.load_state_dict(old["optimizer"])
        scheduler.load_state_dict(old["scheduler"])
    step = int(old.get("step", 0))
    start_epoch = (
        int(old.get("epoch", -1)) + 1 if old.get("epoch_complete", not old) else int(old["epoch"])
    )
    if start_epoch >= epochs or (max_steps and step >= max_steps):
        raise ValueError(
            "Resume budget is already exhausted; increase epochs/max_steps before requesting more training"
        )
    history = []
    if resume:
        history_path = Path(resume).parent / "history.json"
        if history_path.exists():
            saved_history = json.loads(history_path.read_text(encoding="utf-8"))
            if isinstance(saved_history, list):
                # Resuming an earlier best checkpoint must not keep records
                # from later updates that are no longer on this trajectory.
                history = [row for row in saved_history if int(row.get("step", 0)) <= step]
    best = float("inf")
    if resume:
        source_best = Path(resume).parent / "best.pt"
        # Carry a recoverable earlier best checkpoint into a new output folder.
        # If only a last checkpoint was copied, start best selection afresh;
        # its historical best scalar alone cannot recreate historical weights.
        if source_best.exists():
            best_checkpoint = torch.load(source_best, map_location="cpu", weights_only=True)
            if (
                best_checkpoint.get("model_config") == asdict(mc)
                and best_checkpoint.get("training_subject_ids") == train_ids
                and best_checkpoint.get("validation_subject_ids") == val_ids
                and int(best_checkpoint.get("step", 0)) <= step
            ):
                best = float(best_checkpoint["best_val_loss"])
                if source_best.resolve() != (out / "best.pt").resolve():
                    save_checkpoint(out / "best.pt", best_checkpoint)
        elif Path(resume).name == "best.pt":
            best = float(old["best_val_loss"])
            save_checkpoint(out / "best.pt", old)
    loader = _loader(train_pairs, batch_size, True, seed, device)
    start = time.perf_counter()
    write_json(
        out / "resolved_config.json",
        {
            **config,
            "training_shape": list(train_subjects[0]["image"].shape[1:]),
            "training_subject_ids": train_ids,
            "validation_subject_ids": val_ids,
        },
    )

    def validate_and_save(epoch: int, epoch_complete: bool, train_loss: float) -> None:
        nonlocal best
        # Evaluation observes validation labels only to report optional Dice.
        # The checkpoint decision below depends exclusively on image NCC.
        metrics = evaluate(model, val_pairs, device, window)
        val_loss = 1 - metrics["summary"]["ncc_after"]["mean"]
        if not math.isfinite(val_loss):
            raise FloatingPointError("Non-finite registration validation loss")
        row = {
            "step": step,
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)
        print(json.dumps({"registration": row}), flush=True)
        improved = val_loss < best
        best = min(best, val_loss)
        checkpoint = {
            "kind": "spear_registration",
            "model_config": asdict(mc),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "segmenter_config": prior_meta.get("config"),
            "segmenter_label_values": prior_meta.get("label_values"),
            "segmenter_training_subject_ids": prior_meta.get("training_subject_ids", []),
            "segmenter_validation_subject_ids": prior_meta.get("validation_subject_ids", []),
            "epoch": epoch,
            "epoch_complete": epoch_complete,
            "step": step,
            "validation_loss": val_loss,
            "best_val_loss": best,
            "config": config,
            "training_subject_ids": train_ids,
            "validation_subject_ids": val_ids,
            "training_shape": list(train_subjects[0]["image"].shape[1:]),
        }
        save_checkpoint(out / "last.pt", checkpoint)
        if improved:
            save_checkpoint(out / "best.pt", checkpoint)
        write_json(out / "history.json", history)

    for epoch in range(start_epoch, epochs):
        epoch_complete = False
        train_loss = 0.0
        for batch_index, batch in enumerate(loader):
            model.train()
            moving, fixed = (
                batch["moving"].to(device, non_blocking=True),
                batch["fixed"].to(device, non_blocking=True),
            )
            prediction = model(moving, fixed)
            terms = registration_loss(
                prediction,
                fixed,
                ncc_window=window,
                smooth_weight=smooth_weight,
                stage_weights=tc.get("stage_weights"),
                coarse_weight=coarse_weight,
            )
            loss = terms["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite registration loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5)
            optimizer.step()
            step += 1
            train_loss = float(loss.detach())
            epoch_complete = batch_index + 1 == len(loader)
            reached_budget = bool(max_steps and step >= max_steps)
            # At an epoch boundary, advance the scheduler before checkpointing.
            # At a mid-epoch stop it remains unchanged, matching resume logic.
            if not epoch_complete and not reached_budget and step % validate_every == 0:
                validate_and_save(epoch, False, train_loss)
            if reached_budget:
                break
        if epoch_complete:
            scheduler.step()
        final = epoch == epochs - 1 or bool(max_steps and step >= max_steps)
        if final or step % validate_every == 0:
            validate_and_save(epoch, epoch_complete, train_loss)
        if final:
            break
    return {"checkpoint": str(out / "best.pt"), "steps": step, "best_val_loss": best}
