"""Subject-disjoint 3-D MRI preparation, pairing, and NIfTI export.

PyTorch tensors use ``(C, D, H, W) = (C, z, y, x)``. NIfTI arrays and
voxel-to-world affines use ``(x, y, z)``. Resampling onto a shared world grid
is a preprocessing operation, **not** an estimated anatomical registration.
Perform anatomical affine prealignment before preparing the training tensors.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import operator
import re
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to
from torch.utils.data import Dataset

_SPLITS = frozenset({"train", "val", "test"})


def _load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, reporting the offending file for malformed input."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("subjects"), list):
        raise ValueError(f"{path}: expected an object containing a 'subjects' list")
    return payload


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _safe_name(value: str) -> str:
    """Keep readable identifiers while preventing path traversal/collisions."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:70] or "subject"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-{digest}"


def _validate_subjects(subjects: list[dict[str, Any]], root: Path, *, prepared: bool) -> None:
    seen_ids: set[str] = set()
    seen_paths: dict[Path, str] = {}
    for entry in subjects:
        if not isinstance(entry, dict):
            raise ValueError("Every subject must be a JSON object")
        subject_id = entry.get("id")
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("Each subject requires a nonempty string id")
        if subject_id in seen_ids:
            raise ValueError(f"Duplicate subject id: {subject_id}; splits must be subject-disjoint")
        seen_ids.add(subject_id)
        if entry.get("split") not in _SPLITS:
            raise ValueError(f"{subject_id}: split must be train, val, or test")
        key = "npz" if prepared else "image"
        if not isinstance(entry.get(key), str) or not entry[key]:
            raise ValueError(f"{subject_id}: missing {key} path")
        resolved = _resolve(entry[key], root)
        if resolved in seen_paths:
            raise ValueError(
                f"Image reused by {seen_paths[resolved]} and {subject_id}; subject leakage risk"
            )
        seen_paths[resolved] = subject_id
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
    if not any(entry["split"] == "train" for entry in subjects):
        raise ValueError("At least one training subject is required")


def _read_nifti(path: Path, *, label: bool = False) -> nib.Nifti1Image:
    """Validate the full input before resampling, including background voxels."""
    image = nib.load(str(path))
    if len(image.shape) != 3 or any(size < 2 for size in image.shape):
        raise ValueError(f"{path.name}: expected a 3-D volume with every dimension >= 2")
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError(f"{path.name}: affine must be finite and invertible")
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name}: volume contains NaN or infinite values")
    if label and (not np.equal(values, np.rint(values)).all()):
        raise ValueError(f"{path.name}: labels must contain integer anatomy identifiers")
    # NIfTI affines are expressed in header spatial units. Convert known
    # non-mm units before comparing world coordinates across subjects. The
    # usual MRI default for an unspecified unit is assumed to be millimeters.
    spatial_units = image.header.get_xyzt_units()[0]
    unit_scale = {"meter": 1000.0, "micron": 0.001}.get(spatial_units, 1.0)
    if unit_scale != 1.0:
        affine = affine.copy()
        affine[:3, :] *= unit_scale
        image = nib.Nifti1Image(values, affine)
        image.header.set_xyzt_units("mm")
    # Canonicalization permutes/flips axes and adjusts the affine. It does not
    # estimate an anatomical alignment between different subjects.
    return nib.as_closest_canonical(image)


def _resample_to_grid(image, reference_grid, *, order):
    """Avoid padding artifacts from header rounding on an already shared grid."""
    reference_shape, reference_affine = reference_grid
    if image.shape == tuple(reference_shape) and np.allclose(
        image.affine, reference_affine, atol=1e-5, rtol=1e-6
    ):
        return image
    return resample_from_to(image, reference_grid, order=order, mode="constant", cval=0.0)


def _normalize(image: np.ndarray) -> np.ndarray:
    """Normalize one image independently, without labels or dataset statistics."""
    foreground = image[image != 0]
    if foreground.size == 0:
        return np.zeros(image.shape, dtype=np.float32)
    low, high = np.percentile(foreground, [1.0, 99.0])
    if high <= low:
        # Constant foreground (e.g. a synthetic binary object) still retains
        # its contrast to zero-valued background instead of becoming empty.
        result = np.zeros(image.shape, dtype=np.float32)
        result[image != 0] = 1.0
        return result
    result = np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)
    result[image == 0] = 0.0  # Retain zero padding even for negative intensities.
    return result


def _resize_xyz(values: np.ndarray, shape_dhw: tuple[int, int, int], *, label: bool) -> np.ndarray:
    """Resize on voxel-center endpoints, consistent with the exported affine."""
    tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(2, 1, 0))).float()[None, None]
    if label:
        # interpolate(mode='nearest') has a different sampling convention from
        # align_corners=True. An explicit nearest grid keeps label centers and
        # image centers geometrically consistent while preserving label IDs.
        z, y, x = [torch.linspace(-1.0, 1.0, n) for n in shape_dhw]
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        grid = torch.stack((xx, yy, zz), dim=-1)[None]
        result = F.grid_sample(
            tensor, grid, mode="nearest", padding_mode="border", align_corners=True
        )
        return result[0, 0].numpy().round().astype(np.int64)
    result = F.interpolate(tensor, size=shape_dhw, mode="trilinear", align_corners=True)
    return result[0, 0].numpy().astype(np.float32)


def prepare_dataset(
    manifest_path: str | Path,
    output_dir: str | Path,
    shape: Sequence[int] = (36, 36, 36),
    reference: str | Path | None = None,
) -> Path:
    """Prepare NIfTI subjects into portable, pickle-free compressed NPZ files.

    ``shape`` is the final tensor shape ``(D,H,W)``. The shared grid is taken
    from an explicit reference or the first **training** image, never a test
    subject. All volumes are first resampled in physical coordinates onto
    that canonical RAS grid (linear images, nearest labels), then resized to
    the requested shape. Labels are evaluation targets; they never determine
    image scaling or reference selection. Their original anatomy IDs remain.

    Input JSON example::

        {"subjects": [{"id": "s01", "image": "s01.nii.gz",
                       "label": "s01_seg.nii.gz", "split": "train"}]}

    Paths are interpreted relative to the input manifest. Returned manifest
    paths are relative to the prepared output directory for easy relocation.
    """
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if len(shape) != 3 or any(not isinstance(n, (int, np.integer)) or n < 2 for n in shape):
        raise ValueError("shape must contain three integer DHW dimensions >= 2")
    shape_dhw = tuple(int(n) for n in shape)
    payload = _load_json(manifest_path)
    subjects = payload["subjects"]
    _validate_subjects(subjects, manifest_path.parent, prepared=False)
    train_reference = next(entry for entry in subjects if entry["split"] == "train")
    reference_path = (
        _resolve(str(reference), manifest_path.parent)
        if reference is not None
        else _resolve(train_reference["image"], manifest_path.parent)
    )
    reference_image = _read_nifti(reference_path)
    reference_grid = (reference_image.shape, reference_image.affine)
    # The transpose used for torch does not alter the NIfTI affine's XYZ
    # coordinate convention. Endpoint scaling follows align_corners=True.
    shape_xyz = np.asarray(shape_dhw[::-1])
    scale = (np.asarray(reference_image.shape) - 1) / (shape_xyz - 1)
    target_affine = reference_image.affine @ np.diag([*scale, 1.0])
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_subjects = []
    for subject in subjects:
        source_path = _resolve(subject["image"], manifest_path.parent)
        image = _read_nifti(source_path)
        resampled = _resample_to_grid(image, reference_grid, order=1)
        values = resampled.get_fdata(dtype=np.float32)
        image_dhw = _resize_xyz(_normalize(values), shape_dhw, label=False)
        arrays: dict[str, np.ndarray] = {
            "image": image_dhw,
            "affine": target_affine.astype(np.float64),
        }
        label_name = None
        if subject.get("label") is not None:
            if not isinstance(subject["label"], str) or not subject["label"]:
                raise ValueError(f"{subject['id']}: label must be a nonempty path when provided")
            label_path = _resolve(subject["label"], manifest_path.parent)
            labels = _read_nifti(label_path, label=True)
            labels = _resample_to_grid(labels, reference_grid, order=0)
            arrays["label"] = _resize_xyz(labels.get_fdata(dtype=np.float32), shape_dhw, label=True)
            label_name = label_path.name
        filename = _safe_name(subject["id"]) + ".npz"
        np.savez_compressed(output_dir / filename, **arrays)
        item: dict[str, Any] = {
            "id": subject["id"],
            "split": subject["split"],
            "npz": filename,
            "source_image": source_path.name,
        }
        if label_name is not None:
            item["source_label"] = label_name
        prepared_subjects.append(item)
    prepared_payload = {
        "format_version": 1,
        "shape_dhw": list(shape_dhw),
        "affine_xyz": target_affine.tolist(),
        "tensor_axis_order": "DHW=zyx",
        "spatial_units": "mm (unspecified input units are assumed mm)",
        "reference": {
            "source": reference_path.name,
            "selection": "explicit" if reference is not None else "first_training_subject",
        },
        "preprocessing": "Canonical RAS world-grid resampling, not anatomical affine registration; per-image nonzero p1/p99 intensity scaling; endpoint-aligned resize",
        "subjects": prepared_subjects,
    }
    result = output_dir / "manifest.json"
    result.write_text(json.dumps(prepared_payload, indent=2) + "\n", encoding="utf-8")
    return result


class SubjectDataset(Dataset):
    """Load prepared subjects on demand with a bounded CPU LRU cache.

    Every file is validated sequentially at construction, but only the most
    recent ``cache_size`` subjects are retained. The default of eight keeps
    the small demonstration cohort hot without loading a full MRI cohort
    into host memory. ``cache_size=0`` disables retention. Each DataLoader
    worker has its own cache, so account for worker count when sizing it.

    ``subjects`` contains lightweight manifest metadata, **not** volume
    tensors. ``ids`` permits provenance collection without loading volumes.
    Indexing still returns the same image/label/affine tensor dictionary.
    Cached tensors are read-only by convention; clone before augmentation.
    """

    def __init__(
        self, manifest_path: str | Path, split: str = "train", cache_size: int = 8
    ) -> None:
        if split not in _SPLITS:
            raise ValueError("split must be train, val, or test")
        if (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, (int, np.integer))
            or cache_size < 0
        ):
            raise ValueError("cache_size must be a nonnegative integer")
        manifest_path = Path(manifest_path).resolve()
        payload = _load_json(manifest_path)
        _validate_subjects(payload["subjects"], manifest_path.parent, prepared=True)
        self.split = split
        self.cache_size = int(cache_size)
        self._root = manifest_path.parent
        self.subjects = [
            dict(subject) for subject in payload["subjects"] if subject["split"] == split
        ]
        if not self.subjects:
            raise ValueError(f"No subjects in requested split: {split}")
        self.ids = [subject["id"] for subject in self.subjects]
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._common_shape: tuple[int, ...] | None = None
        self._common_affine: np.ndarray | None = None
        # Validate the entire split before training. Loading one file at a
        # time limits residency to the cache plus the currently read subject.
        for index in range(len(self.subjects)):
            item = self._load_subject(index)
            self.subjects[index]["has_label"] = "label" in item
            self._remember(index, item)

    def _load_subject(self, index: int) -> dict[str, Any]:
        subject = self.subjects[index]
        with np.load(_resolve(subject["npz"], self._root), allow_pickle=False) as arrays:
            image = np.asarray(arrays["image"], dtype=np.float32)
            affine = np.asarray(arrays["affine"], dtype=np.float64)
            if image.ndim != 3 or not np.isfinite(image).all():
                raise ValueError(f"{subject['id']}: prepared image must be finite DHW")
            if (
                affine.shape != (4, 4)
                or not np.isfinite(affine).all()
                or abs(np.linalg.det(affine[:3, :3])) < 1e-12
            ):
                raise ValueError(f"{subject['id']}: invalid prepared affine")
            if self._common_shape is not None and (
                image.shape != self._common_shape
                or not np.allclose(affine, self._common_affine, atol=1e-5, rtol=1e-6)
            ):
                raise ValueError(
                    "Subjects must share the same prepared shape and physical reference grid"
                )
            if self._common_shape is None:
                self._common_shape = image.shape
                self._common_affine = affine.copy()
            item = {
                "id": subject["id"],
                "image": torch.from_numpy(image.copy())[None],
                "affine": torch.from_numpy(affine.copy()),
            }
            if "label" in arrays:
                label = np.asarray(arrays["label"])
                if (
                    label.shape != image.shape
                    or not np.isfinite(label).all()
                    or not np.equal(label, np.rint(label)).all()
                ):
                    raise ValueError(
                        f"{subject['id']}: labels must be finite integer DHW on the image grid"
                    )
                item["label"] = torch.from_numpy(label.astype(np.int64, copy=True))[None]
        return item

    def _remember(self, index: int, item: dict[str, Any]) -> None:
        if self.cache_size == 0:
            return
        self._cache[index] = item
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Subject index out of range")
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        item = self._load_subject(index)
        self._remember(index, item)
        return item


class _PairIndices(Sequence[tuple[int, int]]):
    """Constant-memory sequence with the original ordered non-self ordering."""

    def __init__(self, subject_count: int) -> None:
        self.subject_count = subject_count

    def __len__(self) -> int:
        return self.subject_count * (self.subject_count - 1)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Pair index out of range")
        moving, fixed = divmod(index, self.subject_count - 1)
        if fixed >= moving:
            fixed += 1
        return moving, fixed


class PairDataset(Dataset):
    """Enumerate within-split non-self pairs without materializing O(N²) tuples."""

    def __init__(self, subjects: SubjectDataset) -> None:
        if len(subjects) < 2:
            raise ValueError(
                f"At least two {subjects.split} subjects are required for non-self pairs"
            )
        self.subjects = subjects
        self.pairs = _PairIndices(len(subjects))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        moving_index, fixed_index = self.pairs[index]
        moving, fixed = self.subjects[moving_index], self.subjects[fixed_index]
        result = {
            "moving": moving["image"],
            "fixed": fixed["image"],
            "moving_id": moving["id"],
            "fixed_id": fixed["id"],
            "affine": fixed["affine"],
        }
        # Return label pairs together; callers can safely perform unlabeled
        # training or evaluate only pairs for which both labels are available.
        if "label" in moving and "label" in fixed:
            result["moving_label"] = moving["label"]
            result["fixed_label"] = fixed["label"]
        return result


def _single_tensor(value: torch.Tensor, channels: int, name: str) -> np.ndarray:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name}: export one pair at a time (batch size must be one)")
        tensor = tensor[0]
    if tensor.ndim != 4 or tensor.shape[0] != channels:
        raise ValueError(f"{name}: expected ({channels},D,H,W) or (1,{channels},D,H,W)")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name}: cannot export non-finite values")
    return tensor.float().numpy()


def save_prediction(
    output_dir: str | Path,
    moving_id: str,
    fixed_id: str,
    warped: torch.Tensor,
    flow: torch.Tensor,
    affine: torch.Tensor | np.ndarray,
) -> dict[str, Path]:
    """Export a single prediction with explicit physical and flow conventions.

    ``flow`` is a backward sampling displacement in target-grid voxels with
    channels ``(dz,dy,dx)``: ``warped(x)=moving(x+flow(x))``. Its NIfTI vector
    components are reordered to ``(dx,dy,dz)`` on an ``(X,Y,Z,3)`` array.
    These are voxel displacements, **not** physical-mm vectors or an inverse
    field. The sidecar states the convention to prevent accidental misuse.
    """
    image = _single_tensor(warped, 1, "warped")
    field = _single_tensor(flow, 3, "flow")
    if image.shape[1:] != field.shape[1:]:
        raise ValueError("warped and flow must have the same spatial shape")
    affine_array = np.asarray(torch.as_tensor(affine).detach().cpu(), dtype=np.float64)
    if affine_array.shape == (1, 4, 4):
        affine_array = affine_array[0]
    if (
        affine_array.shape != (4, 4)
        or not np.isfinite(affine_array).all()
        or abs(np.linalg.det(affine_array[:3, :3])) < 1e-12
    ):
        raise ValueError("affine must be a finite invertible 4x4 voxel-XYZ-to-world matrix")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_safe_name(moving_id)}_to_{_safe_name(fixed_id)}"
    paths = {
        "warped": output_dir / f"{prefix}_warped.nii.gz",
        "flow": output_dir / f"{prefix}_flow_voxel.nii.gz",
        "metadata": output_dir / f"{prefix}_flow_voxel.json",
    }
    xyz_image = np.ascontiguousarray(image[0].transpose(2, 1, 0))
    xyz_field = np.ascontiguousarray(field[[2, 1, 0]].transpose(3, 2, 1, 0))
    for key, array in (("warped", xyz_image), ("flow", xyz_field)):
        nifti = nib.Nifti1Image(array.astype(np.float32), affine_array)
        nifti.header.set_xyzt_units("mm")
        nifti.set_sform(affine_array, code=1)
        # qform cannot represent shear. The exact affine lives in sform; do
        # not publish a conflicting approximate qform for an oblique grid.
        linear = affine_array[:3, :3]
        directions = linear / np.linalg.norm(linear, axis=0, keepdims=True)
        if np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
            nifti.set_qform(affine_array, code=1)
        else:
            nifti.set_qform(None, code=0)
        if key == "flow":
            nifti.header.set_intent("vector", name="voxel displacement")
        nifti.header["descrip"] = b"SPEAR common-grid backward warp; see JSON sidecar"
        nib.save(nifti, str(paths[key]))
    metadata = {
        "moving_id": moving_id,
        "fixed_id": fixed_id,
        "flow_convention": "backward sampling: warped(x,y,z) = moving((x,y,z) + flow(x,y,z))",
        "array_layout": "X,Y,Z,3",
        "vector_components": ["dx", "dy", "dz"],
        "displacement_units": "voxels of the exported fixed/reference grid (not millimeters)",
        "grid": "Warped image and flow use the exported fixed/reference grid",
        "affine_xyz": affine_array.tolist(),
        "warped_image": paths["warped"].name,
        "flow_image": paths["flow"].name,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return paths
