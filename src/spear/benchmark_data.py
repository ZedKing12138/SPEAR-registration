"""Reproducible preparation of the three SPEAR benchmark datasets.

The pipeline estimates 12-DOF affine alignment with SimpleITK Mattes mutual
information, uses a reference from the training split, and applies per-volume
p1/p99 intensity scaling through :mod:`spear.data`. Dataset presets define the
output sizes. Supplied subject splits are preserved; otherwise a seeded
subject-wise 7:1:2 partition is generated. Anatomical labels are read from the
input manifest or a supplied label directory. IXI/OASIS anatomy labels must
be supplied separately when needed.

All shapes below are PyTorch ``(D,H,W)``; NIfTI affines remain XYZ/world-mm.
Optional per-subject ``mask`` files remove non-brain voxels before alignment.
Masks must be prepared independently; evaluation anatomy labels never supply
registration masks. Analyze, NIfTI, and FreeSurfer MGH/MGZ volumes are accepted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

from .data import _normalize, _safe_name, _validate_subjects, prepare_dataset

DATASET_SHAPES = {
    "ixi": (144, 144, 144),
    "oasis": (162, 198, 162),
    "ibsr18": (225, 225, 225),
}
_EXTENSIONS = (".nii.gz", ".nii", ".img.gz", ".img", ".mgz", ".mgh")


def _dataset_name(dataset: str) -> str:
    name = dataset.lower()
    if name not in DATASET_SHAPES:
        raise ValueError(f"dataset must be one of {', '.join(DATASET_SHAPES)}")
    return name


def _volume_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name.endswith(_EXTENSIONS))


def _stem(path: Path) -> str:
    for extension in _EXTENSIONS:
        if path.name.endswith(extension):
            return path.name[: -len(extension)]
    return path.stem


def _choose(candidates: list[Path], description: str) -> Path:
    """An ambiguous image/label must be resolved in an explicit manifest."""
    if len(candidates) != 1:
        raise ValueError(
            f"{description}: expected exactly one volume, found {len(candidates)}; "
            "use a manifest with explicit image/label paths"
        )
    return candidates[0]


def _external_label(subject: str, image: Path, files: list[Path]) -> Path | None:
    # A dedicated label directory is explicit user input. Exact subject
    # boundaries prevent IXI001 accidentally matching IXI0010, for example.
    boundary = re.compile(rf"(?<![A-Za-z0-9]){re.escape(subject)}(?![A-Za-z0-9])", re.I)
    candidates = [p for p in files if boundary.search(str(p)) or _stem(p) == _stem(image)]
    # An OASIS FreeSurfer release may also contain MR2 retests. A label from
    # another visit is not interchangeable with the selected MR1 anatomy.
    session = re.search(r"OAS1_\d{4}_MR\d+", image.name, flags=re.I)
    if session:
        candidates = [
            p
            for p in candidates
            if (match := re.search(r"OAS1_\d{4}_MR\d+", str(p), flags=re.I)) is None
            or match[0].upper() == session[0].upper()
        ]
    # FreeSurfer commonly includes many label volumes. Prefer aparc+aseg if it
    # is the unique such volume; otherwise require an explicit manifest.
    aparc = [p for p in candidates if _stem(p) == "aparc+aseg"]
    if len(aparc) == 1:
        return aparc[0]
    return _choose(candidates, f"{subject} label") if candidates else None


def discover_subjects(
    dataset: str, raw_dir: str | Path, labels_dir: str | Path | None = None
) -> list[dict[str, Any]]:
    """Discover official archive layouts, returning absolute image paths.

    IXI uses ``IXI*-T1.nii[.gz]``. OASIS-1 uses the averaged, skull-stripped
    ``PROCESSED/MPRAGE/T88_111/*_masked_gfc.img`` release (NIfTI conversions
    are also accepted), selecting only MR1 to exclude retest sessions. The
    unmasked ``*_t88_gfc`` average is a fallback; raw repeat acquisitions
    are intentionally not arbitrarily collapsed. IBSR18 accepts both
    ``IBSR_XX_ana.nii.gz`` and original Analyze layouts. Its ``seg_ana``
    anatomical segmentation is preferred; ``segTRI`` tissue maps are never
    silently substituted. If a release/layout differs, provide a manifest.

    ``labels_dir`` supplies independently obtained labels, including
    FreeSurfer ``subject/mri/aparc+aseg.mgz``. Labels are optional here but
    required later for supervised auxiliary-segmenter pretraining and Dice.
    No data download or license acceptance happens inside this function.
    """
    dataset = _dataset_name(dataset)
    raw_dir = Path(raw_dir).expanduser().resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(raw_dir)
    images = _volume_files(raw_dir)
    external = None
    if labels_dir is not None:
        label_root = Path(labels_dir).expanduser().resolve()
        if not label_root.is_dir():
            raise NotADirectoryError(label_root)
        external = _volume_files(label_root)
    grouped: dict[str, list[Path]] = {}
    if dataset == "ixi":
        for path in images:
            match = re.fullmatch(r"(IXI\d{3})-.+-T1", _stem(path), flags=re.I)
            if match:
                grouped.setdefault(match[1].upper(), []).append(path)
    elif dataset == "ibsr18":
        for path in images:
            match = re.fullmatch(r"(IBSR_\d{2})_ana(?:_strip)?", _stem(path), flags=re.I)
            if match:
                grouped.setdefault(match[1].upper(), []).append(path)
        # The official stripped release may coexist with its unstripped copy.
        # Prefer the intended brain image, while retaining strict ambiguity
        # detection for duplicate conversions of the same volume.
        for subject, candidates in grouped.items():
            stripped = [p for p in candidates if _stem(p).lower().endswith("_ana_strip")]
            if stripped:
                grouped[subject] = stripped
    else:
        oasis: dict[str, list[tuple[int, int, Path]]] = {}
        for path in images:
            # Official processed averages share the subject/session prefix.
            match = re.match(r"(OAS1_\d{4})_MR(\d+)_", _stem(path), flags=re.I)
            if not match or int(match[2]) != 1:
                continue
            name = _stem(path).lower()
            if name.endswith("_masked_gfc"):
                rank = 0
            elif name.endswith("_t88_gfc"):
                rank = 1
            else:
                continue
            oasis.setdefault(match[1].upper(), []).append((int(match[2]), rank, path))
        for subject, records in oasis.items():
            session = min(r[0] for r in records)  # MR1 normally; no retest leakage.
            rank = min(r[1] for r in records if r[0] == session)
            grouped[subject] = [p for s, r, p in records if s == session and r == rank]
    if not grouped:
        raise ValueError(
            f"No recognized {dataset} images in {raw_dir}; provide a manifest for custom layouts"
        )
    subjects = []
    for subject, candidates in sorted(grouped.items()):
        image = _choose(candidates, f"{subject} image")
        entry: dict[str, Any] = {"id": subject, "image": str(image)}
        label = None
        if external is not None:
            label = _external_label(subject, image, external)
        elif dataset == "ibsr18":
            labels = [p for p in images if _stem(p).lower() == f"{subject}_seg_ana".lower()]
            label = _choose(labels, f"{subject} anatomy label") if labels else None
        if label is not None:
            entry["label"] = str(label)
        subjects.append(entry)
    return subjects


def _person_key(dataset: str, entry: dict[str, Any]) -> str:
    # OASIS retest identifiers must remain together even for a custom manifest.
    # A caller-specified subject_id can group other longitudinal releases too.
    if dataset == "oasis":
        match = re.search(r"OAS1_\d{4}", entry["id"], flags=re.I)
        if match is None:
            match = re.search(r"OAS1_\d{4}", Path(entry["image"]).name, flags=re.I)
        if match:
            return match[0].upper()
    return str(entry.get("subject_id", entry["id"]))


def assign_subject_splits(
    dataset: str, subjects: list[dict[str, Any]], seed: int = 17
) -> list[dict[str, Any]]:
    """Preserve complete supplied splits, or deterministically assign 7:1:2.

    Two subjects per split are the minimum for all ordered non-self pairs;
    hence auto-splitting requires >=6 people. Rounding reserves the validation
    and test counts first, giving IBSR18 exactly 12/2/4. Sessions are grouped
    by person before splitting. Partially specified splits are rejected.
    """
    dataset = _dataset_name(dataset)
    records = [dict(s) for s in subjects]
    if not records:
        raise ValueError("The manifest contains no subjects")
    if any(not isinstance(s.get("id"), str) or not s.get("image") for s in records):
        raise ValueError("Every subject requires a nonempty id and image path")
    specified = ["split" in s for s in records]
    if any(specified) and not all(specified):
        raise ValueError("Supply splits for all subjects or none of them")
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in records:
        groups.setdefault(_person_key(dataset, entry), []).append(entry)
    if all(specified):
        for person, visits in groups.items():
            splits = {s["split"] for s in visits}
            if not splits <= {"train", "val", "test"} or len(splits) != 1:
                raise ValueError(f"{person}: invalid split or subject leakage across sessions")
        return records
    people = sorted(groups)
    if len(people) < 6:
        raise ValueError("Automatic pairwise splitting needs at least six people; supply splits")
    np.random.default_rng(seed).shuffle(people)
    n_test = max(2, int(round(0.2 * len(people))))
    n_val = max(2, int(round(0.1 * len(people))))
    n_train = len(people) - n_test - n_val
    for index, person in enumerate(people):
        split = "train" if index < n_train else "val" if index < n_train + n_val else "test"
        for entry in groups[person]:
            entry["split"] = split
    return records


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _load_volume(path: Path, *, label: bool = False) -> nib.Nifti1Image:
    """Convert supported formats and units without guessing a new orientation."""
    source = nib.load(str(path))
    values = source.get_fdata(dtype=np.float32)
    # Several Analyze releases encode a single frame as a 4-D volume.
    while values.ndim > 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3 or any(n < 2 for n in values.shape):
        raise ValueError(f"{path}: expected a 3-D volume (singleton trailing axes are allowed)")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: volume contains nonfinite values")
    if label and not np.equal(values, np.rint(values)).all():
        raise ValueError(f"{path}: anatomy labels must be integers")
    affine = np.asarray(source.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError(f"{path}: invalid spatial affine")
    units = source.header.get_xyzt_units()[0] if hasattr(source.header, "get_xyzt_units") else "mm"
    affine = affine.copy()
    affine[:3, :] *= {"meter": 1000.0, "micron": 0.001}.get(units, 1.0)
    converted = nib.as_closest_canonical(nib.Nifti1Image(values, affine))
    # ITK directions must be orthonormal. Resample shear in physical space;
    # never discard it by merely rewriting a NIfTI header.
    spacing = np.linalg.norm(converted.affine[:3, :3], axis=0)
    directions = converted.affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
        converted = resample_to_output(converted, voxel_sizes=spacing, order=0 if label else 1)
    converted.header.set_xyzt_units("mm")
    return converted


def _sitk_module():
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "Affine preparation requires SimpleITK; install with pip install '.[datasets]'"
        ) from error
    return sitk


def _to_sitk(volume: nib.Nifti1Image, *, normalize: bool = False):
    """Bridge canonical NIfTI RAS/XYZ to ITK LPS/ZYX without flipping anatomy."""
    sitk = _sitk_module()
    values = volume.get_fdata(dtype=np.float32)
    if normalize:
        values = _normalize(values)
    image = sitk.GetImageFromArray(np.ascontiguousarray(values.transpose(2, 1, 0)))
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    matrix = ras_to_lps @ volume.affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    image.SetSpacing(tuple(float(x) for x in spacing))
    image.SetDirection(tuple(float(x) for x in (matrix / spacing).ravel()))
    image.SetOrigin(tuple(float(x) for x in ras_to_lps @ volume.affine[:3, 3]))
    return image


def _estimate_affine(fixed, moving, *, seed: int, threads: int):
    """Estimate a fixed-to-moving pullback transform with a 12-DOF affine."""
    sitk = _sitk_module()
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    registration = sitk.ImageRegistrationMethod()
    registration.SetNumberOfThreads(threads)
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.REGULAR)
    registration.SetMetricSamplingPercentage(0.3, int(seed))
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=0.5,
        numberOfIterations=150,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=12,
        estimateLearningRate=registration.EachIteration,
        maximumStepSizeInPhysicalUnits=1.0,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    factors = [4, 2, 1] if min(fixed.GetSize()) >= 32 else [2, 1]
    registration.SetShrinkFactorsPerLevel(factors)
    registration.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0] if len(factors) == 3 else [1.0, 0.0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial, inPlace=True)
    registration.Execute(fixed, moving)
    matrix = np.asarray(initial.GetMatrix()).reshape(3, 3)
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(initial.GetParameters()).all() or determinant <= 0:
        raise RuntimeError("Affine optimization returned a nonfinite or reflected transform")
    return initial, {
        "metric": "Mattes mutual information (32 bins)",
        "metric_value": float(registration.GetMetricValue()),
        "optimizer_stop": registration.GetOptimizerStopConditionDescription(),
        "affine_determinant": determinant,
        "sampling": "regular 30%, seeded",
        "shrink_factors": factors,
    }


def prepare_benchmark(
    dataset: str,
    manifest_path: str | Path,
    output: str | Path,
    shape: Sequence[int] | None = None,
    seed: int = 17,
    affine: bool = True,
    reference: str | Path | None = None,
    threads: int = 2,
) -> Path:
    """Prepare a benchmark manifest, anatomy-affine alignment, and NPZ tensors.

    The input is ``{"subjects": [{"id": ..., "image": ..., "label": ...}]}``;
    paths are relative to the manifest. Splits may be supplied for every entry
    or omitted for automatic subject-wise splitting. Optional ``mask`` removes
    non-brain signal before alignment. Optional ``label_shares_image_grid``
    explicitly repairs known Analyze label-header inconsistencies only when
    original image/label array shapes match. This is never inferred silently.

    By default the lexicographically first TRAIN subject defines the reference.
    ``reference`` may name a TRAIN subject or a volume path. A path corresponding
    to a validation/test image is rejected; a genuinely external template is
    allowed and recorded explicitly. No annotations enter affine optimization.
    ``affine=False`` means the caller's images must already be prealigned; it
    does not relabel mere world-grid resampling as anatomical registration.

    Aligned NIfTI images, optional nearest-neighbor labels, ITK ``.tfm`` files,
    split manifest, and provenance are written beneath ``output/preprocessing``.
    These are local generated dataset artifacts, never repository fixtures.
    """
    dataset = _dataset_name(dataset)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    target_shape = tuple(DATASET_SHAPES[dataset] if shape is None else shape)
    if len(target_shape) != 3 or any(
        not isinstance(n, (int, np.integer)) or n < 2 for n in target_shape
    ):
        raise ValueError("shape must contain three integer DHW dimensions >= 2")
    manifest_path = Path(manifest_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("subjects"), list):
        raise ValueError("Expected a JSON object with a subjects list")
    records = assign_subject_splits(dataset, payload["subjects"], seed)
    _validate_subjects(records, manifest_path.parent, prepared=False)
    for entry in records:
        for key in ("image", "label", "mask"):
            if entry.get(key) is not None:
                path = _resolve(entry[key], manifest_path.parent)
                if not path.is_file():
                    raise FileNotFoundError(path)
                entry[key] = str(path)
    training = sorted((s for s in records if s["split"] == "train"), key=lambda s: s["id"])
    reference_entry = training[0]
    reference_path = Path(reference_entry["image"])
    reference_selection = "lexicographically_first_training_subject"
    if reference is not None:
        matches = [s for s in records if s["id"] == str(reference)]
        if matches:
            reference_path = Path(matches[0]["image"])
        else:
            reference_path = _resolve(reference, manifest_path.parent)
        matches = [s for s in records if Path(s["image"]) == reference_path]
        if matches and matches[0]["split"] != "train":
            raise ValueError("A validation/test subject cannot be the affine reference")
        reference_entry = matches[0] if matches else None
        reference_selection = "explicit_training_subject" if matches else "external_template"
    if (output / "manifest.json").exists():
        raise FileExistsError(
            f"{output}/manifest.json already exists; choose a new output directory"
        )
    work = output / "preprocessing"
    work.mkdir(parents=True, exist_ok=True)

    def load_image(entry):
        image = _load_volume(Path(entry["image"]))
        if entry.get("mask"):
            mask = _load_volume(Path(entry["mask"]))
            mask = resample_from_to(mask, (image.shape, image.affine), order=0)
            values = image.get_fdata(dtype=np.float32)
            values[mask.get_fdata(dtype=np.float32) <= 0] = 0
            image = nib.Nifti1Image(values, image.affine)
            image.header.set_xyzt_units("mm")
        if not np.any(image.get_fdata(dtype=np.float32)):
            raise ValueError(f"{entry['id']}: image is empty after optional masking")
        return image

    fixed_nib = load_image(reference_entry) if reference_entry else _load_volume(reference_path)
    fixed_path = work / "reference.nii.gz"
    nib.save(fixed_nib, fixed_path)
    sitk = _sitk_module() if affine else None
    fixed_sitk = _to_sitk(fixed_nib, normalize=True) if affine else None
    aligned = []
    transforms = []
    for index, entry in enumerate(records):
        image = load_image(entry)
        slug = _safe_name(entry["id"])
        label = _load_volume(Path(entry["label"]), label=True) if entry.get("label") else None
        repaired_label_header = False
        if label is not None and entry.get("label_shares_image_grid", False):
            # Check original arrays before canonicalization; incompatible
            # shapes cannot be repaired by pretending the grids coincide.
            original_image = nib.load(entry["image"])
            original_label = nib.load(entry["label"])
            values = np.squeeze(original_label.get_fdata(dtype=np.float32))
            original_shape = tuple(n for n in original_image.shape[:3])
            if values.shape != original_shape:
                raise ValueError(f"{entry['id']}: shared-grid label shape differs from image")
            repaired = work / f"{slug}_label_header_repaired.nii.gz"
            repaired_image = nib.Nifti1Image(values, original_image.affine)
            if hasattr(original_image.header, "get_xyzt_units"):
                repaired_image.header.set_xyzt_units(original_image.header.get_xyzt_units()[0])
            nib.save(repaired_image, repaired)
            label = _load_volume(repaired, label=True)
            repaired_label_header = True
        transform_info: dict[str, Any] = {"id": entry["id"], "split": entry["split"]}
        if affine:
            moving_sitk = _to_sitk(image, normalize=True)
            if Path(entry["image"]) == reference_path:
                transform = sitk.AffineTransform(3)
                details = {"reference_identity": True, "affine_determinant": 1.0}
            else:
                try:
                    transform, details = _estimate_affine(
                        fixed_sitk, moving_sitk, seed=seed + index, threads=threads
                    )
                except RuntimeError as error:
                    raise RuntimeError(
                        f"Affine alignment failed for {entry['id']}: {error}"
                    ) from error
            transform_path = work / f"{slug}.tfm"
            sitk.WriteTransform(transform, str(transform_path))
            transformed = sitk.Resample(
                _to_sitk(image), fixed_sitk, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
            )
            values = sitk.GetArrayFromImage(transformed).transpose(2, 1, 0)
            if not np.isfinite(values).all() or not np.any(values):
                raise RuntimeError(f"{entry['id']}: affine result is empty or nonfinite")
            image = nib.Nifti1Image(values, fixed_nib.affine)
            if label is not None:
                transformed_label = sitk.Resample(
                    _to_sitk(label),
                    fixed_sitk,
                    transform,
                    sitk.sitkNearestNeighbor,
                    0.0,
                    sitk.sitkFloat32,
                )
                values = sitk.GetArrayFromImage(transformed_label).transpose(2, 1, 0)
                label = nib.Nifti1Image(np.rint(values).astype(np.int32), fixed_nib.affine)
            transform_info.update(details)
            transform_info["transform"] = transform_path.name
        else:
            transform_info["anatomical_alignment"] = (
                "skipped; caller asserts prior affine alignment"
            )
        image.header.set_xyzt_units("mm")
        image_path = work / f"{slug}_image.nii.gz"
        nib.save(image, image_path)
        prepared_entry = {"id": entry["id"], "split": entry["split"], "image": image_path.name}
        if label is not None:
            label.header.set_xyzt_units("mm")
            label_path = work / f"{slug}_label.nii.gz"
            nib.save(label, label_path)
            prepared_entry["label"] = label_path.name
        aligned.append(prepared_entry)
        transform_info.update(
            {
                "source_image": Path(entry["image"]).name,
                "source_label": Path(entry["label"]).name if entry.get("label") else None,
                "source_mask": Path(entry["mask"]).name if entry.get("mask") else None,
                "label_header_explicitly_repaired": repaired_label_header,
            }
        )
        transforms.append(transform_info)
    aligned_manifest = work / "aligned_manifest.json"
    aligned_manifest.write_text(
        json.dumps({"subjects": aligned}, indent=2) + "\n", encoding="utf-8"
    )
    result = prepare_dataset(aligned_manifest, output, target_shape, reference=fixed_path)
    prepared_payload = json.loads(result.read_text(encoding="utf-8"))
    prepared_payload["dataset"] = dataset
    prepared_payload["reference"] = {
        "source": reference_path.name,
        "subject_id": reference_entry["id"] if reference_entry else None,
        "selection": reference_selection,
    }
    prepared_payload["preprocessing"] = (
        "Estimated 12-DOF anatomical affine alignment; " if affine else "Caller-prealigned images; "
    ) + "canonical RAS shared world grid; nonzero p1/p99 scaling; endpoint-aligned resize"
    prepared_payload["preprocessing_provenance"] = "preprocessing/provenance.json"
    result.write_text(json.dumps(prepared_payload, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "dataset": dataset,
        "seed": seed,
        "shape_dhw": list(target_shape),
        "paper_shape_dhw": list(DATASET_SHAPES[dataset]),
        "splits": {s: sum(r["split"] == s for r in records) for s in ("train", "val", "test")},
        "split_method": "supplied"
        if "split" in payload["subjects"][0]
        else "seeded subject-wise 7:1:2",
        "reference": prepared_payload["reference"],
        "affine_prealignment": affine,
        "simpleitk_version": sitk.Version_VersionString() if sitk else None,
        "threads": threads,
        "transform_convention": "ITK physical LPS millimeters; fixed-to-moving pullback",
        "label_interpolation": "nearest neighbor; original anatomy IDs preserved",
        "annotations": "external inputs only; never used in the affine objective",
        "subjects": transforms,
    }
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return result
