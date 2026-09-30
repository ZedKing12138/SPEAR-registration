#!/usr/bin/env python3
"""Fetch a fixed 12-subject IXITiny subset for CPU development tests.

The upstream archive is pinned by Git commit and SHA-256. Only explicitly
selected image/mask files are extracted; no downloaded Python is executed.
The data are IXI-derived and remain CC BY-SA 3.0, separate from this project's
software license. Keep the generated provenance.json alongside any derived
results. This script does not contain or redistribute patient image bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

COMMIT = "5918756215dfde86ef37d670c906b172b679880b"
ARCHIVE_URL = (
    "https://raw.githubusercontent.com/TorchIO-project/torchio-data/"
    f"{COMMIT}/data/ixi_tiny/ixi_tiny_1.zip"
)
ARCHIVE_SHA256 = "af0b3ed2ee47ff283426fa9560d1e9cd32c11b41535762b8190fe6e2ca693d4c"


def sha256(path: Path) -> str:
    """Hash in chunks instead of keeping the 86 MB archive in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_subset(output: Path, archive: Path | None = None) -> Path:
    """Return a subject-level manifest with disjoint 8/2/2 splits.

    The deterministic first 12 lexicographic IDs are a convenient test fixture,
    not a statistically representative sample of IXI. Masks contain only brain
    foreground/background. Regional anatomical Dice requires separate labels.
    """
    output.mkdir(parents=True, exist_ok=True)
    archive = archive or output / "ixi_tiny_1.zip"
    if not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        partial = archive.with_suffix(".download")
        try:
            with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response:
                with partial.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    checksum = sha256(archive)
    if checksum != ARCHIVE_SHA256:
        raise RuntimeError(f"IXITiny checksum mismatch: {checksum}")

    subjects = []
    with zipfile.ZipFile(archive) as zfile:
        image_members = sorted(
            name
            for name in zfile.namelist()
            if name.startswith("ixi_tiny_1/image/") and name.endswith("_image.nii.gz")
        )[:12]
        if len(image_members) != 12:
            raise RuntimeError("The pinned archive must contain at least 12 image subjects")
        for index, member in enumerate(image_members):
            subject_id = Path(member).name.removesuffix("_image.nii.gz")
            label_member = f"ixi_tiny_1/label/{subject_id}_label.nii.gz"
            record = {
                "subject_id": subject_id,
                "split": "train" if index < 8 else "val" if index < 10 else "test",
            }
            for key, source in (("image", member), ("label", label_member)):
                # Explicit member selection avoids archive path traversal.
                relative = Path(key) / Path(source).name
                target = output / relative
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(zfile.read(source))
                record[key] = relative.as_posix()
                record[f"{key}_sha256"] = sha256(target)
            subjects.append(record)
    provenance = {
        "dataset": "IXITiny, a low-resolution IXI-derived T1 MRI subset",
        "purpose": "Small-volume training and inference check using real MRI",
        "data_license": "CC-BY-SA-3.0",
        "data_license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "original_source": "https://brain-development.org/ixi-dataset/",
        "derivative_provider": "TorchIO project",
        "derivative_documentation": "https://docs.torchio.org/latest/datasets/",
        "archive_url": ARCHIVE_URL,
        "archive_sha256": checksum,
        "upstream_commit": COMMIT,
        "selection": "First 12 lexicographically sorted IDs from the pinned first archive",
        "split_policy": "First 8 training, next 2 validation, last 2 test; subject-disjoint",
        "native_shape_xyz": [83, 44, 55],
        "mask_values": [0, 1],
        "mask_meaning": "Background and whole-brain foreground; no regional anatomical labels",
        "processing": "Archive extraction only; original NIfTI headers and image bytes retained",
        "subjects": subjects,
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    # The CLI's raw-data interface is deliberately small and independent of
    # the richer attribution record. All paths remain relative and portable.
    manifest_payload = {
        "subjects": [
            {
                "id": record["subject_id"],
                "image": record["image"],
                "label": record["label"],
                "split": record["split"],
            }
            for record in subjects
        ]
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ixi_tiny"))
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional already-downloaded archive (still checksum verified)",
    )
    args = parser.parse_args()
    print(download_subset(args.output, args.archive))


if __name__ == "__main__":
    main()
