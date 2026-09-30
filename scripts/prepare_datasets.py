#!/usr/bin/env python3
r"""Download and preprocess the three datasets used in SPEAR.

Install preprocessing support: python -m pip install -e '.[datasets]'

Examples (run from the project root after obtaining provider authorization):
  python scripts/prepare_datasets.py catalog
  python scripts/prepare_datasets.py all --dataset ixi --root data/ixi --terms-accepted
  python scripts/prepare_datasets.py all --dataset oasis --root data/oasis --terms-accepted
  python scripts/prepare_datasets.py all --dataset ibsr18 --root data/ibsr18 \
      --archive /downloads/IBSR_V2.0_nifti_stripped.tgz
  python scripts/prepare_datasets.py prepare --dataset ixi --root data/ixi \
      --labels-dir /data/ixi_anatomical_labels --require-labels
  python scripts/prepare_datasets.py prepare --dataset oasis --root data/oasis \
      --manifest /data/oasis_my_images_labels_splits.json

The download command accepts repeated --archive arguments, --cookie-file, or
--url-list private JSON [{"url": "https://...", "filename": "disc1.tar.gz",
"sha256": "optional known SHA-256"}, ...]. Query tokens/cookies are not logged.
OASIS-1 uses all 12 raw discs and only MR1, one image per person. OASIS requires
its current request/usage process; --terms-accepted is not access approval.

Default preprocessing actually estimates affine alignment to a TRAIN image,
normalizes intensities and uses paper sizes (D,H,W): IXI 144^3, OASIS
162x198x162, IBSR18 225^3. --already-affine-prealigned is only for genuinely
aligned images; matching NIfTI headers alone is not anatomical alignment.
For SPEAR, IBSR18 is NOT halved: the paper's 2x reduction applies to baselines.

Raw IXI/OASIS packages require separately supplied regional anatomy labels.
Attach your anatomical labels through --labels-dir or --manifest. We never
replace missing anatomy with tissue/brain masks. --require-labels catches this
before preparing data for the semantic-prior training workflow. A manifest may
contain image, label, mask, split, and subject_id fields as well as id. Follow
the supplied-split route to retain an existing experimental partition. If
splits are omitted, a seeded subject-wise partition of approximately 7:1:2
is generated.

Only run directories contain downloads, transforms, manifests or provenance;
no images, metrics, trained weights or experiment outputs belong in Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from spear.benchmark_data import DATASET_SHAPES, discover_subjects, prepare_benchmark
from spear.dataset_download import DATASETS, acquire_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["catalog", "download", "prepare", "all"])
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--root", type=Path, help="Local dataset root; default data/DATASET")
    parser.add_argument(
        "--archive",
        action="append",
        type=Path,
        help="Already obtained tar/tgz/zip; repeat for multiple OASIS discs",
    )
    parser.add_argument(
        "--url-list",
        type=Path,
        help="Private JSON with authorized archive URLs, filenames and optional hashes",
    )
    parser.add_argument(
        "--cookie-file",
        type=Path,
        help="Existing Netscape cookie jar for an authorized provider login",
    )
    parser.add_argument(
        "--terms-accepted",
        action="store_true",
        help="Declare you have already followed the provider's access and usage requirements",
    )
    parser.add_argument(
        "--raw-dir", type=Path, help="Override ROOT/raw for already extracted images"
    )
    parser.add_argument(
        "--manifest", type=Path, help="Raw image/label manifest; optional original subject splits"
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Explicitly supplied anatomical label directory, matched by subject",
    )
    parser.add_argument(
        "--require-labels",
        action="store_true",
        help="Fail early if any selected subject lacks a supplied label",
    )
    parser.add_argument(
        "--already-affine-prealigned",
        action="store_true",
        help="Skip affine estimation only for anatomically prealigned inputs",
    )
    parser.add_argument(
        "--ibsr-labels-share-image-grid",
        action="store_true",
        help="IBSR only: explicitly repair a known mismatched label header when voxels correspond one-to-one",
    )
    parser.add_argument(
        "--reference", help="Reference training subject ID (or matching training image path)"
    )
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        metavar=("D", "H", "W"),
        help="Override paper dimensions, e.g. 36 36 36 for a smoke test",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="SimpleITK preprocessing threads (preprocessing runs on CPU)",
    )
    args = parser.parse_args(argv)
    if args.command == "catalog":
        print(
            json.dumps(
                {
                    name: {**info, "paper_shape_dhw": DATASET_SHAPES[name]}
                    for name, info in DATASETS.items()
                },
                indent=2,
            )
        )
        return
    if not args.dataset:
        parser.error("--dataset is required")
    if args.manifest and args.labels_dir:
        parser.error(
            "Put label paths in --manifest, or use --labels-dir with automatic image discovery"
        )
    if args.command == "prepare" and (args.archive or args.url_list or args.cookie_file):
        parser.error(
            "Download options belong to download/all; prepare reads extracted images or a manifest"
        )
    if args.threads < 1 or (args.shape and any(value < 9 or value % 9 for value in args.shape)):
        parser.error("threads must be positive; spatial dimensions must be positive multiples of 9")
    if args.ibsr_labels_share_image_grid and args.dataset != "ibsr18":
        parser.error("--ibsr-labels-share-image-grid is only valid for IBSR18")
    root = (args.root or Path("data") / args.dataset).resolve()
    raw = args.raw_dir or root / "raw"
    if args.command in ("download", "all"):
        if args.raw_dir:
            parser.error("--raw-dir belongs to prepare; download/all extract into ROOT/raw")
        raw = acquire_dataset(
            args.dataset,
            root,
            archives=args.archive,
            url_manifest=args.url_list,
            cookie_file=args.cookie_file,
            terms_accepted=args.terms_accepted,
        )
        print(f"Extracted raw data: {raw}")
        if args.command == "download":
            return
    if args.manifest:
        manifest = args.manifest.resolve()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        # Keep relative image/label paths relative to the operator's manifest.
        records = payload["subjects"]
    else:
        records = discover_subjects(args.dataset, raw, args.labels_dir)
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "input_manifest.json"
        payload = {"dataset": args.dataset, "subjects": records}
    if not records:
        parser.error("No matching subjects found; supply a raw --manifest for a custom layout")
    missing = [str(row["id"]) for row in records if not row.get("label")]
    if missing:
        message = (
            f"{len(missing)}/{len(records)} subjects have no anatomical labels. "
            "Semantic-prior pretraining needs training annotations; provide --labels-dir or a labeled --manifest. "
            "Missing labels are not synthesized."
        )
        if args.require_labels:
            parser.error(message)
        print("Note: " + message, file=sys.stderr)
    if args.ibsr_labels_share_image_grid:
        # A derived local manifest makes an explicit header repair reviewable,
        # without changing the operator's original manifest or image bytes.
        for row in records:
            if row.get("label"):
                row["label_shares_image_grid"] = True
        if args.manifest:
            for row in records:
                for key in ("image", "label", "mask"):
                    if row.get(key):
                        row[key] = str((manifest.parent / row[key]).resolve())
            root.mkdir(parents=True, exist_ok=True)
            manifest = root / "input_manifest.json"
    if not args.manifest or args.ibsr_labels_share_image_grid:
        manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output = prepare_benchmark(
        args.dataset,
        manifest,
        root / "prepared",
        shape=tuple(args.shape) if args.shape else None,
        seed=args.seed,
        affine=not args.already_affine_prealigned,
        reference=args.reference,
        threads=args.threads,
    )
    print(f"Prepared manifest: {output}")
    print(
        f"GPU workflow: bash scripts/run_gpu.sh --manifest {output} --output runs/{args.dataset} --device cuda:0"
    )


if __name__ == "__main__":
    main()
