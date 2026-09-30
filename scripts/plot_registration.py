#!/usr/bin/env python3
"""Plot declared test pairs, without selecting examples by their outcomes.

Requires the optional ``plots`` extra. MRI-derived figures retain the dataset's
CC BY-SA 3.0 license and attribution; they are not covered by the code license.
"""

import argparse
from pathlib import Path

import torch

from spear.data import PairDataset, SubjectDataset
from spear.engine import load_registration, seed_everything


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int, default=2)
    args = parser.parse_args()
    if args.max_pairs < 1:
        raise ValueError("max-pairs must be positive")
    # Keep plotting dependencies out of training/inference installations.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_everything(17, 2)
    model, _ = load_registration(args.checkpoint)
    pairs = PairDataset(SubjectDataset(args.manifest, "test"))
    count = min(args.max_pairs, len(pairs))
    fig, axes = plt.subplots(
        count, 5, figsize=(12, 2.65 * count + 0.7), squeeze=False, constrained_layout=True
    )
    titles = ["Fixed", "Moving", "SPEAR warped", "Abs. error before", "Abs. error after"]
    for index in range(count):
        sample = pairs[index]
        moving, fixed = sample["moving"][None], sample["fixed"][None]
        warped = model(moving, fixed)["warped"]
        z = fixed.shape[2] // 2
        images = [fixed, moving, warped, (moving - fixed).abs(), (warped - fixed).abs()]
        error_max = max(float(images[3][0, 0, z].max()), float(images[4][0, 0, z].max()), 1e-5)
        for col, image in enumerate(images):
            ax = axes[index, col]
            ax.imshow(
                image[0, 0, z].numpy(),
                cmap="gray" if col < 3 else "magma",
                vmin=0,
                vmax=1 if col < 3 else error_max,
                origin="lower",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if index == 0:
                ax.set_title(titles[col])
        axes[index, 0].set_ylabel(f"{sample['moving_id']}\nto {sample['fixed_id']}", fontsize=8)
    fig.suptitle("Held-out real IXITiny pairs · middle axial slice · identical error scales")
    fig.supxlabel(
        "36³ CPU experiment; whole-brain masks only. IXI / TorchIO IXITiny image adaptation: CC BY-SA 3.0.",
        fontsize=8,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
