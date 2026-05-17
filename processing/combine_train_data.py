"""
Combine training data from all 10 MVTec 3D-AD classes into a single
'combine/train/good' directory with renamed files (e.g. bagel_000.png).

Original structure:
  {root}/{class}/train/good/rgb/000.png
  {root}/{class}/train/good/xyz/000.tiff

Output structure:
  {root}/combine/train/good/rgb/{class}_000.png
  {root}/combine/train/good/xyz/{class}_000.tiff
"""

import os
import shutil
import argparse
from glob import glob

MVTEC3D_CLASSES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
]

EYECANDIES_CLASSES = [
    'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto',
    'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop',
    'Marshmallow', 'PeppermintCandy',   
]

def combine_training_data(root_path):
    combine_rgb_dir = os.path.join(root_path, "combine", "train", "good", "rgb")
    combine_xyz_dir = os.path.join(root_path, "combine", "train", "good", "xyz")

    os.makedirs(combine_rgb_dir, exist_ok=True)
    os.makedirs(combine_xyz_dir, exist_ok=True)

    total_copied = 0

    for cls in MVTEC3D_CLASSES:
        rgb_dir = os.path.join(root_path, cls, "train", "good", "rgb")
        xyz_dir = os.path.join(root_path, cls, "train", "good", "xyz")

        rgb_files = sorted(glob(os.path.join(rgb_dir, "*.png")))
        xyz_files = sorted(glob(os.path.join(xyz_dir, "*.tiff")))

        n = min(len(rgb_files), len(xyz_files))
        if n == 0:
            print(f"  [SKIP] {cls}: no files found in {rgb_dir} or {xyz_dir}")
            continue

        for i in range(n):
            rgb_src = rgb_files[i]
            xyz_src = xyz_files[i]
            stem = f"{cls}_{i:03d}"

            rgb_dst = os.path.join(combine_rgb_dir, f"{stem}.png")
            xyz_dst = os.path.join(combine_xyz_dir, f"{stem}.tiff")

            shutil.copy2(rgb_src, rgb_dst)
            shutil.copy2(xyz_src, xyz_dst)

            total_copied += 1

        print(f"  [{cls}] copied {n} files ({n} rgb + {n} xyz)")

    print(f"\nDone. Total pairs copied: {total_copied}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine MVTec 3D-AD training data from all classes into a unified 'combine' directory."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Root path of the MVTec 3D-AD dataset",
    )
    args = parser.parse_args()

    combine_training_data(args.dataset_path)
