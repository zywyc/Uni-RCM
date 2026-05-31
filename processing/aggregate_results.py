"""
Aggregate per-class Uni-RCM result files into a summary table.

Result files are written by uni_rcm_infer.py with the naming pattern:
  {class_name}_uni_rcm_h{hidden_dim}_L{num_blocks}_{epochs_no}ep_{train_batch_size}bs

Each file contains three lines:
  Line 1: title string
  Line 2: metric header
  Line 3: '&'-separated float values (AUPRO@30%, AUPRO@10%, AUPRO@5%, AUPRO@1%, P-AUROC, I-AUROC)

Running this script again after updating any per-class file will overwrite
the previous aggregated CSV automatically.
"""

import os
import argparse
import numpy as np
import pandas as pd

CLASSES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
]
METRICS = ["AUPRO@30%", "AUPRO@10%", "AUPRO@5%", "AUPRO@1%", "P-AUROC", "I-AUROC"]


def find_result_file(directory, class_name):
    """Return the path of the result file for a given class, or None if absent."""
    for fname in os.listdir(directory):
        if fname.startswith(class_name + "_"):
            return os.path.join(directory, fname)
    return None


def parse_result_file(filepath):
    """
    Parse the last non-empty line of a result file.
    Returns a list of 6 floats (one per metric).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.read().strip().splitlines() if ln.strip()]
    last_line = lines[-1]
    return [float(v.strip()) for v in last_line.split("&")]


def aggregate(args):
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    results = {}
    missing = []

    for cls in CLASSES:
        fpath = find_result_file(args.quantitative_folder, cls)
        if fpath is None:
            print(f"  [MISSING] No result file for '{cls}'")
            missing.append(cls)
            continue
        try:
            values = parse_result_file(fpath)
            results[cls] = values
            print(f"  [OK] {cls}: " + " | ".join(f"{v:.3f}" for v in values))
        except Exception as e:
            print(f"  [ERROR] Could not parse '{fpath}': {e}")
            missing.append(cls)

    if not results:
        print("\nNo result files could be parsed. Nothing to aggregate.")
        return

    # Build DataFrame: rows = classes, columns = metrics
    df = pd.DataFrame(results, index=METRICS).T
    df.loc["mean"] = df.mean(axis=0)

    print("\n" + "=" * 70)
    print("Aggregated Results (rows = classes, cols = metrics):")
    print(df.to_string(float_format="%.3f"))
    print("=" * 70)

    # LaTeX table (metrics as rows, classes + mean as columns)
    print("\nLaTeX table:")
    print(df.T.to_latex(float_format="%.3f"))

    # Save CSV — overwrite any previous run
    df.to_csv(args.output_file, float_format="%.4f")
    print(f"\nSaved: {args.output_file}")

    if missing:
        print(f"Note: results missing for {len(missing)} class(es): {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate per-class Uni-RCM inference results into a summary table."
    )
    parser.add_argument(
        "--quantitative_folder",
        default="./results/quantitatives_uni_rcm",
        type=str,
        help="Folder containing per-class result files written by uni_rcm_infer.py.",
    )
    parser.add_argument(
        "--output_file",
        default="./results/aggregated_results.csv",
        type=str,
        help="Output CSV path. Overwritten on every run.",
    )
    args = parser.parse_args()
    aggregate(args)
