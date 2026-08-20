"""Rebuild architecture-ablation test metric figures from an existing CSV."""

import argparse
import os

from RMSD_evaluate_architecture_ablation import (
    NO_DEPTHWISE,
    load_csv,
    plot_bidirectional_summary,
    plot_direction,
    save_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", help="Path to all_test_metrics.csv")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--include_no_depthwise", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.metrics_csv))
    os.makedirs(output_dir, exist_ok=True)
    rows = load_csv(args.metrics_csv)
    if not args.include_no_depthwise:
        rows = [row for row in rows if row["experiment"] != NO_DEPTHWISE]
    if rows and "ssim" not in rows[0]:
        raise RuntimeError(
            "The metrics CSV has no SSIM column. Run "
            "RMSD_evaluate_architecture_ablation.py once before plotting."
        )

    save_csv(rows, os.path.join(output_dir, "all_test_metrics_no_depthwise.csv"))
    for direction in ("X_to_Y", "Y_to_X"):
        plot_direction(
            rows,
            direction,
            os.path.join(output_dir, f"architecture_test_metrics_{direction}.png"),
            args.epoch,
        )
    plot_bidirectional_summary(
        rows,
        os.path.join(output_dir, "architecture_test_metrics_bidirectional_hue.png"),
        args.epoch,
    )
    print(f"Saved test metric figures to: {output_dir}")


if __name__ == "__main__":
    main()
