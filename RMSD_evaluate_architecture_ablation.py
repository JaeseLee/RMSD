"""Evaluate epoch checkpoints from the architecture-ablation study.

This script deliberately reuses the same preprocessing, checkpoint loading, and
streaming pixel-wise metrics as RMSD_evaluate_ablation_epoch100.py.  It adds
architecture-aware labels and reports X->Y and Y->X separately.
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import RMSD_evaluate_ablation_epoch100 as base


ARCHITECTURES = {
    "arch_B0_unet": "U-Net",
    "arch_B1_light_full": "Full",
    "arch_B2_no_depthwise": "No depthwise",
    "arch_B3_no_residual": "No residual",
    "arch_B4_no_eca": "No ECA",
    "arch_B5_no_skip_attention": "No skip attention",
    "arch_B6_no_dilation": "No dilation",
    "arch_B7_batchnorm": "BatchNorm",
    "arch_B8_relu6": "ReLU6",
}
ARCH_ORDER = {name: i for i, name in enumerate(ARCHITECTURES)}
NO_DEPTHWISE = "arch_B2_no_depthwise"
CHANNELS = {
    "X_to_Y": ("MNDWI", "NDVI", "NDWI"),
    "Y_to_X": ("VV", "VH", "RVI"),
}


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("epoch", "n", "corr", "rmse", "ubRMSD", "bias", "ssim"):
            if key in row:
                row[key] = int(row[key]) if key in ("epoch", "n") else float(row[key])
    return rows


def architecture_label(experiment):
    return ARCHITECTURES.get(experiment, experiment.replace("arch_", "").replace("_", " "))


def sort_experiments(experiments):
    return sorted(experiments, key=lambda name: (ARCH_ORDER.get(name, 999), name))


def summarize(rows):
    """Average channel metrics within each direction; never mix optical and SAR units."""
    output = []
    for experiment in sort_experiments({row["experiment"] for row in rows}):
        for direction in ("X_to_Y", "Y_to_X"):
            selected = [row for row in rows
                        if row["experiment"] == experiment and row["direction"] == direction]
            if not selected:
                continue
            item = {
                "experiment": experiment,
                "architecture": architecture_label(experiment),
                "direction": direction,
            }
            for metric in ("corr", "rmse", "ubRMSD", "bias", "ssim"):
                item[f"mean_{metric}"] = float(np.mean([float(row[metric]) for row in selected]))
            output.append(item)
    return output


def metric_limits(values, metric):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0) if metric == "corr" else (-1.0, 1.0)
    if metric in ("corr", "ssim"):
        lo = max(-1.0 if metric == "ssim" else 0.0,
                 values.min() - max(np.ptp(values) * .20, .02))
        return lo, min(1.0, values.max() + max(np.ptp(values) * .12, .01))
    if metric in ("rmse", "ubRMSD"):
        return 0.0, values.max() * 1.16
    bound = max(np.abs(values).max() * 1.20, .01)
    return -bound, bound


def plot_direction(rows, direction, path, epoch):
    selected = [row for row in rows
                if row["direction"] == direction and
                all(key in row and np.isfinite(float(row[key]))
                    for key in ("corr", "rmse", "ubRMSD", "bias", "ssim"))]
    experiments = sort_experiments({row["experiment"] for row in selected})
    if not experiments:
        return
    channels = CHANNELS[direction]
    lookup = {(row["experiment"], row["channel"]): row for row in selected}
    colors = ("#0072B2", "#E69F00", "#009E73")  # colorblind-safe
    metrics = (("corr", "Correlation (R)"), ("ssim", "SSIM"), ("rmse", "RMSE"),
               ("ubRMSD", "ubRMSD"), ("bias", "Bias"))
    x = np.arange(len(experiments))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200, constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        values = np.asarray([[float(lookup[(exp, channel)][metric]) for channel in channels]
                             for exp in experiments])
        means = values.mean(axis=1)
        bars = axis.bar(x, means, width=.72, color="#d1d5db", edgecolor="#374151",
                        linewidth=.8, zorder=2, label="Channel mean")
        for index, (channel, color) in enumerate(zip(channels, colors)):
            axis.scatter(x + (-.16, 0, .16)[index], values[:, index], s=45,
                         color=color, edgecolor="white", linewidth=.5,
                         zorder=4, label=channel)
        if metric == "bias":
            axis.axhline(0, color="#4b5563", linewidth=.9)
        axis.set_ylim(*metric_limits(values.ravel(), metric))
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels([architecture_label(exp) for exp in experiments],
                             rotation=28, ha="right")
        axis.grid(axis="y", alpha=.22, zorder=0)
        axis.set_title(ylabel)
        span = np.diff(axis.get_ylim())[0]
        for bar, value in zip(bars, means):
            offset = .018 * span if value >= 0 else -.018 * span
            axis.text(bar.get_x() + bar.get_width()/2, value + offset, f"{value:.3f}",
                      ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, 1.025))
    direction_label = "SAR → optical" if direction == "X_to_Y" else "Optical → SAR"
    fig.suptitle(f"Architecture ablation — {direction_label} — epoch {epoch}", fontsize=16)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_bidirectional_summary(rows, path, epoch):
    """Plot direction means together, using translation direction as the hue."""
    summary = summarize(rows)
    experiments = sort_experiments({row["experiment"] for row in summary})
    if not experiments:
        return
    lookup = {(row["experiment"], row["direction"]): row for row in summary}
    directions = ("X_to_Y", "Y_to_X")
    direction_labels = ("X→Y (SAR→Optical)", "Y→X (Optical→SAR)")
    colors = ("#0072B2", "#D55E00")
    metrics = (("mean_corr", "Mean correlation (R)"),
               ("mean_ssim", "Mean SSIM"),
               ("mean_rmse", "Mean RMSE"),
               ("mean_ubRMSD", "Mean ubRMSD"),
               ("mean_bias", "Mean bias"))
    x = np.arange(len(experiments))
    width = .36
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200, constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        all_values = []
        for direction_index, (direction, direction_label, color) in enumerate(
                zip(directions, direction_labels, colors)):
            values = np.asarray([float(lookup[(experiment, direction)][metric])
                                 for experiment in experiments])
            all_values.extend(values)
            positions = x + (direction_index - .5) * width
            bars = axis.bar(positions, values, width=width, color=color, alpha=.82,
                            label=direction_label, zorder=2)
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width()/2, value,
                          f"{value:.3f}", ha="center",
                          va="bottom" if value >= 0 else "top", fontsize=7, rotation=90)
        if metric == "mean_bias":
            axis.axhline(0, color="#4b5563", linewidth=.9)
        else:
            axis.set_ylim(bottom=0)
        if metric in ("mean_corr", "mean_ssim"):
            axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_ylabel(title)
        axis.set_xticks(x)
        axis.set_xticklabels([architecture_label(exp) for exp in experiments],
                             rotation=28, ha="right")
        axis.grid(axis="y", alpha=.22, zorder=0)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.5, 1.025))
    fig.suptitle(f"Bidirectional architecture ablation — epoch {epoch}", fontsize=16)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_validation_grid(results, direction, path, epoch):
    experiments = sort_experiments(results)
    if not experiments:
        return
    if direction == "X_to_Y":
        prediction_key, target_key, mask_key = "y_prediction", "y_target", "y_mask"
    else:
        prediction_key, target_key, mask_key = "x_prediction", "x_target", "x_mask"
    channels = CHANNELS[direction]
    ncols = len(experiments) + 1
    fig, axes = plt.subplots(3, ncols, figsize=(2.05*ncols, 6.2), dpi=180,
                             constrained_layout=True, squeeze=False)
    reference = results[experiments[0]]
    for row_index, channel in enumerate(channels):
        for column_index, experiment in enumerate(experiments):
            result = results[experiment]
            image = result[prediction_key][row_index].copy()
            image[~result[mask_key][row_index]] = np.nan
            axes[row_index, column_index].imshow(image, cmap="viridis", vmin=0, vmax=1)
            valid = result[mask_key][row_index] & np.isfinite(image) & np.isfinite(result[target_key][row_index])
            corr = np.corrcoef(image[valid], result[target_key][row_index][valid])[0, 1]
            axes[row_index, column_index].set_title(
                f"{architecture_label(experiment)}\nR={corr:.3f}" if row_index == 0 else f"R={corr:.3f}",
                fontsize=8)
            axes[row_index, column_index].set_ylabel(channel if column_index == 0 else "")
            axes[row_index, column_index].set_xticks([]); axes[row_index, column_index].set_yticks([])
        target = reference[target_key][row_index].copy()
        target[~reference[mask_key][row_index]] = np.nan
        axes[row_index, -1].imshow(target, cmap="viridis", vmin=0, vmax=1)
        axes[row_index, -1].set_title("Target" if row_index == 0 else "", fontsize=8)
        axes[row_index, -1].set_xticks([]); axes[row_index, -1].set_yticks([])
    fig.suptitle(f"Architecture ablation validation sample 0 — {direction} — epoch {epoch}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_outputs(rows, output_dir, epoch):
    save_csv(summarize(rows), os.path.join(output_dir, "architecture_mean_test_metrics.csv"))
    for direction in ("X_to_Y", "Y_to_X"):
        plot_direction(rows, direction,
                       os.path.join(output_dir, f"architecture_test_metrics_{direction}.png"), epoch)
    plot_bidirectional_summary(
        rows,
        os.path.join(output_dir, "architecture_test_metrics_bidirectional_hue.png"),
        epoch,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_dir", default="/Users/jslee/Downloads/SEN12FLOOD/output/architecture_ablation_seed_42")
    parser.add_argument("--rootpath", default="/Users/jslee/Downloads/SEN12FLOOD")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiment", default="arch_B*")
    parser.add_argument("--include_no_depthwise", action="store_true",
                        help="Also evaluate arch_B2_no_depthwise (excluded by default because it produced NaNs).")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--maps_only", action="store_true",
                        help="Run only validation sample 0 and save map figures; skip test-set metrics.")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.study_dir, f"epoch_{args.epoch:03d}_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "all_test_metrics.csv")
    if args.plot_only:
        rows = load_csv(metrics_path)
        if not args.include_no_depthwise:
            rows = [row for row in rows if row["experiment"] != NO_DEPTHWISE]
            save_csv(rows, metrics_path)
        if rows and "ssim" not in rows[0]:
            raise RuntimeError(
                "The existing all_test_metrics.csv has no SSIM column. "
                "Run once without --plot_only to calculate SSIM."
            )
        build_outputs(rows, output_dir, args.epoch)
        print(f"Rebuilt plots from {metrics_path}")
        return

    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available()
                          else "cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    validation, test, scales = base.prepare_data(args.rootpath, args.seed)
    checkpoints = glob.glob(os.path.join(args.study_dir, args.experiment,
                                         f"model_epoch_{args.epoch:03d}.pt"))
    if not args.include_no_depthwise:
        checkpoints = [path for path in checkpoints
                       if os.path.basename(os.path.dirname(path)) != NO_DEPTHWISE]
    checkpoints.sort(key=lambda p: (ARCH_ORDER.get(os.path.basename(os.path.dirname(p)), 999), p))
    if not checkpoints:
        raise FileNotFoundError(f"No model_epoch_{args.epoch:03d}.pt under {args.study_dir}")

    all_rows, validation_results = [], {}
    for checkpoint_path in checkpoints:
        experiment_dir = os.path.dirname(checkpoint_path)
        experiment = os.path.basename(experiment_dir)
        config_paths = glob.glob(os.path.join(experiment_dir, "config_*.json"))
        if not config_paths:
            print(f"[{experiment}] skipped: config_*.json not found")
            continue
        with open(config_paths[0], encoding="utf-8") as stream:
            config = json.load(stream)
        print(f"[{experiment}] evaluating {os.path.basename(checkpoint_path)} on {device}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_xy, model_yx = base.build_model(config).to(device), base.build_model(config).to(device)
        model_xy.load_state_dict(checkpoint["model_xy_state_dict"])
        model_yx.load_state_dict(checkpoint["model_yx_state_dict"])
        model_xy.eval(); model_yx.eval()

        values = base.predict_index_zero(model_xy, model_yx, validation, device)
        base.plot_val0(
            values,
            os.path.join(output_dir, f"{experiment}_validation_maps.png"),
            f"{architecture_label(experiment)} | epoch {args.epoch} | validation sample 0",
        )
        x, y, px, py, mx, my = values
        validation_results[experiment] = {
            "x_target": x[0], "x_prediction": px[0], "x_mask": mx[0],
            "y_target": y[0], "y_prediction": py[0], "y_mask": my[0],
        }
        if not args.maps_only:
            base.evaluate_test_streaming.error_sums = {}
            rows = base.evaluate_test_streaming(model_xy, model_yx, test, device, args.bs,
                                                f"{experiment} test", scales,
                                                include_ssim=True)
            rows = [{"experiment": experiment, "epoch": args.epoch, **row} for row in rows]
            save_csv(rows, os.path.join(output_dir, f"{experiment}_test_metrics.csv"))
            all_rows.extend(rows)
        del model_xy, model_yx
        if device.type == "mps":
            torch.mps.empty_cache()

    for direction in ("X_to_Y", "Y_to_X"):
        plot_validation_grid(validation_results, direction,
                             os.path.join(output_dir, f"architecture_validation_{direction}.png"),
                             args.epoch)
    if not args.maps_only:
        save_csv(all_rows, metrics_path)
        build_outputs(all_rows, output_dir, args.epoch)
    with open(os.path.join(output_dir, "evaluation_info.json"), "w", encoding="utf-8") as stream:
        json.dump({"epoch": args.epoch, "seed": args.seed, "device": str(device),
                   "experiments": len(validation_results), "maps_only": args.maps_only,
                   "study_dir": args.study_dir}, stream, indent=2)
    print(f"Saved architecture-ablation evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
