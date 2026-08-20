"""Plot architecture-ablation validation histories without rerunning models."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from RMSD_evaluate_architecture_ablation import (
    ARCHITECTURES,
    NO_DEPTHWISE,
    architecture_label,
    sort_experiments,
)


def read_history(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        parsed = {"epoch": int(row["epoch"])}
        for key in ("train_loss", "val_loss", "val_xy_loss", "val_yx_loss"):
            parsed[key] = float(row[key])
        output.append(parsed)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study_dir", default="/Users/jslee/Downloads/SEN12FLOOD/output/architecture_ablation_seed_42")
    parser.add_argument("--output_dir", default="architecture_ablation_evaluation")
    parser.add_argument("--include_no_depthwise", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    experiments = sort_experiments(ARCHITECTURES)
    if not args.include_no_depthwise:
        experiments = [name for name in experiments if name != NO_DEPTHWISE]
    histories = {}
    for experiment in experiments:
        history_path = os.path.join(args.study_dir, experiment, "loss_history.csv")
        if not os.path.isfile(history_path):
            print(f"[{experiment}] skipped: {history_path} not found")
            continue
        finite = [row for row in read_history(history_path) if np.isfinite(row["val_loss"])]
        if finite:
            histories[experiment] = finite
    if not histories:
        raise FileNotFoundError("No finite architecture loss histories found")

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=200, constrained_layout=True)
    axes = axes.flat
    best_rows = {}
    for (experiment, history), color in zip(histories.items(), colors):
        epochs = [row["epoch"] for row in history]
        losses = [row["val_loss"] for row in history]
        axes[0].plot(epochs, losses, color=color, linewidth=2,
                     label=architecture_label(experiment))
        best_rows[experiment] = min(history, key=lambda row: row["val_loss"])
    axes[0].set_title("(a) Validation-loss trajectories")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation loss")
    axes[0].grid(alpha=.22)

    x = np.arange(len(best_rows))
    labels = [architecture_label(name) for name in best_rows]
    for axis, key, title in (
        (axes[1], "val_loss", "(b) Best total validation loss"),
        (axes[2], "val_xy_loss", "(c) X→Y loss at best epoch"),
        (axes[3], "val_yx_loss", "(d) Y→X loss at best epoch"),
    ):
        values = [best_rows[name][key] for name in best_rows]
        bars = axis.bar(x, values, color=colors, edgecolor="#374151", linewidth=.7)
        axis.set_title(title); axis.set_ylabel("Loss")
        axis.set_xticks(x); axis.set_xticklabels(labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=.22)
        axis.set_ylim(bottom=max(0, min(values) - .015))
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width()/2, value, f"{value:.3f}",
                      ha="center", va="bottom", fontsize=8)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(.5, 1.06))
    fig.suptitle("Architecture ablation study — validation performance", fontsize=16)
    output_path = os.path.join(args.output_dir, "architecture_ablation_validation.png")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = os.path.join(args.output_dir, "architecture_ablation_validation_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as stream:
        fields = ("experiment", "architecture", "best_epoch", "best_val_loss",
                  "best_val_xy_loss", "best_val_yx_loss")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for experiment, row in best_rows.items():
            writer.writerow({"experiment": experiment,
                             "architecture": architecture_label(experiment),
                             "best_epoch": row["epoch"],
                             "best_val_loss": row["val_loss"],
                             "best_val_xy_loss": row["val_xy_loss"],
                             "best_val_yx_loss": row["val_yx_loss"]})
    print(f"Saved validation figure to: {output_path}")


if __name__ == "__main__":
    main()
