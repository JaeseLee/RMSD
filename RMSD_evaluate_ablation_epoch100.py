"""Evaluate every epoch-100 bidirectional model in an ablation-study folder."""

import argparse
import csv
import glob
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rmsd_light_model import LightAttentionUNet


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2) - x1.size(2), x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.down1, self.down2 = Down(64, 128), Down(128, 256)
        self.down3, self.down4 = Down(256, 512), Down(512, 512)
        self.up1, self.up2 = Up(1024, 256), Up(512, 128)
        self.up3, self.up4 = Up(256, 64), Up(128, 64)
        self.outc = OutConv(64, 3)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        return self.outc(self.up4(self.up3(self.up2(self.up1(x5, x4), x3), x2), x1))


def linear_to_db(arr):
    out = np.full_like(arr, np.nan, dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    out[valid] = 10 * np.log10(arr[valid])
    return out


def convert_sar(arr):
    arr = arr.copy()
    for ch in (0, 1):
        if np.isfinite(np.nanmedian(arr[:, ch])) and np.nanmedian(arr[:, ch]) > 0:
            arr[:, ch] = linear_to_db(arr[:, ch])
    return arr


def prepare_data(rootpath, seed):
    data_dir = os.path.join(rootpath, "processed")
    Xtr = convert_sar(np.load(os.path.join(data_dir, "trn_X_VV_VH_RVI_raw.npy")))
    Ytr = np.load(os.path.join(data_dir, "trn_Y_MNDWI_NDVI_NDWI_raw.npy"))
    Xte = convert_sar(np.load(os.path.join(data_dir, "tst_X_VV_VH_RVI_raw.npy")))
    Yte = np.load(os.path.join(data_dir, "tst_Y_MNDWI_NDVI_NDWI_raw.npy"))

    xmean, xstd = np.nanmean(Xtr, axis=(0, 2, 3)), np.nanstd(Xtr, axis=(0, 2, 3))
    ymean, ystd = np.nanmean(Ytr, axis=(0, 2, 3)), np.nanstd(Ytr, axis=(0, 2, 3))

    def mask(a, mean, std):
        a = a.copy(); lo = mean[None, :, None, None] - 5 * std[None, :, None, None]
        hi = mean[None, :, None, None] + 5 * std[None, :, None, None]
        a[(a < lo) | (a > hi)] = np.nan
        return a

    Xtr, Xte = mask(Xtr, xmean, xstd), mask(Xte, xmean, xstd)
    Ytr, Yte = mask(Ytr, ymean, ystd), mask(Yte, ymean, ystd)
    xmin, xmax = np.nanpercentile(Xtr, 1, axis=(0, 2, 3)), np.nanpercentile(Xtr, 99, axis=(0, 2, 3))
    ymin, ymax = np.nanpercentile(Ytr, 1, axis=(0, 2, 3)), np.nanpercentile(Ytr, 99, axis=(0, 2, 3))

    def scale(a, amin, amax):
        return np.clip((a - amin[None, :, None, None]) /
                       (amax - amin)[None, :, None, None], 0, 1)

    Xtrn, Ytrn = scale(Xtr, xmin, xmax), scale(Ytr, ymin, ymax)
    Xten, Yten = scale(Xte, xmin, xmax), scale(Yte, ymin, ymax)
    # Reproduce the training script's RNG sequence exactly.  The seemingly
    # unused randint call exists in RMSD_model_bidirectional.py before shuffle.
    random.seed(seed); np.random.seed(seed)
    np.random.randint(seed)
    idx = np.arange(len(Xtrn)); np.random.shuffle(idx)
    val_idx = idx[int(len(idx) * 0.8):]
    return (Xtrn[val_idx], Ytrn[val_idx]), (Xten, Yten), (xmin, xmax, ymin, ymax)


def tensors(X, Y):
    mx, my = np.isfinite(X), np.isfinite(Y)
    return tuple(torch.from_numpy(a).float() for a in
                 (np.nan_to_num(X), np.nan_to_num(Y), mx, my))


def build_model(config):
    if config.get("model") == "light_attention":
        return LightAttentionUNet(
            3, 3,
            int(config.get("base_channels", 32)),
            float(config.get("do", 0)),
            use_depthwise=bool(config.get("arch_use_depthwise", True)),
            use_residual=bool(config.get("arch_use_residual", True)),
            use_eca=bool(config.get("arch_use_eca", True)),
            use_skip_attention=bool(config.get("arch_use_skip_attention", True)),
            bottleneck_dilation=int(config.get("arch_bottleneck_dilation", 2)),
            normalization=config.get("arch_norm", "group"),
            activation=config.get("arch_activation", "silu"),
        )
    return UNet()


def metric_rows(pred, true, mask, direction, names, mins, maxs):
    rows = []
    pred = pred * (maxs - mins)[None, :, None, None] + mins[None, :, None, None]
    true = true * (maxs - mins)[None, :, None, None] + mins[None, :, None, None]
    for ch, name in enumerate(names):
        valid = mask[:, ch] & np.isfinite(pred[:, ch]) & np.isfinite(true[:, ch])
        p, t = pred[:, ch][valid].astype(np.float64), true[:, ch][valid].astype(np.float64)
        err = p - t; bias = err.mean(); rmse = np.sqrt(np.mean(err ** 2))
        rows.append({"direction": direction, "channel": name, "n": p.size,
                     "corr": np.corrcoef(p, t)[0, 1], "rmse": rmse,
                     "ubRMSD": np.sqrt(max(rmse ** 2 - bias ** 2, 0)), "bias": bias})
    return rows


def masked_ssim_sums(prediction, target, mask, window_size=11, min_valid_fraction=0.8):
    """Return per-channel sums/counts of masked local SSIM on normalized [0, 1] data."""
    padding = window_size // 2
    weight = F.avg_pool2d(mask.float(), window_size, stride=1, padding=padding)
    denom = weight.clamp_min(1e-8)
    mean_p = F.avg_pool2d(prediction * mask, window_size, 1, padding) / denom
    mean_t = F.avg_pool2d(target * mask, window_size, 1, padding) / denom
    mean_pp = F.avg_pool2d(prediction.square() * mask, window_size, 1, padding) / denom
    mean_tt = F.avg_pool2d(target.square() * mask, window_size, 1, padding) / denom
    mean_pt = F.avg_pool2d((prediction * target) * mask, window_size, 1, padding) / denom
    var_p = (mean_pp - mean_p.square()).clamp_min(0)
    var_t = (mean_tt - mean_t.square()).clamp_min(0)
    cov = mean_pt - mean_p * mean_t
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mean_p * mean_t + c1) * (2 * cov + c2) /
            ((mean_p.square() + mean_t.square() + c1) * (var_p + var_t + c2)))
    valid = (weight >= min_valid_fraction) & mask.bool() & torch.isfinite(ssim)
    sums, counts = [], []
    for channel in range(prediction.shape[1]):
        channel_values = ssim[:, channel][valid[:, channel]]
        sums.append(channel_values.sum().item())
        counts.append(channel_values.numel())
    return sums, counts


def evaluate_test_streaming(model_xy, model_yx, data, device, bs, label, scales,
                            include_ssim=False):
    """Accumulate exact pixel-wise metrics without retaining all 512x512 predictions."""
    xmin, xmax, ymin, ymax = scales
    specs = [("X_to_Y", ["MNDWI", "NDVI", "NDWI"], ymin, ymax),
             ("Y_to_X", ["VV", "VH", "RVI"], xmin, xmax)]
    stats = {(direction, name): np.zeros(7, dtype=np.float64)
             for direction, names, _, _ in specs for name in names}
    ssim_stats = {(direction, name): np.zeros(2, dtype=np.float64)
                  for direction, names, _, _ in specs for name in names}
    loader = DataLoader(TensorDataset(*tensors(*data)), batch_size=bs, shuffle=False)
    with torch.no_grad():
        for X, Y, MX, MY in tqdm(loader, desc=label, leave=False):
            Xd, Yd = X.to(device), Y.to(device)
            PYd, PXd = model_xy(Xd), model_yx(Yd)
            if include_ssim:
                for pred_t, true_t, mask_t, (direction, names, _, _) in (
                    (PYd, Yd, MY.to(device), specs[0]),
                    (PXd, Xd, MX.to(device), specs[1]),
                ):
                    sums, counts = masked_ssim_sums(pred_t, true_t, mask_t)
                    for channel, name in enumerate(names):
                        ssim_stats[(direction, name)] += (sums[channel], counts[channel])
            PY = PYd.cpu().numpy(); PX = PXd.cpu().numpy()
            for pred, true, mask, (direction, names, mins, maxs) in (
                (PY, Y.numpy(), MY.numpy().astype(bool), specs[0]),
                (PX, X.numpy(), MX.numpy().astype(bool), specs[1]),
            ):
                pred = pred * (maxs - mins)[None, :, None, None] + mins[None, :, None, None]
                true = true * (maxs - mins)[None, :, None, None] + mins[None, :, None, None]
                for ch, name in enumerate(names):
                    valid = mask[:, ch] & np.isfinite(pred[:, ch]) & np.isfinite(true[:, ch])
                    p = pred[:, ch][valid].astype(np.float64); t = true[:, ch][valid].astype(np.float64)
                    err = p - t
                    stats[(direction, name)] += (p.size, p.sum(), t.sum(), np.square(p).sum(),
                                                 np.square(t).sum(), (p * t).sum(), np.square(err).sum())
                    # Bias needs signed error; store it separately in the final slot by expanding stats.
                    key = (direction, name)
                    if key not in evaluate_test_streaming.error_sums:
                        evaluate_test_streaming.error_sums[key] = 0.0
                    evaluate_test_streaming.error_sums[key] += err.sum()
    rows = []
    for (direction, name), (n, sp, st, sp2, st2, spt, se2) in stats.items():
        mp, mt = sp / n, st / n
        vp, vt = max(sp2 / n - mp ** 2, 0), max(st2 / n - mt ** 2, 0)
        cov = spt / n - mp * mt; bias = evaluate_test_streaming.error_sums[(direction, name)] / n
        rmse = np.sqrt(se2 / n)
        row = {"direction": direction, "channel": name, "n": int(n),
               "corr": cov / np.sqrt(vp * vt), "rmse": rmse,
               "ubRMSD": np.sqrt(max(rmse ** 2 - bias ** 2, 0)), "bias": bias}
        if include_ssim:
            ssim_sum, ssim_count = ssim_stats[(direction, name)]
            row["ssim"] = ssim_sum / ssim_count if ssim_count else np.nan
        rows.append(row)
    return rows


evaluate_test_streaming.error_sums = {}


def predict_all(model_xy, model_yx, data, device, bs, label):
    loader = DataLoader(TensorDataset(*tensors(*data)), batch_size=bs, shuffle=False)
    py, px, ys, xs, mys, mxs = [], [], [], [], [], []
    with torch.no_grad():
        for X, Y, MX, MY in tqdm(loader, desc=label, leave=False):
            Xd, Yd = X.to(device), Y.to(device)
            py.append(model_xy(Xd).cpu().numpy()); px.append(model_yx(Yd).cpu().numpy())
            xs.append(X.numpy()); ys.append(Y.numpy()); mxs.append(MX.numpy().astype(bool)); mys.append(MY.numpy().astype(bool))
    return tuple(np.concatenate(v) for v in (xs, ys, px, py, mxs, mys))


def predict_index_zero(model_xy, model_yx, data, device):
    X, Y, MX, MY = (value[:1] for value in tensors(*data))
    with torch.no_grad():
        PY = model_xy(X.to(device)).cpu().numpy()
        PX = model_yx(Y.to(device)).cpu().numpy()
    return X.numpy(), Y.numpy(), PX, PY, MX.numpy().astype(bool), MY.numpy().astype(bool)


def add_gaussian_noise_to_sar_prediction(prediction, mask, std, seed):
    """Add reproducible Gaussian noise to normalized Y->X output for visual experiments."""
    rng = np.random.default_rng(seed)
    noisy = prediction.astype(np.float32, copy=True)
    noise = rng.normal(0.0, std, size=noisy.shape).astype(np.float32)
    noisy[mask] = np.clip(noisy[mask] + noise[mask], 0.0, 1.0)
    return noisy


def plot_val0(values, path, title):
    X, Y, PX, PY, MX, MY = values
    arrays = [(X, MX, "X true", ["VV", "VH", "RVI"]),
              (PX, MX, "X pred (Y→X)", ["VV", "VH", "RVI"]),
              (Y, MY, "Y true", ["MNDWI", "NDVI", "NDWI"]),
              (PY, MY, "Y pred (X→Y)", ["MNDWI", "NDVI", "NDWI"])]
    fig, axes = plt.subplots(4, 3, figsize=(10, 11), dpi=160, constrained_layout=True)
    for r, (arr, mask, row, names) in enumerate(arrays):
        for c in range(3):
            image = arr[0, c].copy(); image[~mask[0, c]] = np.nan
            im = axes[r, c].imshow(image, cmap="viridis", vmin=0, vmax=1)
            axes[r, c].set_title(f"{row}: {names[c]}", fontsize=9)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            fig.colorbar(im, ax=axes[r, c], fraction=.046, pad=.03)
    fig.suptitle(title); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def plot_validation_all_schemes(validation_results, path, epoch, direction="X_to_Y"):
    """Combine every scheme's prediction and target for one translation direction."""
    if not validation_results:
        return

    def ablation_number(name):
        short = name.replace("unet_ablation_", "").replace("ablation_", "")
        try:
            return int(short.split("_", 1)[0].lstrip("A"))
        except ValueError:
            return 999

    light = sorted((e for e in validation_results if not e.startswith("unet_")), key=ablation_number)
    unet = sorted((e for e in validation_results if e.startswith("unet_")), key=ablation_number)
    experiment_columns = light + unet
    if not experiment_columns:
        return

    if direction == "X_to_Y":
        target_key, prediction_key, mask_key = "y_target", "y_prediction", "y_mask"
        channel_names = ("MNDWI", "NDVI", "NDWI")
        direction_title = "SAR → optical indices (X → Y)"
        colorbar_label = "Normalized optical index"
    elif direction in ("Y_to_X", "Y_to_X_noisy"):
        target_key, prediction_key, mask_key = "x_target", "x_prediction", "x_mask"
        channel_names = ("VV", "VH", "RVI")
        if direction == "Y_to_X_noisy":
            prediction_key = "x_prediction_noisy"
            direction_title = "Optical indices → SAR + Gaussian noise (Y → X)"
        else:
            direction_title = "Optical indices → SAR (Y → X)"
        colorbar_label = "Normalized SAR channel"
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    reference = validation_results[experiment_columns[0]]
    target, target_mask = reference[target_key], reference[mask_key]
    ncols = len(experiment_columns) + 1
    fig, axes = plt.subplots(
        3, ncols,
        figsize=(max(16, 2.05 * ncols), 7.4),
        dpi=180,
        constrained_layout=True,
        squeeze=False,
    )

    last_image = None
    for col, experiment in enumerate(experiment_columns):
        result = validation_results[experiment]
        for ch, channel in enumerate(channel_names):
            ax = axes[ch, col]
            image = result[prediction_key][ch].copy()
            valid = result[mask_key][ch] & np.isfinite(image) & np.isfinite(result[target_key][ch])
            image[~result[mask_key][ch]] = np.nan
            last_image = ax.imshow(image, cmap="viridis", vmin=0, vmax=1)
            corr = np.corrcoef(image[valid], result[target_key][ch][valid])[0, 1] if np.any(valid) else np.nan
            ax.text(.025, .975, f"R={corr:.2f}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, bbox={"facecolor": "white", "alpha": .72, "edgecolor": "none", "pad": 1.2})
            ax.set_xticks([]); ax.set_yticks([])
            if ch == 0:
                ax.set_title(scheme_label(experiment).replace("\n", " "), fontsize=8)
            if col == 0:
                ax.set_ylabel(channel, fontsize=10, fontweight="bold")

    target_col = ncols - 1
    for ch, channel in enumerate(channel_names):
        ax = axes[ch, target_col]
        image = target[ch].copy(); image[~target_mask[ch]] = np.nan
        last_image = ax.imshow(image, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if ch == 0:
            ax.set_title("Target", fontsize=9, fontweight="bold")

    # Architecture group labels follow the example layout without divider lines.
    if light:
        fig.text((len(light) / 2) / ncols, 1.015, "LightAttention", ha="center", fontsize=12, fontweight="bold")
    if unet:
        fig.text((len(light) + len(unet) / 2) / ncols, 1.015, "UNet", ha="center", fontsize=12, fontweight="bold")
    fig.text((ncols - .5) / ncols, 1.015, "Reference", ha="center", fontsize=12, fontweight="bold")
    fig.colorbar(last_image, ax=axes.ravel().tolist(), orientation="horizontal",
                 fraction=.035, pad=.025, aspect=60, label=colorbar_label)
    fig.suptitle(f"Validation index 0 — {direction_title} — epoch {epoch}", y=1.06, fontsize=14)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def scheme_label(experiment):
    """Turn folder names into compact labels used on the x axis."""
    name = experiment.replace("unet_ablation_", "").replace("ablation_", "")
    parts = name.split("_", 1)
    return parts[0] + ("\n" + parts[1].replace("_", " ").title() if len(parts) > 1 else "")


def summarize_xy_metrics(rows):
    """Mean over the three X->Y target channels for each scheme."""
    summary = []
    experiments = sorted({r["experiment"] for r in rows if r["direction"] == "X_to_Y"})
    for experiment in experiments:
        selected = [r for r in rows if r["experiment"] == experiment and r["direction"] == "X_to_Y"]
        item = {"experiment": experiment, "architecture": "UNet" if experiment.startswith("unet_") else "LightAttention"}
        for metric in ("corr", "rmse", "ubRMSD", "bias"):
            item[f"mean_{metric}"] = float(np.mean([float(r[metric]) for r in selected]))
        summary.append(item)
    return summary


def shared_metric_limits(rows):
    """Compute common y limits used by every architecture comparison figure."""
    xy = [r for r in rows if r["direction"] == "X_to_Y"]
    limits = {}
    for metric in ("corr", "ubRMSD", "bias"):
        values = np.asarray([float(r[metric]) for r in xy], dtype=float)
        if metric == "corr":
            # Correlation has a fixed, directly interpretable scale.
            limits[metric] = (0.0, 1.0)
        elif metric == "ubRMSD":
            limits[metric] = (0.0, max(values.max() * 1.12, 0.01))
        else:
            # A symmetric bias axis makes positive/negative errors comparable.
            bound = max(np.max(np.abs(values)) * 1.18, 0.01)
            limits[metric] = (-bound, bound)
    return limits


def plot_scheme_comparison(rows, path, title, y_limits=None):
    """Example-style bars (channel mean) with colored channel observations."""
    xy = [r for r in rows if r["direction"] == "X_to_Y"]
    experiments = sorted({r["experiment"] for r in xy})
    if not experiments:
        return
    channels = ("MNDWI", "NDVI", "NDWI")
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    panels = (("corr", "(a) Correlation R [-]"),
              ("ubRMSD", "(b) ubRMSD [-]"),
              ("bias", "(c) Bias [-]"))
    lookup = {(r["experiment"], r["channel"]): r for r in xy}
    x = np.arange(len(experiments))
    fig, axes = plt.subplots(1, 3, figsize=(max(15, len(experiments) * 2.3), 5.8),
                             dpi=180, constrained_layout=True)
    for ax, (metric, panel_title) in zip(axes, panels):
        channel_values = np.array([
            [float(lookup[(experiment, channel)][metric]) for channel in channels]
            for experiment in experiments
        ])
        means = channel_values.mean(axis=1)
        bars = ax.bar(x, means, width=.78, color="lightgray", edgecolor="0.25",
                      linewidth=1.0, label="Mean", zorder=2)
        offsets = (-.17, 0, .17)
        for j, (channel, color) in enumerate(zip(channels, colors)):
            ax.scatter(x + offsets[j], channel_values[:, j], s=48, color=color,
                       edgecolor="0.2", linewidth=.6, label=channel, zorder=4)
        ax.axhline(0, color="0.3", linewidth=.8, zorder=1)
        ax.set_title(panel_title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([scheme_label(e) for e in experiments], fontsize=9)
        ax.grid(axis="y", alpha=.25, zorder=0)
        if y_limits is not None:
            ax.set_ylim(*y_limits[metric])
        span = max(np.ptp(np.r_[channel_values.ravel(), means]), .01)
        for bar, mean in zip(bars, means):
            if metric == "bias":
                offset = .035 * span if mean >= 0 else -.035 * span
                va = "bottom" if mean >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, mean + offset, f"{mean:.3f}",
                        ha="center", va=va, fontsize=8, zorder=5)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, max(0, mean * .05), f"{mean:.3f}",
                        ha="center", va="bottom", fontsize=8, zorder=5)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(ch) for ch in channels] + [labels.index("Mean")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", bbox_to_anchor=(.5, 1.035),
               ncol=4, frameon=False, fontsize=10)
    fig.suptitle(title, y=1.09, fontsize=14)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def save_summary_and_plots(rows, output_dir):
    summary = summarize_xy_metrics(rows)
    save_csv(summary, os.path.join(output_dir, "scheme_mean_test_metrics.csv"))
    y_limits = shared_metric_limits(rows)
    groups = {
        "all": rows,
        "light_attention": [r for r in rows if not r["experiment"].startswith("unet_")],
        "unet": [r for r in rows if r["experiment"].startswith("unet_")],
    }
    titles = {"all": "Epoch 100 ablation schemes", "light_attention": "LightAttention ablation schemes",
              "unet": "UNet ablation schemes"}
    for name, group_rows in groups.items():
        if group_rows:
            plot_scheme_comparison(
                group_rows,
                os.path.join(output_dir, f"scheme_comparison_{name}.png"),
                titles[name],
                y_limits=y_limits,
            )


def load_metrics_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    numeric = ("epoch", "n", "corr", "rmse", "ubRMSD", "bias")
    for row in rows:
        for key in numeric:
            if key in row:
                row[key] = int(row[key]) if key in ("epoch", "n") else float(row[key])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study_dir", default="/Users/jslee/Downloads/SEN12FLOOD/output/ablation_study")
    p.add_argument("--rootpath", default="/Users/jslee/Downloads/SEN12FLOOD")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--experiment", default="*", help="Experiment folder glob, e.g. 'unet_*'.")
    p.add_argument("--plot_only", action="store_true", help="Rebuild plots from all_test_metrics.csv without inference.")
    p.add_argument("--sar_noise_std", type=float, default=0.04,
                   help="Gaussian noise std added only to normalized Y->X validation visualizations; 0 disables it.")
    p.add_argument("--sar_noise_seed", type=int, default=2026,
                   help="Seed for reproducible Gaussian SAR visualization noise.")
    # Jupyter/IPython injects its own arguments (for example --f=kernel.json).
    # Keep this script runnable both from a terminal and from an IDE notebook cell.
    args, unknown_args = p.parse_known_args()
    if unknown_args:
        print(f"Ignoring notebook/IDE arguments: {unknown_args}")
    if args.sar_noise_std < 0:
        p.error("--sar_noise_std must be zero or positive")
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available()
                          else "cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    out = args.output_dir or os.path.join(args.study_dir, f"epoch_{args.epoch:03d}_evaluation")
    os.makedirs(out, exist_ok=True)
    if args.plot_only:
        metrics_path = os.path.join(out, "all_test_metrics.csv")
        rows = load_metrics_csv(metrics_path)
        save_summary_and_plots(rows, out)
        print(f"Rebuilt scheme summary plots from: {metrics_path}")
        return
    val, test, scales = prepare_data(args.rootpath, args.seed)
    xmin, xmax, ymin, ymax = scales
    all_rows = []
    validation_results = {}
    checkpoints = sorted(glob.glob(os.path.join(args.study_dir, args.experiment, f"model_epoch_{args.epoch:03d}.pt")))
    if not checkpoints:
        raise FileNotFoundError("No epoch checkpoint found")
    for experiment_index, checkpoint_path in enumerate(checkpoints):
        exp_dir = os.path.dirname(checkpoint_path); experiment = os.path.basename(exp_dir)
        config_path = glob.glob(os.path.join(exp_dir, "config_*.json"))[0]
        with open(config_path, encoding="utf-8") as f: config = json.load(f)
        print(f"[{experiment}] loading {os.path.basename(checkpoint_path)}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_xy, model_yx = build_model(config).to(device), build_model(config).to(device)
        model_xy.load_state_dict(checkpoint["model_xy_state_dict"]); model_yx.load_state_dict(checkpoint["model_yx_state_dict"])
        model_xy.eval(); model_yx.eval()
        val_values = predict_index_zero(model_xy, model_yx, val, device)
        plot_val0(val_values, os.path.join(out, f"{experiment}_validation_index_0.png"),
                  f"{experiment} | epoch {args.epoch} | validation index 0")
        X_val, Y_val, PX_val, PY_val, MX_val, MY_val = val_values
        validation_results[experiment] = {
            "x_target": X_val[0],
            "x_prediction": PX_val[0],
            "x_prediction_noisy": add_gaussian_noise_to_sar_prediction(
                PX_val[0], MX_val[0], args.sar_noise_std,
                args.sar_noise_seed + experiment_index,
            ),
            "x_mask": MX_val[0],
            "y_target": Y_val[0],
            "y_prediction": PY_val[0],
            "y_mask": MY_val[0],
        }
        evaluate_test_streaming.error_sums = {}
        rows = evaluate_test_streaming(model_xy, model_yx, test, device, args.bs,
                                       f"{experiment} test", scales)
        for row in rows: row["experiment"] = experiment; row["epoch"] = args.epoch
        rows = [{"experiment": r.pop("experiment"), "epoch": r.pop("epoch"), **r} for r in rows]
        save_csv(rows, os.path.join(out, f"{experiment}_test_metrics.csv")); all_rows.extend(rows)
        del model_xy, model_yx
        if device.type == "mps": torch.mps.empty_cache()
    save_csv(all_rows, os.path.join(out, "all_test_metrics.csv"))
    save_summary_and_plots(all_rows, out)
    plot_validation_all_schemes(
        validation_results,
        os.path.join(out, "validation_index_0_all_schemes.png"),
        args.epoch,
    )
    light_validation_results = {
        name: result for name, result in validation_results.items()
        if not name.startswith("unet_")
    }
    unet_validation_results = {
        name: result for name, result in validation_results.items()
        if name.startswith("unet_")
    }
    plot_validation_all_schemes(
        light_validation_results,
        os.path.join(out, "validation_index_0_light_attention.png"),
        args.epoch,
    )
    plot_validation_all_schemes(
        unet_validation_results,
        os.path.join(out, "validation_index_0_unet.png"),
        args.epoch,
    )
    plot_validation_all_schemes(
        validation_results,
        os.path.join(out, "validation_index_0_y_to_x_all_schemes.png"),
        args.epoch,
        direction="Y_to_X",
    )
    plot_validation_all_schemes(
        light_validation_results,
        os.path.join(out, "validation_index_0_y_to_x_light_attention.png"),
        args.epoch,
        direction="Y_to_X",
    )
    plot_validation_all_schemes(
        unet_validation_results,
        os.path.join(out, "validation_index_0_y_to_x_unet.png"),
        args.epoch,
        direction="Y_to_X",
    )
    if args.sar_noise_std > 0:
        noise_tag = f"std{args.sar_noise_std:g}".replace(".", "p")
        plot_validation_all_schemes(
            validation_results,
            os.path.join(out, f"validation_index_0_y_to_x_gaussian_{noise_tag}_all_schemes.png"),
            args.epoch,
            direction="Y_to_X_noisy",
        )
        plot_validation_all_schemes(
            light_validation_results,
            os.path.join(out, f"validation_index_0_y_to_x_gaussian_{noise_tag}_light_attention.png"),
            args.epoch,
            direction="Y_to_X_noisy",
        )
        plot_validation_all_schemes(
            unet_validation_results,
            os.path.join(out, f"validation_index_0_y_to_x_gaussian_{noise_tag}_unet.png"),
            args.epoch,
            direction="Y_to_X_noisy",
        )
    with open(os.path.join(out, "evaluation_info.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": args.epoch, "validation_index": 0, "seed": args.seed,
                   "sar_noise_std": args.sar_noise_std, "sar_noise_seed": args.sar_noise_seed,
                   "device": str(device), "experiments": len(checkpoints)}, f, indent=2)
    print(f"Saved evaluation results to: {out}")


if __name__ == "__main__":
    main()
