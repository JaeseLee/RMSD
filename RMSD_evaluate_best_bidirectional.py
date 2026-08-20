import argparse
import csv
import glob
import json
import os
import random

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
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def find_latest_checkpoint(rootpath):
    pattern = os.path.join(rootpath, "output", "*", "best_model.pt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No best_model.pt found under {pattern}")
    return max(matches, key=os.path.getmtime)


def resolve_checkpoint(rootpath, date_out, checkpoint):
    if checkpoint:
        return os.path.abspath(os.path.expanduser(checkpoint))
    if date_out:
        return os.path.join(rootpath, "output", date_out, "best_model.pt")
    return find_latest_checkpoint(rootpath)


def load_train_config(checkpoint_path):
    ckpt_dir = os.path.dirname(checkpoint_path)
    config_files = sorted(glob.glob(os.path.join(ckpt_dir, "config_*.json")))
    if not config_files:
        return {}
    with open(config_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def load_loss_history(checkpoint_path):
    csv_path = os.path.join(os.path.dirname(checkpoint_path), "loss_history.csv")
    if not os.path.exists(csv_path):
        return None, csv_path

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value) if value not in ("", None) else np.nan
            rows.append(parsed)

    return rows, csv_path


def plot_loss_history(loss_history, output_dir, show=False):
    if not loss_history:
        return None

    def values(name):
        return np.array([row.get(name, np.nan) for row in loss_history], dtype=np.float64)

    epochs = values("epoch")
    has_components = any(
        key in loss_history[0]
        for key in ["train_xy_loss", "train_yx_loss", "train_kl_loss", "val_xy_loss", "val_yx_loss", "val_kl_loss"]
    )

    if has_components:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=160, constrained_layout=True)
        ax_total, ax_components = axes
    else:
        fig, ax_total = plt.subplots(figsize=(10, 5), dpi=160, constrained_layout=True)
        ax_components = None

    ax_total.plot(epochs, values("train_loss"), label="train loss", linewidth=2)
    ax_total.plot(epochs, values("val_loss"), label="val loss", linewidth=2)
    if "best_val_loss" in loss_history[0]:
        ax_total.plot(epochs, values("best_val_loss"), label="best val loss", linewidth=1.5, linestyle="--")
    ax_total.set_title("Training loss history")
    ax_total.set_xlabel("Epoch")
    ax_total.set_ylabel("Loss")
    ax_total.grid(True, alpha=0.3)
    ax_total.legend()

    if ax_components is not None:
        component_specs = [
            ("train_xy_loss", "train xy"),
            ("val_xy_loss", "val xy"),
            ("train_yx_loss", "train yx"),
            ("val_yx_loss", "val yx"),
            ("train_kl_loss", "train kl"),
            ("val_kl_loss", "val kl"),
        ]
        for key, label in component_specs:
            if key in loss_history[0]:
                linestyle = "--" if key.startswith("val") else "-"
                ax_components.plot(epochs, values(key), label=label, linewidth=1.6, linestyle=linestyle)
        ax_components.set_title("Bidirectional loss components")
        ax_components.set_xlabel("Epoch")
        ax_components.set_ylabel("Loss")
        ax_components.grid(True, alpha=0.3)
        ax_components.legend(ncol=3)

    output_path = os.path.join(output_dir, "loss_history_plot.png")
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def get_device(device_name, gpu):
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_masked_loss(name):
    losses = {
        "L1": nn.L1Loss,
        "MAE": nn.L1Loss,
        "MSE": nn.MSELoss,
        "SmoothL1": nn.SmoothL1Loss,
    }
    if name not in losses:
        raise ValueError(f"Unsupported loss: {name}. Choose from {list(losses)}")

    base_loss = losses[name](reduction="none")

    def masked_loss(pred, target, mask):
        loss = base_loss(pred, target)
        mask = mask.to(dtype=torch.bool, device=loss.device)
        if not torch.any(mask):
            return pred.sum() * 0.0
        return loss[mask].mean()

    return masked_loss


def soft_histogram_kl_loss(pred, target, mask, bins=32, sigma=0.03, value_range=(0.0, 1.0), eps=1e-6):
    mask = mask.to(dtype=torch.bool, device=pred.device)
    bin_centers = torch.linspace(
        value_range[0],
        value_range[1],
        bins,
        device=pred.device,
        dtype=pred.dtype,
    )

    kl_losses = []
    for ch in range(pred.shape[1]):
        ch_mask = mask[:, ch]
        if not torch.any(ch_mask):
            continue

        pred_vals = pred[:, ch][ch_mask].reshape(-1, 1).clamp(value_range[0], value_range[1])
        target_vals = target[:, ch][ch_mask].reshape(-1, 1).clamp(value_range[0], value_range[1])
        pred_weights = torch.exp(-0.5 * ((pred_vals - bin_centers) / sigma) ** 2)
        target_weights = torch.exp(-0.5 * ((target_vals - bin_centers) / sigma) ** 2)
        pred_hist = pred_weights.sum(dim=0)
        target_hist = target_weights.sum(dim=0)
        pred_prob = pred_hist / (pred_hist.sum() + eps)
        target_prob = target_hist / (target_hist.sum() + eps)
        kl_losses.append(F.kl_div(torch.log(pred_prob + eps), target_prob, reduction="sum"))

    if len(kl_losses) == 0:
        return pred.sum() * 0.0
    return torch.stack(kl_losses).mean()


def linear_to_db_safe(sar):
    sar = sar.astype(np.float32, copy=True)
    out = np.full_like(sar, np.nan, dtype=np.float32)
    valid = np.isfinite(sar) & (sar > 0)
    out[valid] = 10.0 * np.log10(sar[valid])
    return out


def convert_sar_channels_to_db(arr, channels=(0, 1)):
    arr = arr.copy()
    for ch in channels:
        ch_data = arr[:, ch]
        ch_median = np.nanmedian(ch_data)
        if np.isfinite(ch_median) and ch_median > 0:
            arr[:, ch] = linear_to_db_safe(ch_data)
        else:
            arr[:, ch] = ch_data
    return arr


def mask_outside_nsigma(arr, mean, std, nsigma=5.0):
    arr = arr.copy()
    mean = np.asarray(mean).reshape(1, -1, 1, 1)
    std = np.asarray(std).reshape(1, -1, 1, 1)
    bad = (arr < mean - nsigma * std) | (arr > mean + nsigma * std)
    arr[bad] = np.nan
    return arr


def scale_with_checkpoint(arr, arr_min, arr_max):
    arr_min = np.asarray(arr_min).reshape(1, -1, 1, 1)
    arr_max = np.asarray(arr_max).reshape(1, -1, 1, 1)
    arr_range = np.where(arr_max == arr_min, 1.0, arr_max - arr_min)
    return np.clip((arr - arr_min) / arr_range, 0.0, 1.0)


def inverse_scale(arr, arr_min, arr_max):
    arr_min = np.asarray(arr_min).reshape(1, -1, 1, 1)
    arr_max = np.asarray(arr_max).reshape(1, -1, 1, 1)
    arr_range = np.where(arr_max == arr_min, 1.0, arr_max - arr_min)
    return arr * arr_range + arr_min


def make_bidirectional_tensors(X, Y):
    X_mask = np.isfinite(X)
    Y_mask = np.isfinite(Y)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float(),
        torch.from_numpy(X_mask),
        torch.from_numpy(Y_mask),
    )


def add_metric_stats(stats, pred, target, mask, prefix, channel_names):
    pred = pred.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    mask = mask.astype(bool, copy=False)

    for ch, name in enumerate(channel_names):
        valid = mask[:, ch] & np.isfinite(pred[:, ch]) & np.isfinite(target[:, ch])
        if not np.any(valid):
            continue
        p = pred[:, ch][valid]
        t = target[:, ch][valid]
        key = f"{prefix}_{name}"
        item = stats.setdefault(
            key,
            {
                "n": 0,
                "sum_p": 0.0,
                "sum_t": 0.0,
                "sum_p2": 0.0,
                "sum_t2": 0.0,
                "sum_pt": 0.0,
                "sum_err": 0.0,
                "sum_err2": 0.0,
            },
        )
        err = p - t
        item["n"] += p.size
        item["sum_p"] += p.sum()
        item["sum_t"] += t.sum()
        item["sum_p2"] += np.square(p).sum()
        item["sum_t2"] += np.square(t).sum()
        item["sum_pt"] += (p * t).sum()
        item["sum_err"] += err.sum()
        item["sum_err2"] += np.square(err).sum()


def finalize_metric_stats(stats):
    rows = []
    for key, item in sorted(stats.items()):
        n = item["n"]
        mean_p = item["sum_p"] / n
        mean_t = item["sum_t"] / n
        var_p = max(item["sum_p2"] / n - mean_p**2, 0.0)
        var_t = max(item["sum_t2"] / n - mean_t**2, 0.0)
        cov = item["sum_pt"] / n - mean_p * mean_t
        denom = np.sqrt(var_p * var_t)
        corr = np.nan if denom == 0 else cov / denom
        rows.append(
            {
                "metric_key": key,
                "n": n,
                "corr": corr,
                "rmse": np.sqrt(item["sum_err2"] / n),
                "ubRMSD": np.sqrt(max(var_p + var_t - 2.0 * cov, 0.0)),
                "bias": item["sum_err"] / n,
            }
        )
    return rows


def build_datasets(rootpath, checkpoint, seed):
    data_dir = os.path.join(rootpath, "processed")
    X_trn = np.load(os.path.join(data_dir, "trn_X_VV_VH_RVI_raw.npy"))
    Y_trn = np.load(os.path.join(data_dir, "trn_Y_MNDWI_NDVI_NDWI_raw.npy"))
    X_tst = np.load(os.path.join(data_dir, "tst_X_VV_VH_RVI_raw.npy"))
    Y_tst = np.load(os.path.join(data_dir, "tst_Y_MNDWI_NDVI_NDWI_raw.npy"))

    X_trn = convert_sar_channels_to_db(X_trn)
    X_tst = convert_sar_channels_to_db(X_tst)

    X_trn_3sig = mask_outside_nsigma(X_trn, checkpoint["X_mean"], checkpoint["X_std"], nsigma=5)
    X_tst_3sig = mask_outside_nsigma(X_tst, checkpoint["X_mean"], checkpoint["X_std"], nsigma=5)
    Y_trn_3sig = mask_outside_nsigma(Y_trn, checkpoint["Y_mean"], checkpoint["Y_std"], nsigma=5)
    Y_tst_3sig = mask_outside_nsigma(Y_tst, checkpoint["Y_mean"], checkpoint["Y_std"], nsigma=5)

    X_trn_norm = scale_with_checkpoint(X_trn_3sig, checkpoint["X_min"], checkpoint["X_max"])
    X_tst_norm = scale_with_checkpoint(X_tst_3sig, checkpoint["X_min"], checkpoint["X_max"])
    Y_trn_norm = scale_with_checkpoint(Y_trn_3sig, checkpoint["Y_min"], checkpoint["Y_max"])
    Y_tst_norm = scale_with_checkpoint(Y_tst_3sig, checkpoint["Y_min"], checkpoint["Y_max"])

    random.seed(seed)
    np.random.seed(seed)
    np.random.randint(seed)

    idx = np.arange(X_trn_norm.shape[0])
    np.random.shuffle(idx)
    idx_val = idx[int(idx.size * 0.8) :]

    return {
        "validation": make_bidirectional_tensors(X_trn_norm[idx_val], Y_trn_norm[idx_val]),
        "test": make_bidirectional_tensors(X_tst_norm, Y_tst_norm),
    }


def evaluate_split(
    split_name,
    tensors,
    model_xy,
    model_yx,
    criterion,
    device,
    batch_size,
    kl_weight,
    kl_bins,
    kl_sigma,
    checkpoint,
):
    dataset = TensorDataset(*tensors)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    losses = []
    xy_losses = []
    yx_losses = []
    kl_losses = []
    stats_norm = {}
    stats_physical = {}
    x_names = ["VV", "VH", "RVI"]
    y_names = ["MNDWI", "NDVI", "NDWI"]

    model_xy.eval()
    model_yx.eval()
    with torch.no_grad():
        for Xb, Yb, MXb, MYb in tqdm(loader, desc=f"{split_name}"):
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            MXb = MXb.to(device)
            MYb = MYb.to(device)

            pred_y = model_xy(Xb)
            pred_x = model_yx(Yb)
            loss_xy = criterion(pred_y, Yb, MYb)
            loss_yx = criterion(pred_x, Xb, MXb)
            kl_xy = soft_histogram_kl_loss(pred_y, Yb, MYb, bins=kl_bins, sigma=kl_sigma)
            kl_yx = soft_histogram_kl_loss(pred_x, Xb, MXb, bins=kl_bins, sigma=kl_sigma)
            kl_loss = kl_xy + kl_yx
            loss = loss_xy + loss_yx + kl_weight * kl_loss

            losses.append(loss.item())
            xy_losses.append(loss_xy.item())
            yx_losses.append(loss_yx.item())
            kl_losses.append(kl_loss.item())

            X_np = Xb.detach().cpu().numpy()
            Y_np = Yb.detach().cpu().numpy()
            pred_x_np = pred_x.detach().cpu().numpy()
            pred_y_np = pred_y.detach().cpu().numpy()
            MX_np = MXb.detach().cpu().numpy()
            MY_np = MYb.detach().cpu().numpy()

            add_metric_stats(stats_norm, pred_y_np, Y_np, MY_np, "XY_norm", y_names)
            add_metric_stats(stats_norm, pred_x_np, X_np, MX_np, "YX_norm", x_names)
            add_metric_stats(
                stats_physical,
                inverse_scale(pred_y_np, checkpoint["Y_min"], checkpoint["Y_max"]),
                inverse_scale(Y_np, checkpoint["Y_min"], checkpoint["Y_max"]),
                MY_np,
                "XY_physical",
                y_names,
            )
            add_metric_stats(
                stats_physical,
                inverse_scale(pred_x_np, checkpoint["X_min"], checkpoint["X_max"]),
                inverse_scale(X_np, checkpoint["X_min"], checkpoint["X_max"]),
                MX_np,
                "YX_physical",
                x_names,
            )

    summary = {
        "split": split_name,
        "loss": float(np.mean(losses)),
        "xy_loss": float(np.mean(xy_losses)),
        "yx_loss": float(np.mean(yx_losses)),
        "kl_loss": float(np.mean(kl_losses)),
        "n_batches": len(losses),
    }
    return summary, finalize_metric_stats(stats_norm) + finalize_metric_stats(stats_physical)


def save_results(rows, summaries, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "best_model_eval_results.json")
    csv_path = os.path.join(output_dir, "best_model_eval_metrics.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summaries, "metrics": rows}, f, indent=4)

    fieldnames = ["split", "metric_key", "n", "corr", "rmse", "ubRMSD", "bias"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def plot_validation_grid(
    X,
    Y_true,
    Y_pred,
    X_pred,
    X_mask,
    Y_mask,
    save_path,
    epoch,
    sample_idx=0,
    split_name="validation",
    show=False,
):
    X_true = X.detach().cpu().numpy()
    Y_true = Y_true.detach().cpu().numpy()
    Y_pred = Y_pred.detach().cpu().numpy()
    X_pred = X_pred.detach().cpu().numpy()
    X_mask = X_mask.detach().cpu().numpy().astype(bool)
    Y_mask = Y_mask.detach().cpu().numpy().astype(bool)

    sample_idx = min(sample_idx, X_true.shape[0] - 1)
    X_true_sample = X_true[sample_idx].copy()
    X_pred_sample = X_pred[sample_idx].copy()
    Y_true_sample = Y_true[sample_idx].copy()
    Y_pred_sample = Y_pred[sample_idx].copy()
    X_mask_sample = X_mask[sample_idx]
    Y_mask_sample = Y_mask[sample_idx]

    X_true_sample[~X_mask_sample] = np.nan
    X_pred_sample[~X_mask_sample] = np.nan
    Y_true_sample[~Y_mask_sample] = np.nan
    Y_pred_sample[~Y_mask_sample] = np.nan

    rows = [
        ("X_true", X_true_sample, ["VV", "VH", "RVI"]),
        ("X_pred (Y->X)", X_pred_sample, ["VV", "VH", "RVI"]),
        ("Y_true", Y_true_sample, ["MNDWI", "NDVI", "NDWI"]),
        ("Y_pred (X->Y)", Y_pred_sample, ["MNDWI", "NDVI", "NDWI"]),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(11, 12), dpi=180, constrained_layout=True)
    for row_idx, (row_name, arr, channel_names) in enumerate(rows):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(arr[col_idx], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{row_name}: {channel_names[col_idx]}", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(f"Bidirectional {split_name} sample - epoch {epoch + 1:03d}", fontsize=12)
    fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    
parser = argparse.ArgumentParser(description="Evaluate a bidirectional RMSD best checkpoint.")
parser.add_argument("--rootpath", type=str, default="/Users/jslee/Downloads/SEN12FLOOD/")
parser.add_argument("--date_out", type=str, default=None, help="Experiment folder name under rootpath/output.")
parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to best_model.pt.")
parser.add_argument("--bs", type=int, default=None, help="Batch size. Defaults to training config bs.")
parser.add_argument("--loss", type=str, default=None, help="Loss. Defaults to training config loss.")
parser.add_argument("--seed", type=int, default=None, help="Seed. Defaults to training config seed.")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--device", type=str, default="auto", help="auto, cpu, mps, cuda, cuda:0, ...")
parser.add_argument("--kl_weight", type=float, default=None)
parser.add_argument("--kl_bins", type=int, default=None)
parser.add_argument("--kl_sigma", type=float, default=None)
parser.add_argument("--save", type=str2bool, default=True)
parser.add_argument("--plot_loss", type=str2bool, default=True, help="Plot loss_history.csv from the checkpoint folder.")
parser.add_argument("--show_plot", type=str2bool, default=True, help="Show the loss plot window/notebook output.")
parser.add_argument("--plot_results", type=str2bool, default=True, help="Plot validation/test prediction result grids.")
parser.add_argument("--show_result_plot", type=str2bool, default=True, help="Show validation/test prediction result grids.")
parser.add_argument("--vis_sample_idx", type=int, default=4, help="Sample index in the first validation/test batch to visualize.")
args, _ = parser.parse_known_args()

rootpath = os.path.abspath(os.path.expanduser(args.rootpath))
checkpoint_path = resolve_checkpoint(rootpath, args.date_out, args.checkpoint)
train_config = load_train_config(checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

batch_size = args.bs if args.bs is not None else int(train_config.get("bs", 4))
loss_name = args.loss if args.loss is not None else train_config.get("loss", "L1")
seed = args.seed if args.seed is not None else int(train_config.get("seed", 42))
kl_weight = args.kl_weight if args.kl_weight is not None else float(train_config.get("kl_weight", 0.01))
kl_bins = args.kl_bins if args.kl_bins is not None else int(train_config.get("kl_bins", 32))
kl_sigma = args.kl_sigma if args.kl_sigma is not None else float(train_config.get("kl_sigma", 0.03))
device = get_device(args.device, args.gpu)

print(f"Checkpoint: {checkpoint_path}")
print(f"Device: {device}")
print(f"Config: bs={batch_size}, loss={loss_name}, seed={seed}, kl_weight={kl_weight}")
if "epoch" in checkpoint:
    print(f"Best epoch: {int(checkpoint['epoch']) + 1}")
if "val_loss" in checkpoint:
    print(f"Stored best val loss: {float(checkpoint['val_loss']):.6f}")

datasets = build_datasets(rootpath, checkpoint, seed)

model_name = train_config.get("model", "unet")
base_channels = int(train_config.get("base_channels", 32))

def build_model():
    if model_name == "light_attention":
        return LightAttentionUNet(
            n_channels=3,
            n_classes=3,
            base_channels=base_channels,
            dropout=float(train_config.get("do", 0.0)),
            use_depthwise=bool(train_config.get("arch_use_depthwise", True)),
            use_residual=bool(train_config.get("arch_use_residual", True)),
            use_eca=bool(train_config.get("arch_use_eca", True)),
            use_skip_attention=bool(train_config.get("arch_use_skip_attention", True)),
            bottleneck_dilation=int(train_config.get("arch_bottleneck_dilation", 2)),
            normalization=train_config.get("arch_norm", "group"),
            activation=train_config.get("arch_activation", "silu"),
        )
    return UNet(n_channels=3, n_classes=3, bilinear=True)


model_xy = build_model().to(device)
model_yx = build_model().to(device)
print(f"Architecture: {model_name}")
model_xy.load_state_dict(checkpoint["model_xy_state_dict"])
model_yx.load_state_dict(checkpoint["model_yx_state_dict"])
criterion = get_masked_loss(loss_name)



# loss_history, loss_csv_path = load_loss_history(checkpoint_path)
# if loss_history is None:
#     print(f"\nLoss history not found: {loss_csv_path}")
# else:
#     plot_path = plot_loss_history(
#         loss_history,
#         os.path.dirname(checkpoint_path),
#         show=args.show_plot,
#     )
#     print(f"Saved: {plot_path}")


if args.plot_results:
    vis_dir = os.path.join(os.path.dirname(checkpoint_path), "evaluation_figures")
    os.makedirs(vis_dir, exist_ok=True)
    best_epoch = int(checkpoint.get("epoch", -1))

    model_xy.eval()
    model_yx.eval()

    for split_name in ["validation", "test"]:
        loader_vis = DataLoader(
            TensorDataset(*datasets[split_name]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        Xb, Yb, MXb, MYb = next(iter(loader_vis))
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        MXb = MXb.to(device)
        MYb = MYb.to(device)

        with torch.no_grad():
            pred_y = model_xy(Xb)
            pred_x = model_yx(Yb)

        save_path = os.path.join(vis_dir, f"{split_name}_best_result.png")
        plot_validation_grid(
            Xb.detach().cpu(),
            Yb.detach().cpu(),
            pred_y.detach().cpu(),
            pred_x.detach().cpu(),
            MXb.detach().cpu(),
            MYb.detach().cpu(),
            save_path=save_path,
            epoch=best_epoch,
            sample_idx=args.vis_sample_idx,
            split_name=split_name,
            show=args.show_result_plot,
        )
        print(f"Saved: {save_path}")
