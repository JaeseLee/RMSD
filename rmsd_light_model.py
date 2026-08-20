"""Lightweight attention U-Net and structure-aware losses for Radar <-> MS."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels, maximum=8):
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _normalization(name, channels):
    if name == "group":
        return nn.GroupNorm(_groups(channels), channels)
    if name == "batch":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"Unsupported normalization: {name}")


def _activation(name):
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ECA(nn.Module):
    """Efficient Channel Attention: channel reweighting with very few parameters."""
    '''ECA-inspired channel attention with kernal size (k) = 3'''

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        weights = self.pool(x).squeeze(-1).transpose(1, 2)
        weights = torch.sigmoid(self.conv(weights).transpose(1, 2).unsqueeze(-1))
        return x * weights


class DSResidualBlock(nn.Module):
    """Pointwise + depthwise residual block using GroupNorm for small batches."""

    def __init__(self, in_channels, out_channels, dilation=1, attention=True,
                 dropout=0.0, depthwise=True, residual=True,
                 normalization="group", activation="silu"):
        super().__init__()
        self.use_residual = residual
        self.project_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _normalization(normalization, out_channels),
            _activation(activation),
        )
        conv_groups = out_channels if depthwise else 1
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=conv_groups,
                bias=False,
            ),
            _normalization(normalization, out_channels),
            _activation(activation),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            _normalization(normalization, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.attention = ECA(out_channels) if attention else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.activation = _activation(activation)

    def forward(self, x):
        residual = self.skip(x) if self.use_residual else None
        x = self.depthwise(self.project_in(x))
        x = self.attention(x)
        x = self.dropout(x)
        return self.activation(residual + x if residual is not None else x)


class AttentionGate(nn.Module):
    """Suppress irrelevant encoder features before a decoder skip connection."""

    def __init__(self, skip_channels, gate_channels, activation="silu"):
        super().__init__()
        hidden = max(min(skip_channels, gate_channels) // 2, 8)
        self.skip_proj = nn.Conv2d(skip_channels, hidden, 1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, hidden, 1, bias=False)
        self.mask = nn.Sequential(_activation(activation), nn.Conv2d(hidden, 1, 1), nn.Sigmoid())

    def forward(self, skip, gate):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return skip * self.mask(self.skip_proj(skip) + self.gate_proj(gate))


class LightDown(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, **block_kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DSResidualBlock(in_channels, out_channels, dropout=dropout, **block_kwargs),
        )

    def forward(self, x):
        return self.block(x)


class LightUp(nn.Module):
    def __init__(self, deep_channels, skip_channels, out_channels, use_gate=True,
                 dropout=0.0, gate_activation="silu", **block_kwargs):
        super().__init__()
        self.gate = AttentionGate(skip_channels, deep_channels, gate_activation) if use_gate else None
        self.block = DSResidualBlock(
            deep_channels + skip_channels, out_channels, dropout=dropout, **block_kwargs
        )

    def forward(self, deep, skip):
        if self.gate is not None:
            skip = self.gate(skip, deep)
        deep = F.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((skip, deep), dim=1))


class LightAttentionUNet(nn.Module):
    """Compact residual U-Net with ECA and gated deep skip connections."""

    def __init__(self, n_channels=3, n_classes=3, base_channels=32, dropout=0.0,
                 use_depthwise=True, use_residual=True, use_eca=True,
                 use_skip_attention=True, bottleneck_dilation=2,
                 normalization="group", activation="silu"):
        super().__init__()
        b = base_channels
        widths = (b, 2 * b, 4 * b, 8 * b, 12 * b)
        block_kwargs = {
            "attention": use_eca,
            "depthwise": use_depthwise,
            "residual": use_residual,
            "normalization": normalization,
            "activation": activation,
        }
        self.inc = DSResidualBlock(n_channels, widths[0], dropout=dropout, **block_kwargs)
        self.down1 = LightDown(widths[0], widths[1], dropout, **block_kwargs)
        self.down2 = LightDown(widths[1], widths[2], dropout, **block_kwargs)
        self.down3 = LightDown(widths[2], widths[3], dropout, **block_kwargs)
        self.down4 = LightDown(widths[3], widths[4], dropout, **block_kwargs)
        # Dilated bottleneck enlarges context without another downsampling stage.
        self.bottleneck = DSResidualBlock(
            widths[4], widths[4], dilation=bottleneck_dilation, dropout=dropout, **block_kwargs
        )
        self.up1 = LightUp(widths[4], widths[3], widths[3], use_gate=use_skip_attention,
                           dropout=dropout, gate_activation=activation, **block_kwargs)
        self.up2 = LightUp(widths[3], widths[2], widths[2], use_gate=use_skip_attention,
                           dropout=dropout, gate_activation=activation, **block_kwargs)
        self.up3 = LightUp(widths[2], widths[1], widths[1], use_gate=False,
                           dropout=dropout, **block_kwargs)
        self.up4 = LightUp(widths[1], widths[0], widths[0], use_gate=False,
                           dropout=dropout, **block_kwargs)
        self.outc = nn.Conv2d(widths[0], n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottleneck(self.down4(x4))
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def masked_gradient_loss(pred, target, mask):
    """L1 loss on horizontal/vertical gradients at jointly valid pixels."""
    mask = mask.bool()
    dx_mask = mask[..., :, 1:] & mask[..., :, :-1]
    dy_mask = mask[..., 1:, :] & mask[..., :-1, :]
    dx = (pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    terms = []
    if dx_mask.any():
        terms.append(dx.abs()[dx_mask].mean())
    if dy_mask.any():
        terms.append(dy.abs()[dy_mask].mean())
    return torch.stack(terms).mean() if terms else pred.sum() * 0.0


def masked_ssim_loss(pred, target, mask, window_size=3, eps=1e-6):
    """Small-window SSIM loss; invalid pixels are excluded from the final mean."""
    pad = window_size // 2
    mu_p = F.avg_pool2d(pred, window_size, 1, pad)
    mu_t = F.avg_pool2d(target, window_size, 1, pad)
    var_p = (F.avg_pool2d(pred * pred, window_size, 1, pad) - mu_p.square()).clamp_min(0.0)
    var_t = (F.avg_pool2d(target * target, window_size, 1, pad) - mu_t.square()).clamp_min(0.0)
    cov = F.avg_pool2d(pred * target, window_size, 1, pad) - mu_p * mu_t
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / (
        (mu_p.square() + mu_t.square() + c1) * (var_p + var_t + c2) + eps
    )
    valid = mask.bool()
    return (1.0 - ssim[valid]).mean() if valid.any() else pred.sum() * 0.0
