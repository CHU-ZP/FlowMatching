from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(time_emb))[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 2),
        num_res_blocks: int = 2,
        time_embedding_dim: int = 256,
        dropout: float = 0.0,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.num_classes = num_classes
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.class_emb = (
            nn.Embedding(num_classes, time_embedding_dim) if num_classes is not None else None
        )

        channels = [base_channels * mult for mult in channel_mult]
        ch = base_channels
        skip_channels: list[int] = []
        self.downs = nn.ModuleList()
        for level, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, out_ch, time_embedding_dim, dropout))
                ch = out_ch
                skip_channels.append(ch)
            downsample = Downsample(ch) if level != len(channels) - 1 else nn.Identity()
            self.downs.append(nn.ModuleDict({"blocks": blocks, "downsample": downsample}))

        self.mid1 = ResBlock(ch, ch, time_embedding_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_embedding_dim, dropout)

        self.ups = nn.ModuleList()
        for level, out_ch in reversed(list(enumerate(channels))):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_channels.pop()
                blocks.append(ResBlock(ch + skip_ch, out_ch, time_embedding_dim, dropout))
                ch = out_ch
            upsample = Upsample(ch) if level != 0 else nn.Identity()
            self.ups.append(nn.ModuleDict({"blocks": blocks, "upsample": upsample}))

        self.out_norm = nn.GroupNorm(_group_count(ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t.ndim != 1:
            t = t.view(t.shape[0])

        time_emb = sinusoidal_time_embedding(t, self.time_embedding_dim)
        time_emb = self.time_mlp(time_emb)
        if self.class_emb is not None:
            if y is None:
                raise ValueError("class labels are required when num_classes is set")
            if y.ndim != 1:
                y = y.view(y.shape[0])
            time_emb = time_emb + self.class_emb(y.long())

        h = self.init_conv(x)
        skips: list[torch.Tensor] = []
        for down in self.downs:
            for block in down["blocks"]:
                h = block(h, time_emb)
                skips.append(h)
            h = down["downsample"](h)

        h = self.mid1(h, time_emb)
        h = self.mid2(h, time_emb)

        for up in self.ups:
            for block in up["blocks"]:
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb)
            h = up["upsample"](h)

        return self.out_conv(self.act(self.out_norm(h)))
