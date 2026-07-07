from __future__ import annotations

import torch


def expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape [B] time values so they broadcast over image tensors."""
    while t.ndim < x.ndim:
        t = t[..., None]
    return t


def linear_interpolation(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_img = expand_time(t, x0)
    return (1.0 - t_img) * x0 + t_img * x1


def target_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    return x1 - x0


def sample_linear_path(
    x1: torch.Tensor,
    t: torch.Tensor | None = None,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = x1.shape[0]
    if x0 is None:
        x0 = torch.randn_like(x1)
    if t is None:
        t = torch.rand(batch_size, device=x1.device, dtype=x1.dtype)

    xt = linear_interpolation(x0, x1, t)
    velocity = target_velocity(x0, x1)
    return xt, velocity, t, x0
