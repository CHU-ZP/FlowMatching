from __future__ import annotations

import torch


@torch.no_grad()
def sample_ode(
    model: torch.nn.Module,
    shape: tuple[int, int, int, int],
    steps: int,
    device: torch.device | str,
    method: str = "euler",
    x0: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    return_intermediates: bool = False,
    trajectory_every: int = 1,
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if method not in {"euler", "heun"}:
        raise ValueError(f"Unknown sampler method: {method}")

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    device = torch.device(device)
    x = torch.randn(shape, device=device) if x0 is None else x0.to(device)
    batch_size = x.shape[0]
    if y is not None:
        y = y.to(device=device, dtype=torch.long)
        if y.ndim != 1:
            y = y.view(y.shape[0])
        if y.shape[0] != batch_size:
            raise ValueError(f"Expected {batch_size} labels, got {y.shape[0]}")
    dt = 1.0 / steps
    frames: list[torch.Tensor] | None = [] if return_intermediates else None
    trajectory_every = max(1, trajectory_every)

    if frames is not None:
        frames.append(x.detach().cpu())

    for i in range(steps):
        t_value = i / steps
        t = torch.full((batch_size,), t_value, device=device, dtype=x.dtype)
        v = model(x, t, y) if y is not None else model(x, t)

        if method == "euler":
            x = x + dt * v
        else:
            t_next = torch.full((batch_size,), (i + 1) / steps, device=device, dtype=x.dtype)
            x_euler = x + dt * v
            v_next = model(x_euler, t_next, y) if y is not None else model(x_euler, t_next)
            x = x + 0.5 * dt * (v + v_next)

        if frames is not None and ((i + 1) % trajectory_every == 0 or i == steps - 1):
            frames.append(x.detach().cpu())

    if was_training and hasattr(model, "train"):
        model.train()

    return x, frames
