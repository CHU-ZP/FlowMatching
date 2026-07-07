from __future__ import annotations

import torch
import torch.nn.functional as F

from .paths import sample_linear_path


def flow_matching_loss(
    model: torch.nn.Module,
    x1: torch.Tensor,
    y: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    x0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xt, target_v, t, x0 = sample_linear_path(x1=x1, t=t, x0=x0)
    pred_v = model(xt, t, y) if y is not None else model(xt, t)
    loss = F.mse_loss(pred_v, target_v)
    info = {
        "pred_v": pred_v.detach(),
        "target_v": target_v.detach(),
        "xt": xt.detach(),
        "t": t.detach(),
        "x0": x0.detach(),
    }
    if y is not None:
        info["y"] = y.detach()
    return loss, info
