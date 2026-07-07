from __future__ import annotations

import argparse
import contextlib
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image
from tqdm import tqdm
import yaml

from data import get_cifar10_dataloader
from flow import flow_matching_loss, sample_ode
from models import TinyUNet


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in self.shadow.items():
            value.mul_(self.decay).add_(state[key].detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in self.shadow.items():
            state[key].copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {key: value.clone() for key, value in state["shadow"].items()}


@contextlib.contextmanager
def ema_weights(model: torch.nn.Module, ema: EMA | None):
    if ema is None:
        yield
        return
    backup = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if value.dtype.is_floating_point
    }
    ema.copy_to(model)
    try:
        yield
    finally:
        state = model.state_dict()
        for key, value in backup.items():
            state[key].copy_(value)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.resolve().parent.parent / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: dict[str, Any]) -> TinyUNet:
    num_classes = int(cfg.get("num_classes", 10)) if bool(cfg.get("class_conditional", False)) else None
    return TinyUNet(
        in_channels=int(cfg.get("in_channels", 3)),
        out_channels=int(cfg.get("in_channels", 3)),
        base_channels=int(cfg.get("base_channels", 64)),
        channel_mult=tuple(cfg.get("channel_mult", [1, 2, 2])),
        num_res_blocks=int(cfg.get("num_res_blocks", 2)),
        time_embedding_dim=int(cfg.get("time_embedding_dim", 256)),
        dropout=float(cfg.get("dropout", 0.0)),
        num_classes=num_classes,
    )


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def preview_labels(num_images: int, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(num_classes, device=device)
    repeats = (num_images + num_classes - 1) // num_classes
    return labels.repeat(repeats)[:num_images]


@torch.no_grad()
def save_samples(
    model: torch.nn.Module,
    ema: EMA | None,
    cfg: dict[str, Any],
    device: torch.device,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_images = int(cfg.get("num_sample_images", 64))
    steps = int(cfg.get("num_steps_sampling", 50))
    method = str(cfg.get("sampling_method", "euler"))
    image_size = int(cfg.get("image_size", 32))
    in_channels = int(cfg.get("in_channels", 3))
    labels = None
    if bool(cfg.get("class_conditional", False)):
        labels = preview_labels(num_images, int(cfg.get("num_classes", 10)), device)
    with ema_weights(model, ema):
        samples, _ = sample_ode(
            model=model,
            shape=(num_images, in_channels, image_size, image_size),
            steps=steps,
            device=device,
            method=method,
            y=labels,
        )
    save_image(to_image_range(samples), output_path, nrow=8)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA | None,
    cfg: dict[str, Any],
    epoch: int,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "config": cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 pixel-space Flow Matching.")
    default_config = Path(__file__).resolve().parent / "configs" / "cifar10_unet_fm.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = project_path(config_path, str(cfg.get("out_dir", "./runs/cifar10_fm")))
    data_dir = project_path(config_path, str(cfg.get("data_dir", "./datasets")))
    checkpoint_dir = out_dir / "checkpoints"
    sample_dir = out_dir / "samples"

    loader = get_cifar10_dataloader(
        data_dir=str(data_dir),
        batch_size=int(cfg.get("batch_size", 128)),
        num_workers=int(cfg.get("num_workers", 4)),
    )
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )
    ema = EMA(model, float(cfg.get("ema_decay", 0.999))) if bool(cfg.get("ema", True)) else None

    start_epoch = 1
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ema is not None and checkpoint.get("ema") is not None:
            ema.load_state_dict(checkpoint["ema"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))

    epochs = int(cfg.get("epochs", 100))
    log_every = int(cfg.get("log_every_steps", 50))
    save_every = int(cfg.get("save_every_epochs", 10))
    sample_every = int(cfg.get("sample_every_epochs", 10))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.0))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        running_loss = 0.0
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) if bool(cfg.get("class_conditional", False)) else None
            loss, _ = flow_matching_loss(model, images, y=labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            global_step += 1
            running_loss += loss.item()
            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                progress.set_postfix(loss=f"{avg_loss:.4f}")
                running_loss = 0.0

        save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            ema=ema,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
        )
        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                ema=ema,
                cfg=cfg,
                epoch=epoch,
                global_step=global_step,
            )
        if sample_every > 0 and epoch % sample_every == 0:
            save_samples(model, ema, cfg, device, sample_dir / f"epoch_{epoch:04d}.png")


if __name__ == "__main__":
    main()
