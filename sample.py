from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image
import yaml

from flow import sample_ode
from models import TinyUNet


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.resolve().parent.parent / path


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


def load_model_weights(model: torch.nn.Module, checkpoint: dict[str, Any], use_ema: bool) -> None:
    model.load_state_dict(checkpoint["model"])
    if use_ema and checkpoint.get("ema") is not None:
        state = model.state_dict()
        for key, value in checkpoint["ema"]["shadow"].items():
            state[key].copy_(value)


def make_class_labels(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor | None, int, int, str]:
    class_conditional = bool(cfg.get("class_conditional", False))
    if not class_conditional:
        if args.class_label is not None or args.class_grid:
            raise ValueError("The loaded config is unconditional, but class sampling was requested.")
        nrow = max(1, int(math.sqrt(args.num_samples)))
        return None, args.num_samples, nrow, "samples"

    num_classes = int(cfg.get("num_classes", 10))
    if args.class_grid:
        samples_per_class = max(1, args.samples_per_class)
        labels = torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)
        return labels, labels.shape[0], samples_per_class, "class_grid"

    if args.class_label is not None:
        if args.class_label < 0 or args.class_label >= num_classes:
            raise ValueError(f"class-label must be in [0, {num_classes - 1}]")
        labels = torch.full((args.num_samples,), args.class_label, device=device, dtype=torch.long)
        nrow = max(1, int(math.sqrt(args.num_samples)))
        return labels, args.num_samples, nrow, f"class_{args.class_label}"

    labels = torch.arange(args.num_samples, device=device, dtype=torch.long) % num_classes
    nrow = max(1, int(math.sqrt(args.num_samples)))
    return labels, args.num_samples, nrow, "class_cycle"


def tensor_to_pil_grid(x: torch.Tensor, nrow: int) -> Image.Image:
    grid = make_grid(to_image_range(x), nrow=nrow)
    array = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(array)


def save_trajectory_gif(frames: list[torch.Tensor], path: Path, nrow: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [tensor_to_pil_grid(frame, nrow=nrow) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=80,
        loop=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a CIFAR-10 Flow Matching checkpoint.")
    default_config = Path(__file__).resolve().parent / "configs" / "cifar10_unet_fm.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--method", type=str, choices=("euler", "heun"), default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--class-grid", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--trajectory-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    out_dir = project_path(config_path, str(cfg.get("out_dir", "./runs/cifar10_fm")))
    checkpoint_path = args.ckpt or (out_dir / "checkpoints" / "latest.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint.get("config", cfg)

    model = build_model(model_cfg).to(device)
    load_model_weights(model, checkpoint, use_ema=not args.no_ema)
    model.eval()

    steps = args.steps or int(model_cfg.get("num_steps_sampling", 50))
    method = args.method or str(model_cfg.get("sampling_method", "euler"))
    image_size = int(model_cfg.get("image_size", 32))
    in_channels = int(model_cfg.get("in_channels", 3))
    labels, num_samples, nrow, output_tag = make_class_labels(args, model_cfg, device)

    samples, frames = sample_ode(
        model=model,
        shape=(num_samples, in_channels, image_size, image_size),
        steps=steps,
        device=device,
        method=method,
        y=labels,
        return_intermediates=args.save_trajectory,
        trajectory_every=args.trajectory_every,
    )

    output_path = args.out or (out_dir / "samples" / f"{output_tag}_{method}_{steps:03d}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(to_image_range(samples), output_path, nrow=nrow)

    if args.save_trajectory and frames is not None:
        gif_path = output_path.with_suffix(".gif")
        save_trajectory_gif(frames, gif_path, nrow=nrow)

    print(f"Saved samples to {output_path}")


if __name__ == "__main__":
    main()
