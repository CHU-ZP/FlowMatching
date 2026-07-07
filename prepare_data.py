from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from torchvision import datasets, transforms

from data import CIFAR10_CLASSES


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.resolve().parent.parent / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify CIFAR-10 data.")
    default_config = Path(__file__).resolve().parent / "configs" / "cifar10_unet_fm.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    data_dir = args.data_dir or project_path(config_path, str(cfg.get("data_dir", "./datasets")))
    data_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    download = not args.no_download
    train_set = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=download,
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        transform=transform,
        download=download,
    )

    image, label = train_set[0]
    print(f"Data dir: {data_dir}")
    print(f"Train images: {len(train_set)}")
    print(f"Test images: {len(test_set)}")
    print(f"Sample shape: {tuple(image.shape)}")
    print(f"Sample value range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"Sample label: {label} ({CIFAR10_CLASSES[label]})")
    print("CIFAR-10 is ready.")


if __name__ == "__main__":
    main()
