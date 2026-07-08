from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from data import CIFAR10_CLASSES, get_cifar10_dataset


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.resolve().parent.parent / path


def optional_project_path(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    return project_path(config_path, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify CIFAR-10 data.")
    default_config = Path(__file__).resolve().parent / "configs" / "cifar10_unet_fm.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--data-source", choices=("huggingface", "torchvision"), default=None)
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--hf-dataset-id", type=str, default=None)
    parser.add_argument("--hf-config-name", type=str, default=None)
    parser.add_argument("--hf-endpoint", type=str, default=None)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    data_dir = args.data_dir or project_path(config_path, str(cfg.get("data_dir", "./datasets")))
    data_source = args.data_source or str(cfg.get("data_source", "huggingface"))
    hf_cache_dir = args.hf_cache_dir or optional_project_path(config_path, cfg.get("hf_cache_dir"))
    hf_dataset_id = args.hf_dataset_id or str(cfg.get("hf_dataset_id", "uoft-cs/cifar10"))
    hf_config_name = args.hf_config_name or cfg.get("hf_config_name")
    if data_source == "huggingface" and hf_cache_dir is None:
        hf_cache_dir = data_dir / "huggingface"
    if args.hf_endpoint is not None:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    data_dir.mkdir(parents=True, exist_ok=True)
    if hf_cache_dir is not None:
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
    download = not args.no_download
    train_set = get_cifar10_dataset(
        data_dir=str(data_dir),
        train=True,
        download=download,
        data_source=data_source,
        hf_cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
        hf_dataset_id=hf_dataset_id,
        hf_config_name=hf_config_name,
    )
    test_set = get_cifar10_dataset(
        data_dir=str(data_dir),
        train=False,
        download=download,
        data_source=data_source,
        hf_cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
        hf_dataset_id=hf_dataset_id,
        hf_config_name=hf_config_name,
    )

    image, label = train_set[0]
    print(f"Data source: {data_source}")
    if data_source == "huggingface":
        print(f"Hugging Face dataset id: {hf_dataset_id}")
    print(f"Data dir: {data_dir}")
    if hf_cache_dir is not None:
        print(f"Hugging Face cache dir: {hf_cache_dir}")
    print(f"Train images: {len(train_set)}")
    print(f"Test images: {len(test_set)}")
    print(f"Sample shape: {tuple(image.shape)}")
    print(f"Sample value range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"Sample label: {label} ({CIFAR10_CLASSES[label]})")
    print("CIFAR-10 is ready.")


if __name__ == "__main__":
    main()
