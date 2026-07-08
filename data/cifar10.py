from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision import transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def cifar10_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class HuggingFaceCIFAR10(Dataset):
    def __init__(
        self,
        split: str,
        cache_dir: str,
        transform: transforms.Compose,
        dataset_id: str = "uoft-cs/cifar10",
        config_name: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        from datasets import DownloadConfig, load_dataset

        download_config = DownloadConfig(local_files_only=local_files_only)
        self.dataset = load_dataset(
            dataset_id,
            name=config_name,
            split=split,
            cache_dir=cache_dir,
            download_config=download_config,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self.dataset[index]
        image = item.get("img", item.get("image"))
        if image is None:
            raise KeyError("Expected Hugging Face CIFAR-10 image field `img` or `image`.")
        if getattr(image, "mode", None) != "RGB":
            image = image.convert("RGB")
        label = int(item["label"])
        return self.transform(image), label


def get_cifar10_dataset(
    data_dir: str,
    train: bool = True,
    download: bool = True,
    data_source: str = "huggingface",
    hf_cache_dir: str | None = None,
    hf_dataset_id: str = "uoft-cs/cifar10",
    hf_config_name: str | None = None,
) -> Dataset:
    transform = cifar10_transform()
    data_source = data_source.lower()
    if data_source == "torchvision":
        return tv_datasets.CIFAR10(
            root=data_dir,
            train=train,
            transform=transform,
            download=download,
        )
    if data_source == "huggingface":
        split = "train" if train else "test"
        return HuggingFaceCIFAR10(
            split=split,
            cache_dir=hf_cache_dir or data_dir,
            transform=transform,
            dataset_id=hf_dataset_id,
            config_name=hf_config_name,
            local_files_only=not download,
        )
    raise ValueError(f"Unknown CIFAR-10 data_source: {data_source}")


def get_cifar10_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    train: bool = True,
    download: bool = True,
    drop_last: bool = True,
    data_source: str = "huggingface",
    hf_cache_dir: str | None = None,
    hf_dataset_id: str = "uoft-cs/cifar10",
    hf_config_name: str | None = None,
) -> DataLoader:
    dataset = get_cifar10_dataset(
        data_dir=data_dir,
        train=train,
        download=download,
        data_source=data_source,
        hf_cache_dir=hf_cache_dir,
        hf_dataset_id=hf_dataset_id,
        hf_config_name=hf_config_name,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
