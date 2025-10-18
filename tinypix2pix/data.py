"""Data utilities for the Tiny Pix2Pix project."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import TrainingConfig


class IdentityPairDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Dataset that pairs each image with itself.

    The original Keras implementation trained on CIFAR-10 images as both
    source and target to showcase the training loop. This dataset mimics that
    behaviour for demonstration purposes.
    """

    def __init__(self, root: Optional[str], train: bool, image_size: int, download: bool = True) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ]
        )
        dataset_root = root or "./data"
        self._dataset = datasets.CIFAR10(root=dataset_root, train=train, transform=transform, download=download)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, _ = self._dataset[index]
        return image, image


def build_dataloader(config: TrainingConfig) -> DataLoader[Tuple[torch.Tensor, torch.Tensor]]:
    """Build a dataloader for the Tiny Pix2Pix training pipeline."""

    dataset = IdentityPairDataset(
        root=str(config.dataset_root) if config.dataset_root else None,
        train=True,
        image_size=config.image_size,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
