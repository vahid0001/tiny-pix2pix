"""Backward compatible training script for Tiny Pix2Pix."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from tinypix2pix import (
    PatchGANDiscriminator,
    Pix2PixTrainer,
    TrainingConfig,
    UNetGenerator,
    build_dataloader,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tiny Pix2Pix model (backward-compatible entry point).")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_root=args.dataset_root,
        device=args.device,
    )

    dataloader = build_dataloader(config)
    generator = UNetGenerator(in_channels=config.channels, out_channels=config.channels)
    discriminator = PatchGANDiscriminator(in_channels=config.channels)

    trainer = Pix2PixTrainer(generator, discriminator, config)

    for state in trainer.train(dataloader):
        print(
            f"Epoch {state.epoch}/{config.epochs} | Step {state.step} | "
            f"Gen Loss: {state.generator_loss:.4f} | Disc Loss: {state.discriminator_loss:.4f}"
        )


if __name__ == "__main__":
    main()
