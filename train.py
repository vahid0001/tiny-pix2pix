"""Command-line interface for training the Tiny Pix2Pix model."""

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


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Tiny Pix2Pix model using PyTorch.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to the dataset root directory.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Adam learning rate.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 parameter.")
    parser.add_argument("--lambda-l1", type=float, default=100.0, help="Weight for the L1 reconstruction loss.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory to store model checkpoints.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use for training.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loading workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--log-interval", type=int, default=50, help="Number of steps between logging updates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    config = TrainingConfig(
        image_size=32,
        channels=3,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        lambda_l1=args.lambda_l1,
        dataset_root=args.dataset_root,
        num_workers=args.num_workers,
        device=args.device,
        model_dir=args.model_dir,
    )

    dataloader = build_dataloader(config)
    generator = UNetGenerator(in_channels=config.channels, out_channels=config.channels)
    discriminator = PatchGANDiscriminator(in_channels=config.channels)
    trainer = Pix2PixTrainer(generator, discriminator, config)

    for state in trainer.train(dataloader, log_interval=args.log_interval):
        print(
            f"Epoch {state.epoch}/{config.epochs} | Step {state.step} | "
            f"Gen Loss: {state.generator_loss:.4f} | Disc Loss: {state.discriminator_loss:.4f}"
        )


if __name__ == "__main__":
    main()
