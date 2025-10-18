"""Configuration dataclasses for Tiny Pix2Pix training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _default_model_dir() -> Path:
    return Path("models")


@dataclass
class TrainingConfig:
    """Configuration parameters for training the Pix2Pix model."""

    image_size: int = 32
    channels: int = 3
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 2e-4
    beta1: float = 0.5
    lambda_l1: float = 100.0
    dataset_root: Optional[Path] = None
    num_workers: int = 2
    device: str = "cuda"  # Trainer will fall back to CPU if unavailable.
    model_dir: Path = field(default_factory=_default_model_dir)

    def ensure_model_dir(self) -> None:
        """Ensure the model directory exists."""

        self.model_dir.mkdir(parents=True, exist_ok=True)
