"""Training utilities for the Tiny Pix2Pix project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .models import PatchGANDiscriminator, UNetGenerator


@dataclass
class TrainingState:
    epoch: int
    step: int
    generator_loss: float
    discriminator_loss: float


class Pix2PixTrainer:
    """Encapsulates the Pix2Pix training loop."""

    def __init__(
        self,
        generator: UNetGenerator,
        discriminator: PatchGANDiscriminator,
        config: TrainingConfig,
    ) -> None:
        self.config = config
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

        self.optim_generator = Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, 0.999),
        )
        self.optim_discriminator = Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, 0.999),
        )

        self._real_label_cache: torch.Tensor | None = None
        self._fake_label_cache: torch.Tensor | None = None

    def _label_tensor(self, reference: torch.Tensor, value: float) -> torch.Tensor:
        """Return a cached label tensor matching the shape of ``reference``."""

        cache = self._real_label_cache if value == 1.0 else self._fake_label_cache
        if cache is None or cache.shape != reference.shape:
            cache = torch.full_like(reference, fill_value=value, device=self.device)
            if value == 1.0:
                self._real_label_cache = cache
            else:
                self._fake_label_cache = cache
        return cache

    def _save_checkpoint(self, epoch: int) -> None:
        """Persist model weights to disk."""

        self.config.ensure_model_dir()
        torch.save(self.generator.state_dict(), self.config.model_dir / f"generator_epoch_{epoch}.pt")
        torch.save(self.discriminator.state_dict(), self.config.model_dir / f"discriminator_epoch_{epoch}.pt")

    def train(self, dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]], log_interval: int = 50) -> Iterable[TrainingState]:
        """Train the Pix2Pix model, yielding progress updates."""

        step = 0
        last_generator_loss = 0.0
        last_discriminator_loss = 0.0
        for epoch in range(1, self.config.epochs + 1):
            for batch_idx, (source, target) in enumerate(dataloader, start=1):
                source = source.to(self.device)
                target = target.to(self.device)

                fake = self.generator(source)

                # --- Train discriminator ---
                self.optim_discriminator.zero_grad()
                pred_real = self.discriminator(source, target)
                pred_fake = self.discriminator(source, fake.detach())

                loss_real = self.criterion_gan(pred_real, self._label_tensor(pred_real, 1.0))
                loss_fake = self.criterion_gan(pred_fake, self._label_tensor(pred_fake, 0.0))
                loss_discriminator = 0.5 * (loss_real + loss_fake)
                loss_discriminator.backward()
                self.optim_discriminator.step()

                # --- Train generator ---
                self.optim_generator.zero_grad()
                pred_fake = self.discriminator(source, fake)
                loss_gan = self.criterion_gan(pred_fake, self._label_tensor(pred_fake, 1.0))
                loss_l1 = self.criterion_l1(fake, target)
                loss_generator = loss_gan + self.config.lambda_l1 * loss_l1
                loss_generator.backward()
                self.optim_generator.step()

                step += 1
                last_generator_loss = loss_generator.item()
                last_discriminator_loss = loss_discriminator.item()
                if step % log_interval == 0:
                    yield TrainingState(
                        epoch=epoch,
                        step=step,
                        generator_loss=last_generator_loss,
                        discriminator_loss=last_discriminator_loss,
                    )

            self._save_checkpoint(epoch)

        # Final state at the end of training
        yield TrainingState(
            epoch=self.config.epochs,
            step=step,
            generator_loss=last_generator_loss,
            discriminator_loss=last_discriminator_loss,
        )
