import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

from .ebm import EBM
from .samplers.base_sampler import Sampler
from .regularizers.base_regularizer import Regularizer

class EBMLightningModule(pl.LightningModule):
    """
    The main PyTorch Lightning module for training Energy-Based Models.
    This module orchestrates the EBM, sampler, and regularizers.
    """
    def __init__(self,
                 ebm_model: EBM,
                 sampler: Sampler,
                 optimizer_config: Dict[str, Any],
                 regularizers: List[Regularizer] = None,
                 num_scales: int = 1):
        """
        Args:
            ebm_model (EBM): The energy-based model.
            sampler (Sampler): The MCMC sampler for generating negative samples.
            optimizer_config (Dict[str, Any]): Configuration for the optimizer (e.g., {'name': 'Adam', 'lr': 1e-4}).
            regularizers (List[Regularizer], optional): A list of regularizers to apply. Defaults to None.
        """
        super().__init__()
        self.ebm_model = ebm_model
        self.sampler = sampler
        self.optimizer_config = optimizer_config
        self.regularizers = nn.ModuleList(regularizers if regularizers is not None else [])
        self.num_scales = num_scales

        # This is important for PyTorch Lightning to track your model's parameters
        # and other hyperparameters, making saving and loading checkpoints robust.
        self.save_hyperparameters(ignore=['ebm_model', 'sampler', 'regularizers'])

    def _calculate_multi_scale_energy(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total energy of a batch of samples by summing the
        energies over multiple resolutions of the samples.
        """
        total_energy = torch.zeros(samples.shape[0], device=samples.device)

        for scale in range(self.num_scales):
            if scale > 0:
                # Downsample by a factor of 2 for each new scale
                samples = F.avg_pool2d(samples, kernel_size=2)

            # The EBM network must be able to handle variable input sizes
            # if num_scales > 1. This is typically achieved with adaptive pooling.
            total_energy += self.ebm_model(samples).squeeze()

        return total_energy

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.

        Args:
            batch (Any): The output from the DataLoader. Assumed to be a tensor of positive samples,
                         or a list/tuple where the first element is the tensor of positive samples.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The total loss for this training step.
        """
        # 1. Unpack positive samples from the batch
        if isinstance(batch, (list, tuple)):
            positive_samples = batch[0]
        else:
            positive_samples = batch

        # Positive samples are automatically moved to the correct device by Lightning.

        # 2. Generate negative samples using the sampler
        # The sampler's `sample` method is responsible for creating tensors on the correct device.
        negative_samples = self.sampler.sample(
            self.ebm_model,
            n_samples=positive_samples.shape[0],
            sample_shape=positive_samples.shape[1:]  # Pass the shape of an individual sample
        )

        # 3. Calculate the core Contrastive Divergence (CD) loss
        positive_energy = self._calculate_multi_scale_energy(positive_samples)
        negative_energy = self._calculate_multi_scale_energy(negative_samples)

        cd_loss = positive_energy.mean() - negative_energy.mean()
        self.log('train_loss/cd_loss', cd_loss, on_step=True, on_epoch=True, prog_bar=True)

        total_loss = cd_loss

        # 4. Calculate and add regularization losses
        for reg in self.regularizers:
            reg_loss = reg.calculate_loss(self.ebm_model, positive_samples, negative_samples)
            # Use a descriptive name for logging, e.g., "L2EnergyRegularizer_loss"
            reg_name = reg.__class__.__name__
            self.log(f'train_loss/{reg_name}', reg_loss, on_step=True, on_epoch=True)
            total_loss = total_loss + reg_loss

        self.log('train_loss/total_loss', total_loss, on_step=True, on_epoch=True)

        # 5. Return the total loss
        return total_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Sets up the optimizer based on the configuration dictionary.

        Raises:
            ValueError: If the specified optimizer is not found in `torch.optim`.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer_name = self.optimizer_config.get('name', 'Adam')
        lr = self.optimizer_config.get('lr', 1e-4)

        optimizer_class = getattr(torch.optim, optimizer_name, None)
        if optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")

        # self.parameters() correctly includes parameters from self.ebm_model
        optimizer = optimizer_class(self.parameters(), lr=lr)

        return optimizer
