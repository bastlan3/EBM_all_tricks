import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
from copy import deepcopy

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
                 num_scales: int = 1,
                 target_ebm_model: Optional[EBM] = None,
                 ema_decay: Optional[float] = 0.999):
        """
        Args:
            ebm_model (EBM): The "online" EBM, which is updated by the optimizer.
            sampler (Sampler): The MCMC sampler for generating negative samples.
            optimizer_config (Dict[str, Any]): Configuration for the optimizer.
            regularizers (List[Regularizer], optional): A list of regularizers.
            num_scales (int): Number of scales for multi-scale energy calculation.
            target_ebm_model (Optional[EBM]): An optional "target" EBM. If provided, EMA updates
                                              will be used for stabilization.
            ema_decay (float): The decay rate for the EMA of the target model.
        """
        super().__init__()
        self.ebm_model = ebm_model
        self.target_ebm_model = target_ebm_model
        self.sampler = sampler
        self.optimizer_config = optimizer_config
        self.regularizers = nn.ModuleList(regularizers if regularizers is not None else [])
        self.num_scales = num_scales
        self.ema_decay = ema_decay

        if self.target_ebm_model is not None:
            # Ensure target model parameters are not updated by the optimizer
            self.target_ebm_model.requires_grad_(False)

        # This is important for PyTorch Lightning to track your model's parameters
        # and other hyperparameters, making saving and loading checkpoints robust.
        self.save_hyperparameters(ignore=['ebm_model', 'sampler', 'regularizers', 'target_ebm_model'])

    def _calculate_multi_scale_energy(self, samples: torch.Tensor, model: EBM) -> torch.Tensor:
        """
        Calculates the total energy of a batch of samples by summing the
        energies over multiple resolutions of the samples, using the provided model.
        """
        total_energy = torch.zeros(samples.shape[0], device=samples.device)

        for scale in range(self.num_scales):
            if scale > 0:
                samples = F.avg_pool2d(samples, kernel_size=2)
            total_energy += model(samples).squeeze()

        return total_energy

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Hook to update the target model using EMA after each training batch, if a target model is provided.
        """
        if self.target_ebm_model is not None:
            online_params = self.ebm_model.parameters()
            target_params = self.target_ebm_model.parameters()

            for p_online, p_target in zip(online_params, target_params):
                p_target.data.copy_(self.ema_decay * p_target.data + (1.0 - self.ema_decay) * p_online.data)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.
        """
        # 1. Unpack positive samples from the batch
        if isinstance(batch, (list, tuple)):
            positive_samples = batch[0]
        else:
            positive_samples = batch

        # Determine which model to use for sampling and negative energy calculation
        model_for_sampling = self.target_ebm_model if self.target_ebm_model is not None else self.ebm_model
        if self.target_ebm_model is not None:
            self.target_ebm_model.to(self.device)

        # 2. Generate negative samples
        progress = self.trainer.global_step / self.trainer.max_steps if self.trainer.max_steps else 0.0
        negative_samples = self.sampler.sample(
            model_for_sampling,
            n_samples=positive_samples.shape[0],
            sample_shape=positive_samples.shape[1:],
            progress=progress
        )

        # 3. Calculate the core Contrastive Divergence (CD) loss
        positive_energy = self._calculate_multi_scale_energy(positive_samples, self.ebm_model)
        negative_energy = self._calculate_multi_scale_energy(negative_samples, model_for_sampling)

        cd_loss = positive_energy.mean() - negative_energy.mean()
        self.log('train_loss/cd_loss', cd_loss, on_step=True, on_epoch=True, prog_bar=True)

        total_loss = cd_loss

        # 4. Calculate and add regularization losses
        # Regularizers should operate on the online model to affect the gradient update
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
