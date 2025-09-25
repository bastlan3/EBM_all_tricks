import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Callable

from .langevin import LangevinSampler
from ..ebm import EBM

class AugmentedLangevinSampler(LangevinSampler):
    """
    A Langevin sampler that periodically applies a data augmentation transform
    as an additional MCMC transition operator. This can help the sampler
    jump between modes and improve mixing.
    """
    def __init__(self, config: Dict, transform: Optional[Callable] = None):
        """
        Initializes the AugmentedLangevinSampler.

        Args:
            config (Dict): Configuration dictionary for the parent LangevinSampler.
            transform (Optional[Callable]): A data augmentation transform to be applied.
                                             Should operate on PyTorch tensors.
        """
        super().__init__(config)
        self.transform = transform
        self.augment_interval = config.get('augment_interval', 5)

    def sample(self, ebm_model: EBM, n_samples: int, sample_shape: Tuple, initial_states: torch.Tensor = None) -> torch.Tensor:
        """
        Generates samples using Langevin Dynamics with periodic augmentation.

        Args:
            ebm_model (EBM): The Energy-Based Model to sample from.
            n_samples (int): The number of samples to generate.
            sample_shape (Tuple): The shape of a single sample.
            initial_states (torch.Tensor, optional): Starting points for the chains.

        Returns:
            torch.Tensor: The generated samples.
        """
        device = ebm_model.device

        if initial_states is not None:
            samples = initial_states.to(device)
        else:
            samples = torch.rand(n_samples, *sample_shape, device=device) * 2 - 1

        for step in range(self.k_steps):
            # Apply standard Langevin step
            samples = self._step(samples, ebm_model)

            # Apply augmentation transform periodically
            if self.transform and (step + 1) % self.augment_interval == 0:
                # Note: This is a simple implementation. A more advanced version
                # could include a Metropolis-Hastings acceptance step for the
                # augmentation, but for many common augmentations (like flips),
                # this is often omitted.
                samples = self.transform(samples)

        return samples.detach()

    def _step(self, samples: torch.Tensor, ebm_model: EBM) -> torch.Tensor:
        """
        Performs a single Langevin step. This is copied from the parent class
        to be used internally.
        """
        samples.requires_grad_(True)
        energy = ebm_model(samples)
        grad, = torch.autograd.grad(energy.sum(), samples, create_graph=False)

        new_samples = samples.detach() - self.step_size * grad + self.noise_scale * torch.randn_like(samples)
        return new_samples.detach()

    @staticmethod
    def get_hyperparameter_info() -> dict:
        info = super(AugmentedLangevinSampler, AugmentedLangevinSampler).get_hyperparameter_info()
        info.update({
            "augment_interval": {
                "description": "Number of MCMC steps between applying the augmentation transform.",
                "recommended": "5-10",
                "type": "int"
            }
        })
        return info