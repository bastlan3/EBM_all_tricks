from abc import ABC, abstractmethod
import torch
from ..ebm import EBM

class Sampler(ABC):
    """
    Abstract base class for all MCMC samplers.
    """
    def __init__(self, config: dict):
        """
        Args:
            config (dict): A dictionary of configuration parameters for the sampler.
        """
        self.config = config

    @abstractmethod
    def sample(self, ebm_model: EBM, n_samples: int, sample_shape: tuple, initial_states: torch.Tensor = None) -> torch.Tensor:
        """
        The core method for generating samples from the EBM's distribution.

        Args:
            ebm_model (EBM): The EBM model from which to sample. The sampler will use ebm_model.device.
            n_samples (int): The number of samples to generate.
            sample_shape (tuple): The shape of a single sample (e.g., (C, H, W)).
            initial_states (torch.Tensor, optional): The starting points for the MCMC chains.
                                                     If None, they should be initialized randomly.
                                                     Defaults to None.

        Returns:
            torch.Tensor: A tensor of generated samples.
        """
        pass
