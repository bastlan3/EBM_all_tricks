import torch
from typing import Dict, Tuple

from .base_sampler import Sampler
from ..ebm import EBM

class ParallelTemperingSampler(Sampler):
    """
    Implements Parallel Tempering sampling, also known as Replica Exchange MCMC.
    This method runs multiple MCMC chains in parallel at different "temperatures"
    and periodically proposes swaps between adjacent chains. Hotter chains can
    explore the energy landscape more broadly, and swapping allows these discoveries
    to be passed to the colder chains, improving mixing.
    """

    def __init__(self, config: Dict):
        """
        Initializes the ParallelTemperingSampler.

        Args:
            config (Dict): A dictionary containing configuration parameters.
                - n_chains (int): The number of parallel chains (temperatures).
                - betas (Tuple[float], optional): A tuple of inverse temperatures. If None,
                  a linear schedule from 1.0 to 0.1 is created.
                - swap_interval (int): Number of MCMC steps between swap proposals.
                - k_steps (int): Number of internal MCMC steps per sample() call.
                - step_size (float): Step size for the internal Langevin sampler.
                - noise_scale (float): Noise scale for the internal Langevin sampler.
        """
        super().__init__(config)
        self.n_chains = config.get('n_chains', 8)

        betas = config.get('betas', None)
        if betas is None:
            self.betas = torch.linspace(1.0, 0.1, self.n_chains)
        else:
            self.betas = torch.tensor(betas)
            self.n_chains = len(self.betas)

        self.swap_interval = config.get('swap_interval', 5)

        # Internal Langevin sampler configuration
        self.k_steps = config.get('k_steps', 20)
        self.step_size = config.get('step_size', 0.1)
        self.noise_scale = config.get('noise_scale', 0.01)

        # This will hold the state of all chains: (n_chains, n_samples, *sample_shape)
        self.chains_state = None

    def _langevin_step(self, samples: torch.Tensor, ebm_model: EBM, beta: float) -> torch.Tensor:
        """Performs a single Langevin step with a tempered energy function."""
        samples.requires_grad_(True)
        # The potential is the EBM's energy
        potential_energy = ebm_model(samples)
        # The gradient is taken with respect to the tempered energy
        grad, = torch.autograd.grad((beta * potential_energy).sum(), samples, create_graph=False)

        new_samples = samples.detach() - self.step_size * grad + self.noise_scale * torch.randn_like(samples)
        return new_samples.detach()

    def sample(self, ebm_model: EBM, n_samples: int, sample_shape: Tuple, initial_states: torch.Tensor = None) -> torch.Tensor:
        """
        Generates samples using Parallel Tempering. Note that the state of the chains
        is maintained internally between calls.

        Args:
            ebm_model (EBM): The Energy-Based Model to sample from.
            n_samples (int): The number of samples to generate for each chain.
            sample_shape (Tuple): The shape of a single sample.
            initial_states (torch.Tensor, optional): If provided, used to initialize the
                                                     coldest chain (beta=1). Not recommended for PT.

        Returns:
            torch.Tensor: Samples from the coldest chain (beta=1).
        """
        device = ebm_model.device
        self.betas = self.betas.to(device)

        if self.chains_state is None:
            # Initialize states for all chains if they don't exist
            self.chains_state = torch.rand(self.n_chains, n_samples, *sample_shape, device=device) * 2 - 1
            if initial_states is not None:
                # If initial states are provided, use them for the coldest chain
                self.chains_state[0] = initial_states.to(device)

        # Run internal MCMC steps
        for step in range(self.k_steps):
            # Propose swaps at regular intervals
            if step > 0 and step % self.swap_interval == 0:
                for i in range(self.n_chains - 1):
                    # Propose swap between adjacent chains i and i+1
                    beta_i, beta_j = self.betas[i], self.betas[i+1]
                    samples_i, samples_j = self.chains_state[i], self.chains_state[i+1]

                    energy_i = ebm_model(samples_i)
                    energy_j = ebm_model(samples_j)

                    # M-H acceptance probability for swapping
                    log_acceptance_ratio = (beta_j - beta_i) * (energy_i - energy_j)
                    acceptance_ratio = torch.exp(log_acceptance_ratio).clamp_(0, 1)

                    u = torch.rand_like(acceptance_ratio)
                    accept_mask = (u < acceptance_ratio)

                    while accept_mask.dim() < samples_i.dim():
                        accept_mask = accept_mask.unsqueeze(-1)

                    # Perform the swap where accepted
                    swapped_i = torch.where(accept_mask, samples_j, samples_i)
                    swapped_j = torch.where(accept_mask, samples_i, samples_j)
                    self.chains_state[i] = swapped_i
                    self.chains_state[i+1] = swapped_j

            # Apply one Langevin step to each chain
            new_chains_state = self.chains_state.clone()
            for i in range(self.n_chains):
                new_chains_state[i] = self._langevin_step(self.chains_state[i], ebm_model, self.betas[i])
            self.chains_state = new_chains_state

        # Return samples from the coldest chain (beta=1)
        return self.chains_state[0].detach()

    @staticmethod
    def get_hyperparameter_info() -> dict:
        return {
            "n_chains": {
                "description": "Number of parallel chains (temperatures) to run.",
                "recommended": "8-16",
                "type": "int"
            },
            "betas": {
                "description": "Tuple of inverse temperatures. If None, a linear schedule is used.",
                "recommended": "None (to use the default linear schedule) or a geometric schedule.",
                "type": "Tuple[float] or None"
            },
            "swap_interval": {
                "description": "Number of MCMC steps between swap proposals.",
                "recommended": "5-10",
                "type": "int"
            },
            "k_steps": {
                "description": "Number of internal Langevin steps per sample() call.",
                "recommended": "10-20",
                "type": "int"
            },
            "step_size": {
                "description": "Step size for the internal Langevin sampler.",
                "recommended": "1e-4 to 1e-5",
                "type": "float"
            },
            "noise_scale": {
                "description": "Noise scale for the internal Langevin sampler.",
                "recommended": "0.005",
                "type": "float"
            }
        }
