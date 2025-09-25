import torch
from typing import Dict, Tuple

from .base_sampler import Sampler
from ..ebm import EBM

class HMCSampler(Sampler):
    """
    Implements Hamiltonian Monte Carlo (HMC) sampling. HMC uses an auxiliary
    momentum variable and simulates Hamiltonian dynamics to generate distant
    proposal states that still have a high probability of acceptance.
    """

    def __init__(self, config: Dict):
        """
        Initializes the HMCSampler.

        Args:
            config (Dict): A dictionary containing configuration parameters.
                - n_leapfrog_steps (int): The number of leapfrog steps to simulate.
                - step_size (float): The step size (epsilon) for the leapfrog integrator.
        """
        super().__init__(config)
        self.n_leapfrog_steps = config.get('n_leapfrog_steps', 20)
        self.step_size = config.get('step_size', 0.1)

    def _compute_potential_energy(self, ebm_model: EBM, samples: torch.Tensor) -> torch.Tensor:
        """Computes the potential energy U(q), which is the EBM's energy."""
        return ebm_model(samples)

    def _compute_kinetic_energy(self, momentum: torch.Tensor) -> torch.Tensor:
        """Computes the kinetic energy K(p) = 0.5 * p^T @ p."""
        # Sum over all dimensions except the batch dimension
        return 0.5 * torch.sum(momentum * momentum, dim=list(range(1, momentum.dim())))

    def _get_grad(self, ebm_model: EBM, samples: torch.Tensor) -> torch.Tensor:
        """Computes the gradient of the potential energy w.r.t. the samples."""
        samples.requires_grad_(True)
        energy = self._compute_potential_energy(ebm_model, samples)
        grad, = torch.autograd.grad(energy.sum(), samples, create_graph=False)
        samples.requires_grad_(False)
        return grad.detach()

    def sample(self, ebm_model: EBM, n_samples: int, sample_shape: Tuple, initial_states: torch.Tensor = None) -> torch.Tensor:
        """
        Generates samples from the EBM using a single HMC update.

        Args:
            ebm_model (EBM): The Energy-Based Model to sample from.
            n_samples (int): The number of samples to generate.
            sample_shape (Tuple): The shape of a single sample.
            initial_states (torch.Tensor, optional): Starting points for the MCMC chains.

        Returns:
            torch.Tensor: The generated samples.
        """
        device = ebm_model.device
        if initial_states is not None:
            q = initial_states.to(device)
        else:
            q = torch.rand(n_samples, *sample_shape, device=device) * 2 - 1

        # Store the current position for the MH acceptance step
        current_q = q.clone()

        # 1. Sample initial momentum from a standard normal distribution
        p = torch.randn_like(q, device=device)
        current_p = p.clone()

        # 2. Leapfrog integration to simulate Hamiltonian dynamics
        # First half-step for momentum
        grad = self._get_grad(ebm_model, q)
        p -= 0.5 * self.step_size * grad

        # Full steps for position and momentum
        for _ in range(self.n_leapfrog_steps - 1):
            q += self.step_size * p
            grad = self._get_grad(ebm_model, q)
            p -= self.step_size * grad

        # Final full step for position
        q += self.step_size * p
        # Final half-step for momentum
        grad = self._get_grad(ebm_model, q)
        p -= 0.5 * self.step_size * grad

        # Negate momentum to make the proposal symmetric (standard practice)
        p = -p

        # 3. Metropolis-Hastings acceptance step
        current_U = self._compute_potential_energy(ebm_model, current_q)
        current_K = self._compute_kinetic_energy(current_p)

        proposal_U = self._compute_potential_energy(ebm_model, q)
        proposal_K = self._compute_kinetic_energy(p)

        # Calculate acceptance probability: min(1, exp(current_H - proposal_H))
        log_acceptance_ratio = (current_U + current_K) - (proposal_U + proposal_K)
        acceptance_ratio = torch.exp(log_acceptance_ratio).clamp_(0, 1)

        u = torch.rand_like(acceptance_ratio, device=device)
        accept_mask = (u < acceptance_ratio)

        # Unsqueeze mask to match sample dimensions for torch.where
        while accept_mask.dim() < q.dim():
            accept_mask = accept_mask.unsqueeze(-1)

        final_samples = torch.where(accept_mask, q, current_q)

        return final_samples.detach()

    @staticmethod
    def get_hyperparameter_info() -> dict:
        return {
            "n_leapfrog_steps": {
                "description": "Number of leapfrog steps to simulate Hamiltonian dynamics.",
                "recommended": "10-20. More steps allow for more distant proposals.",
                "type": "int"
            },
            "step_size": {
                "description": "Step size (epsilon) for the leapfrog integrator.",
                "recommended": "Highly sensitive. Requires tuning. Start small, e.g., 0.05-0.1.",
                "type": "float"
            }
        }
