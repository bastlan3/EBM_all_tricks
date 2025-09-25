import torch
from typing import Dict, Tuple

from .langevin import LangevinSampler
from ..ebm import EBM

class DifferentiableLangevinSampler(LangevinSampler):
    """
    A version of the Langevin Sampler where the MCMC steps are part of the
    computation graph. This is a prerequisite for more advanced EBM objectives
    that require backpropagating through the sampling process.
    """

    def _step(self, samples: torch.Tensor, ebm_model: EBM) -> torch.Tensor:
        """
        Performs a single Langevin step, keeping the operation in the graph.

        The key difference from the standard Langevin step is `create_graph=True`.
        """
        samples.requires_grad_(True)
        energy = ebm_model(samples)

        # create_graph=True allows for gradients to be backpropagated through
        # the sampler itself, which is necessary for the corrected CD objective.
        grad, = torch.autograd.grad(energy.sum(), samples, create_graph=True)

        # We do not detach the samples here, allowing the gradient history to be preserved.
        new_samples = samples - self.step_size * grad + self.noise_scale * torch.randn_like(samples)

        return new_samples

    def sample(self, ebm_model: EBM, n_samples: int, sample_shape: Tuple, initial_states: torch.Tensor = None) -> torch.Tensor:
        """
        Generates samples using differentiable Langevin dynamics. The final output
        tensor remains attached to the computation graph.
        """
        device = ebm_model.device
        if initial_states is not None:
            samples = initial_states.to(device)
        else:
            samples = torch.rand(n_samples, *sample_shape, device=device) * 2 - 1

        for _ in range(self.k_steps):
            samples = self._step(samples, ebm_model)

        # Do NOT detach the final samples. The calling module will be responsible
        # for deciding when to detach.
        return samples

    @staticmethod
    def get_hyperparameter_info() -> dict:
        return {
            "k_steps": {
                "description": "Number of MCMC steps to run. Gradients will flow back through all steps.",
                "recommended": "5-20. Fewer steps are common due to computational cost.",
                "type": "int"
            },
            "step_size": {
                "description": "Step size for the Langevin dynamics update.",
                "recommended": "Varies greatly. Start with 1e-4 to 1e-5.",
                "type": "float"
            },
            "noise_scale": {
                "description": "Scale of the Gaussian noise added at each step.",
                "recommended": "Typically small, e.g., 0.005 to 0.01.",
                "type": "float"
            }
        }