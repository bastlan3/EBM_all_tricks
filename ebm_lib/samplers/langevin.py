import torch
from typing import Dict, Tuple

from .base_sampler import Sampler
from ..ebm import EBM


class LangevinSampler(Sampler):
    """
    Implements Langevin Dynamics sampling, a method for MCMC sampling that uses
    gradient information from the EBM.
    """

    def __init__(self, config: Dict):
        """
        Initializes the LangevinSampler.

        Args:
            config (Dict): A dictionary containing configuration parameters.
                - k_steps (int): The number of MCMC steps to run.
                - step_size (float): The step size for the gradient update.
                - noise_scale (float): The scale of the Gaussian noise to add at each step.
        """
        super().__init__(config)
        self.k_steps = config.get('k_steps', 20)
        self.step_size = config.get('step_size', 0.1)
        self.noise_scale = config.get('noise_scale', 0.01)

    def sample(self, ebm_model: EBM, n_samples: int, sample_shape: Tuple, initial_states: torch.Tensor = None) -> torch.Tensor:
        """
        Generates samples using Langevin Dynamics.

        Args:
            ebm_model (EBM): The Energy-Based Model to sample from.
            n_samples (int): The number of samples to generate.
            sample_shape (Tuple): The shape of a single sample (e.g., (channels, height, width)).
            initial_states (torch.Tensor, optional): Starting points for the MCMC chains. If None,
                                                     samples are initialized from a uniform distribution.

        Returns:
            torch.Tensor: The generated samples.
        """
        # Get the device directly from the EBM model's new property
        device = ebm_model.device

        if initial_states is not None:
            samples = initial_states.to(device)
        else:
            # Initialize samples randomly from a uniform distribution over [-1, 1]
            samples = torch.rand(n_samples, *sample_shape, device=device) * 2 - 1

        for _ in range(self.k_steps):
            # We need to compute gradients with respect to the samples
            samples.requires_grad_(True)

            # Calculate the energy of the current samples
            energy = ebm_model(samples)

            # Calculate the gradient of the energy with respect to the samples
            # We set create_graph=False as we don't need to backprop through the MCMC process
            grad, = torch.autograd.grad(energy.sum(), samples, create_graph=False)

            # Detach samples before the update step to break the computation graph
            samples = samples.detach()

            # The Langevin dynamics update rule
            # x_{t+1} = x_t - step_size * grad_E(x_t) + noise_scale * z_t
            samples -= self.step_size * grad
            samples += self.noise_scale * torch.randn_like(samples)

        # Return the final samples, detached from the computation graph
        return samples.detach()

    @staticmethod
    def get_hyperparameter_info() -> dict:
        return {
            "k_steps": {
                "description": "Number of MCMC steps to run.",
                "recommended": "20-60 for training, can be higher for sampling.",
                "type": "int"
            },
            "step_size": {
                "description": "Step size (learning rate) for the Langevin dynamics update.",
                "recommended": "Varies greatly. Start with 1e-4 to 1e-5.",
                "type": "float"
            },
            "noise_scale": {
                "description": "Scale of the Gaussian noise added at each step.",
                "recommended": "Typically small, e.g., 0.005 to 0.01.",
                "type": "float"
            }
        }


class ReplayBufferLangevinSampler(LangevinSampler):
    """
    A Langevin sampler that uses a replay buffer to initialize MCMC chains.
    This helps to "mix" the chains better and avoids starting from scratch every time,
    which is a common technique for training EBMs.
    """

    def __init__(self, config: Dict):
        """
        Initializes the ReplayBufferLangevinSampler.

        Args:
            config (Dict): A dictionary containing configuration parameters.
                - All parameters for LangevinSampler.
                - buffer_size (int): The maximum number of samples to store in the buffer.
                - replay_probability (float): The probability of re-initializing a chain from the buffer.
        """
        super().__init__(config)
        self.buffer_size = config.get('buffer_size', 10000)
        self.replay_probability = config.get('replay_probability', 0.95)

        # The buffer is lazily initialized on the first call to sample()
        # to ensure it's on the correct device and has the correct shape.
        self.replay_buffer = None
        self.buffer_ptr = 0

    def sample(self, ebm_model: EBM, n_samples: int, sample_shape: Tuple, initial_states: torch.Tensor = None) -> torch.Tensor:
        """
        Generates samples, using the replay buffer to provide initial states for the MCMC chains.

        Args:
            ebm_model (EBM): The Energy-Based Model to sample from.
            n_samples (int): The number of samples to generate.
            sample_shape (Tuple): The shape of a single sample.
            initial_states (torch.Tensor, optional): If provided, these are used directly, bypassing the
                                                     replay buffer logic for this call. Defaults to None.

        Returns:
            torch.Tensor: The generated samples.
        """
        device = ebm_model.device

        # Lazily initialize the buffer on the first run
        if self.replay_buffer is None:
            self.replay_buffer = torch.rand(self.buffer_size, *sample_shape, device=device) * 2 - 1

        if initial_states is None:
            # If no initial states are given, create them from buffer and random noise
            n_replay = int(n_samples * self.replay_probability)
            n_random = n_samples - n_replay

            replay_starts = torch.empty(0, *sample_shape, device=device)
            if n_replay > 0 and self.buffer_size > 0:
                # Get random indices from the buffer
                buffer_indices = torch.randint(0, self.buffer_size, (n_replay,), device=device)
                replay_starts = self.replay_buffer[buffer_indices]

            random_starts = torch.empty(0, *sample_shape, device=device)
            if n_random > 0:
                # Initialize the rest randomly
                random_starts = torch.rand(n_random, *sample_shape, device=device) * 2 - 1

            # Combine the initial states
            initial_states = torch.cat([replay_starts, random_starts], dim=0)

        # Run the actual MCMC sampling using the parent class method
        final_samples = super().sample(ebm_model, n_samples, sample_shape, initial_states=initial_states)

        # Add the new samples to the replay buffer (circular buffer logic)
        if self.buffer_size > 0:
            if self.buffer_ptr + n_samples > self.buffer_size:
                # Wrap around the buffer
                part1_size = self.buffer_size - self.buffer_ptr
                self.replay_buffer[self.buffer_ptr:] = final_samples[:part1_size]
                self.replay_buffer[:n_samples - part1_size] = final_samples[part1_size:]
                self.buffer_ptr = n_samples - part1_size
            else:
                self.replay_buffer[self.buffer_ptr:self.buffer_ptr + n_samples] = final_samples
                self.buffer_ptr += n_samples

        return final_samples

    @staticmethod
    def get_hyperparameter_info() -> dict:
        # Start with the parent's hyperparameters
        info = super(ReplayBufferLangevinSampler, ReplayBufferLangevinSampler).get_hyperparameter_info()
        # Add the new ones
        info.update({
            "buffer_size": {
                "description": "Maximum number of samples to store in the replay buffer.",
                "recommended": "10000",
                "type": "int"
            },
            "replay_probability": {
                "description": "Probability of re-initializing a chain from the buffer vs. from noise.",
                "recommended": "0.95 (i.e., 5% refresh rate from noise)",
                "type": "float"
            }
        })
        return info


class MALASampler(Sampler):
    """
    Implements the Metropolis-Adjusted Langevin Algorithm (MALA). This method uses
    a Langevin-based proposal and a Metropolis-Hastings acceptance step to sample
    from the EBM's distribution. This can improve sample quality by correcting for
    discretization errors in the Langevin dynamics.
    """

    def __init__(self, config: Dict):
        """
        Initializes the MALASampler.

        Args:
            config (Dict): A dictionary containing configuration parameters.
                - k_steps (int): The number of MCMC steps to run.
                - step_size (float): The step size for the gradient update.
                - noise_scale (float): The scale of the Gaussian noise for the proposal.
        """
        super().__init__(config)
        self.k_steps = config.get('k_steps', 20)
        self.step_size = config.get('step_size', 0.1)
        self.noise_scale = config.get('noise_scale', 0.01)

    def sample(self, ebm_model: EBM, n_samples: int, sample_shape: Tuple, initial_states: torch.Tensor = None) -> torch.Tensor:
        """
        Generates samples using MALA.

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
            samples = initial_states.to(device)
        else:
            # Initialize samples randomly from a uniform distribution over [-1, 1]
            samples = torch.rand(n_samples, *sample_shape, device=device) * 2 - 1

        for _ in range(self.k_steps):
            # --- MALA Step ---
            samples.requires_grad_(True)
            energy_current = ebm_model(samples)
            grad, = torch.autograd.grad(energy_current.sum(), samples, create_graph=False)

            # 1. Generate proposal samples using one step of Langevin dynamics
            proposal_samples = samples.detach() - self.step_size * grad + self.noise_scale * torch.randn_like(samples)

            # 2. Calculate acceptance probability (simplified M-H, ignoring proposal ratio)
            # This is technically Metropolis-Hastings with a Langevin proposal.
            energy_proposal = ebm_model(proposal_samples)
            log_acceptance_ratio = energy_current.detach() - energy_proposal
            acceptance_ratio = torch.exp(log_acceptance_ratio).clamp_(0, 1)

            # 3. Accept or reject the proposal
            u = torch.rand_like(acceptance_ratio)
            accept_mask = (u < acceptance_ratio)

            # Unsqueeze mask to match sample dimensions for torch.where
            while accept_mask.dim() < samples.dim():
                accept_mask = accept_mask.unsqueeze(-1)

            samples = torch.where(accept_mask, proposal_samples, samples).detach()
            # --- End MALA Step ---

        return samples.detach()

    @staticmethod
    def get_hyperparameter_info() -> dict:
        # MALA shares the same core hyperparameters as Langevin
        return {
            "k_steps": {
                "description": "Number of MCMC steps to run.",
                "recommended": "20-60 for training, can be higher for sampling.",
                "type": "int"
            },
            "step_size": {
                "description": "Step size (learning rate) for the Langevin proposal.",
                "recommended": "Varies greatly. Start with 1e-4 to 1e-5.",
                "type": "float"
            },
            "noise_scale": {
                "description": "Scale of the Gaussian noise for the Langevin proposal.",
                "recommended": "Typically small, e.g., 0.005 to 0.01.",
                "type": "float"
            }
        }
