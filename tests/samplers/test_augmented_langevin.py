import pytest
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

from ebm_lib.ebm import EBM
from ebm_lib.samplers.augmented_langevin import AugmentedLangevinSampler

class SimpleQuadraticEBM(EBM):
    """A simple EBM with a quadratic energy function E(x) = 0.5 * ||x||^2."""
    def __init__(self):
        super().__init__(network=nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x * x, dim=list(range(1, x.dim())))

@pytest.fixture
def ebm_model() -> SimpleQuadraticEBM:
    return SimpleQuadraticEBM()

@pytest.fixture
def sampler_config() -> dict:
    return {'k_steps': 10, 'step_size': 0.1, 'noise_scale': 0.01, 'augment_interval': 5}

def test_augmented_sampler_init(sampler_config):
    """Tests that the sampler initializes correctly."""
    sampler = AugmentedLangevinSampler(sampler_config)
    assert sampler.augment_interval == 5
    assert sampler.transform is None

def test_augmented_sampler_no_transform(ebm_model, sampler_config):
    """Tests that the sampler behaves like a normal Langevin sampler if no transform is given."""
    sampler = AugmentedLangevinSampler(sampler_config, transform=None)
    initial_samples = torch.ones(4, 1, 8, 8)
    final_samples = sampler.sample(ebm_model, 4, (1, 8, 8), initial_states=initial_samples.clone())

    # It should have moved from the initial state
    assert not torch.allclose(initial_samples, final_samples)

def test_augmentation_is_applied(ebm_model, sampler_config):
    """
    Tests that the augmentation transform is correctly applied during the MCMC chain.
    """
    # Use a deterministic transform like a 90-degree rotation
    # Note: torchvision.transforms.v2 operates on tensors directly
    transform = transforms.RandomRotation(degrees=(90, 90))

    sampler = AugmentedLangevinSampler(sampler_config, transform=transform)
    sampler.k_steps = 5 # Ensure augmentation is applied exactly once on the last step
    sampler.augment_interval = 5
    sampler.noise_scale = 0.0 # Disable noise for a deterministic test
    sampler.step_size = 0.0 # Disable gradient step for a deterministic test

    # Create a non-symmetrical initial state to clearly see the rotation
    initial_samples = torch.zeros(1, 1, 4, 4)
    initial_samples[0, 0, 0, :] = 1.0 # A line at the top

    final_samples = sampler.sample(ebm_model, 1, (1, 4, 4), initial_states=initial_samples.clone())

    # The expected output is the initial sample rotated by 90 degrees
    expected_output = torch.rot90(initial_samples, 1, [2, 3])

    assert not torch.allclose(initial_samples, final_samples)
    assert torch.allclose(final_samples, expected_output)