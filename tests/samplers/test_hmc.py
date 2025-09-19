import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM
from ebm_lib.samplers.hmc import HMCSampler

# A simple EBM with a quadratic energy function E(x) = 0.5 * ||x||^2, minimized at x=0.
# This is useful for testing samplers as the gradient is simply grad_E(x) = x.
class SimpleQuadraticEBM(EBM):
    def __init__(self):
        super().__init__(network=nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x * x, dim=list(range(1, x.dim())))

@pytest.fixture
def ebm_model() -> SimpleQuadraticEBM:
    """Provides a simple EBM instance for testing."""
    return SimpleQuadraticEBM()

@pytest.fixture
def hmc_sampler_config() -> dict:
    """Provides a standard configuration for the HMC sampler."""
    return {'n_leapfrog_steps': 10, 'step_size': 0.1}

def test_hmc_sampler_init(hmc_sampler_config):
    """Tests that the HMC sampler initializes correctly."""
    sampler = HMCSampler(hmc_sampler_config)
    assert sampler.n_leapfrog_steps == 10
    assert sampler.step_size == 0.1

def test_hmc_sample_shape(ebm_model, hmc_sampler_config):
    """Tests that the HMC sampler returns a tensor of the correct shape."""
    sampler = HMCSampler(hmc_sampler_config)
    n_samples = 5
    sample_shape = (2,)
    samples = sampler.sample(ebm_model, n_samples, sample_shape)
    assert samples.shape == (n_samples, *sample_shape)

def test_hmc_energy_conservation(ebm_model):
    """
    Tests that the Hamiltonian is roughly conserved during leapfrog integration
    for a very small step size, which is a key property of the integrator.
    """
    config = {'n_leapfrog_steps': 20, 'step_size': 1e-3}
    sampler = HMCSampler(config)

    q = torch.randn(1, 2)
    p = torch.randn_like(q)

    current_U = sampler._compute_potential_energy(ebm_model, q)
    current_K = sampler._compute_kinetic_energy(p)
    current_H = current_U + current_K

    # Manually perform leapfrog integration to check energy conservation
    q_new, p_new = q.clone(), p.clone()
    grad = sampler._get_grad(ebm_model, q_new)
    p_new -= 0.5 * sampler.step_size * grad
    for _ in range(sampler.n_leapfrog_steps - 1):
        q_new += sampler.step_size * p_new
        grad = sampler._get_grad(ebm_model, q_new)
        p_new -= sampler.step_size * grad
    q_new += sampler.step_size * p_new
    grad = sampler._get_grad(ebm_model, q_new)
    p_new -= 0.5 * sampler.step_size * grad

    proposal_U = sampler._compute_potential_energy(ebm_model, q_new)
    proposal_K = sampler._compute_kinetic_energy(p_new)
    proposal_H = proposal_U + proposal_K

    # The change in Hamiltonian should be very small for a good integrator
    assert torch.allclose(current_H, proposal_H, atol=1e-3)

def test_hmc_sampling_movement(ebm_model, hmc_sampler_config):
    """
    Functionally tests that HMC samples move towards lower-energy regions,
    implying the overall algorithm is working.
    """
    sampler = HMCSampler(hmc_sampler_config)
    # Start samples far from the energy minimum (x=0)
    initial_samples = torch.ones(5, 2) * 10

    final_samples = sampler.sample(ebm_model, 5, (2,), initial_states=initial_samples.clone())

    # The norm of the final samples should be smaller, as they have moved towards x=0
    assert torch.norm(final_samples) < torch.norm(initial_samples)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for device test")
def test_hmc_sample_device(ebm_model, hmc_sampler_config):
    """Tests that the HMC sampler works correctly on a CUDA device."""
    device = torch.device("cuda:0")
    ebm_model.to(device)
    sampler = HMCSampler(hmc_sampler_config)
    n_samples = 5
    sample_shape = (2,)
    samples = sampler.sample(ebm_model, n_samples, sample_shape)
    assert samples.device.type == 'cuda'
