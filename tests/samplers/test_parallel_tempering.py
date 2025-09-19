import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM
from ebm_lib.samplers.parallel_tempering import ParallelTemperingSampler

class SimpleQuadraticEBM(EBM):
    """A simple EBM with a quadratic energy function E(x) = 0.5 * ||x||^2."""
    def __init__(self):
        super().__init__(network=nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(x * x, dim=list(range(1, x.dim())))

@pytest.fixture
def ebm_model() -> SimpleQuadraticEBM:
    """Provides a simple EBM instance for testing."""
    return SimpleQuadraticEBM()

@pytest.fixture
def pt_sampler_config() -> dict:
    """Provides a standard configuration for the Parallel Tempering sampler."""
    return {
        'n_chains': 4,
        'k_steps': 10,
        'swap_interval': 5,
        'step_size': 0.1,
        'noise_scale': 0.01,
        'betas': (1.0, 0.5, 0.2, 0.1)
    }

def test_pt_sampler_init(pt_sampler_config):
    """Tests that the Parallel Tempering sampler initializes correctly."""
    sampler = ParallelTemperingSampler(pt_sampler_config)
    assert sampler.n_chains == 4
    assert sampler.k_steps == 10
    assert sampler.swap_interval == 5
    assert sampler.chains_state is None
    assert len(sampler.betas) == 4

def test_pt_sample_shape_and_state_creation(ebm_model, pt_sampler_config):
    """
    Tests that the sampler returns a tensor of the correct shape and that
    the internal chain state is initialized correctly.
    """
    sampler = ParallelTemperingSampler(pt_sampler_config)
    n_samples = 5
    sample_shape = (2,)

    # Before sampling, the state should be None
    assert sampler.chains_state is None

    samples = sampler.sample(ebm_model, n_samples, sample_shape)

    # The output should be the samples from the coldest chain
    assert samples.shape == (n_samples, *sample_shape)

    # The internal state for all chains should now exist
    assert sampler.chains_state is not None
    assert sampler.chains_state.shape == (sampler.n_chains, n_samples, *sample_shape)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for device test")
def test_pt_sample_device(ebm_model, pt_sampler_config):
    """
    Tests that the sampler's internal state and output are on the correct CUDA device.
    """
    device = torch.device("cuda:0")
    ebm_model.to(device)
    sampler = ParallelTemperingSampler(pt_sampler_config)
    n_samples = 5
    sample_shape = (2,)

    samples = sampler.sample(ebm_model, n_samples, sample_shape)

    assert samples.device.type == 'cuda'
    assert sampler.chains_state.device.type == 'cuda'
    assert sampler.betas.device.type == 'cuda'

def test_pt_swap_functional(pt_sampler_config):
    """
    A functional test to ensure swaps are happening. We create a scenario where a swap
    is highly probable and check if the chain states have mixed.
    """
    # Use an EBM where energy is just the first coordinate
    class LinearEBM(EBM):
        def __init__(self):
            super().__init__(nn.Identity())
        def forward(self, x):
            return x[:, 0]

    ebm_model = LinearEBM()

    pt_sampler_config['k_steps'] = 6
    pt_sampler_config['swap_interval'] = 5
    pt_sampler_config['noise_scale'] = 0.0 # No noise for deterministic test
    sampler = ParallelTemperingSampler(pt_sampler_config)

    n_samples = 1
    sample_shape = (1,)

    # Manually set the chain states
    # Chain 0 (cold, beta=1.0) has low energy
    # Chain 1 (hot, beta=0.5) has high energy
    # A swap is energetically favorable (log_A = (0.5-1.0)*(0-10) = 5.0 > 0)
    sampler.chains_state = torch.zeros(sampler.n_chains, n_samples, *sample_shape)
    sampler.chains_state[0] = 0.0  # Energy = 0
    sampler.chains_state[1] = 10.0 # Energy = 10

    initial_cold_state = sampler.chains_state[0].clone()

    # Run sampling, which will include one swap attempt
    sampler.sample(ebm_model, n_samples, sample_shape)

    # After the swap and subsequent Langevin steps, the cold chain's state
    # should have changed significantly, likely towards the initial hot state.
    # We check that it's no longer near its original low-energy state.
    assert not torch.allclose(initial_cold_state, sampler.chains_state[0], atol=1e-1)
