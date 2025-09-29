import pytest
import torch
import torch.nn as nn

# Adjust the import path to be relative to the project root
from ebm_lib.ebm import EBM
from ebm_lib.samplers.langevin import LangevinSampler, ReplayBufferLangevinSampler, MALASampler

# A simple EBM for testing purposes.
# The energy function E(x) = 0.5 * ||x||^2 is minimized at x=0.
# The gradient is simply grad_E(x) = x.
class SimpleQuadraticEBM(EBM):
    def __init__(self):
        super().__init__(network=nn.Identity()) # The network is not used here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the sum of squares for each sample in the batch
        return 0.5 * torch.sum(x * x, dim=list(range(1, x.dim())))

@pytest.fixture
def ebm_model() -> SimpleQuadraticEBM:
    """Provides a simple EBM instance for testing."""
    return SimpleQuadraticEBM()

@pytest.fixture
def sampler_config() -> dict:
    """Provides a standard configuration for the Langevin sampler."""
    return {'k_steps': 10, 'step_size': 0.1, 'noise_scale': 0.01}

def test_langevin_sampler_init(sampler_config):
    """Tests that the sampler initializes with the correct attributes."""
    sampler = LangevinSampler(sampler_config)
    assert sampler.k_steps == 10
    assert sampler.step_size == 0.1
    assert sampler.noise_scale == 0.01

def test_langevin_sample_shape(ebm_model, sampler_config):
    """Tests that the sampler returns a tensor of the correct shape."""
    sampler = LangevinSampler(sampler_config)
    n_samples = 5
    sample_shape = (2,)  # A simple 2-dimensional sample
    samples = sampler.sample(ebm_model, n_samples, sample_shape)
    assert samples.shape == (n_samples, *sample_shape)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for device test")
def test_langevin_sample_device(ebm_model, sampler_config):
    """Tests that the output samples are on the same device as the model."""
    device = torch.device("cuda:0")
    ebm_model.to(device)
    sampler = LangevinSampler(sampler_config)
    n_samples = 5
    sample_shape = (2,)
    samples = sampler.sample(ebm_model, n_samples, sample_shape)
    assert samples.device.type == 'cuda'

def test_langevin_samples_change(ebm_model, sampler_config):
    """Tests that the MCMC process actually modifies the samples."""
    sampler = LangevinSampler(sampler_config)
    n_samples = 5
    sample_shape = (2,)
    initial_samples = torch.ones(n_samples, *sample_shape) * 10 # Start far from the minimum

    final_samples = sampler.sample(ebm_model, n_samples, sample_shape, initial_states=initial_samples.clone())

    # The samples should have changed after 10 MCMC steps
    assert not torch.allclose(initial_samples, final_samples)
    # For this simple EBM, the norm of the samples should decrease as they move towards the minimum (x=0)
    assert torch.norm(final_samples) < torch.norm(initial_samples)

def test_langevin_with_initial_states(ebm_model, sampler_config):
    """Tests that providing initial states works correctly."""
    sampler = LangevinSampler(sampler_config)
    n_samples = 5
    sample_shape = (2,)
    initial_states = torch.randn(n_samples, *sample_shape) * 5

    # Run for 0 steps to check if initial states are handled properly
    sampler.k_steps = 0
    final_samples = sampler.sample(ebm_model, n_samples, sample_shape, initial_states=initial_states.clone())

    # With k_steps=0, the output should be identical to the initial states
    assert torch.allclose(initial_states, final_samples)


# --- Tests for ReplayBufferLangevinSampler ---

@pytest.fixture
def replay_sampler_config() -> dict:
    """Provides a standard configuration for the replay buffer sampler."""
    return {
        'k_steps': 2,
        'step_size': 0.1,
        'noise_scale': 0.01,
        'buffer_size': 100,
        'replay_probability': 0.95
    }

def test_replay_sampler_init(replay_sampler_config):
    """Tests that the replay sampler initializes correctly."""
    sampler = ReplayBufferLangevinSampler(replay_sampler_config)
    assert sampler.buffer_size == 100
    assert sampler.replay_probability == 0.95
    assert sampler.replay_buffer is None

def test_replay_buffer_creation(ebm_model, replay_sampler_config):
    """Tests that the replay buffer is created on the first sample call."""
    sampler = ReplayBufferLangevinSampler(replay_sampler_config)
    n_samples = 10
    sample_shape = (2,)
    sampler.sample(ebm_model, n_samples, sample_shape)

    assert sampler.replay_buffer is not None
    assert sampler.replay_buffer.shape == (sampler.buffer_size, *sample_shape)
    assert sampler.replay_buffer.device.type == ebm_model.device.type

def test_replay_buffer_update(ebm_model, replay_sampler_config):
    """Tests that the replay buffer is updated correctly after sampling."""
    sampler = ReplayBufferLangevinSampler(replay_sampler_config)
    n_samples = 10
    sample_shape = (2,)

    # First call to initialize and populate the buffer
    initial_generated_samples = sampler.sample(ebm_model, n_samples, sample_shape)

    # Check that the buffer contains these samples at the start
    assert torch.allclose(sampler.replay_buffer[:n_samples], initial_generated_samples)
    assert sampler.buffer_ptr == n_samples

    # Second call
    next_generated_samples = sampler.sample(ebm_model, n_samples, sample_shape)
    assert not torch.allclose(initial_generated_samples, next_generated_samples)
    assert torch.allclose(sampler.replay_buffer[n_samples:2 * n_samples], next_generated_samples)
    assert sampler.buffer_ptr == 2 * n_samples

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for device test")
def test_replay_buffer_device(ebm_model, replay_sampler_config):
    """Tests that the replay buffer is moved to the correct device."""
    device = torch.device("cuda:0")
    ebm_model.to(device)

    sampler = ReplayBufferLangevinSampler(replay_sampler_config)
    sampler.sample(ebm_model, n_samples=5, sample_shape=(2,))

    assert sampler.replay_buffer.device.type == 'cuda'


# --- Tests for MALASampler ---

@pytest.fixture
def mala_sampler_config() -> dict:
    """Provides a config for the MALA sampler."""
    return {'k_steps': 20, 'step_size': 0.1, 'noise_scale': 0.01}

def test_mala_sampler_init(mala_sampler_config):
    """Tests that the MALA sampler initializes correctly."""
    sampler = MALASampler(mala_sampler_config)
    assert sampler.k_steps == 20
    assert sampler.step_size == 0.1

def test_mala_sample_shape(ebm_model, mala_sampler_config):
    """Tests that the MALA sampler returns a tensor of the correct shape."""
    sampler = MALASampler(mala_sampler_config)
    n_samples = 5
    sample_shape = (2,)
    samples = sampler.sample(ebm_model, n_samples, sample_shape)
    assert samples.shape == (n_samples, *sample_shape)

def test_mala_acceptance_logic(mala_sampler_config):
    """
    Functionally tests that MALA samples move towards lower-energy regions,
    which implies the acceptance/rejection step is working correctly.
    """
    ebm_quad = SimpleQuadraticEBM()
    sampler = MALASampler(mala_sampler_config)

    # Start samples far from the energy minimum (x=0)
    initial_samples = torch.ones(5, 2) * 10

    final_samples = sampler.sample(ebm_quad, 5, (2,), initial_states=initial_samples.clone())

    # The norm of the final samples should be smaller, as they have moved towards x=0
    assert torch.norm(final_samples) < torch.norm(initial_samples)

def test_langevin_sampler_with_noise_generator(ebm_model, sampler_config):
    """
    Tests that the Langevin sampler can use a custom noise generator for
    initializing the MCMC chains.
    """
    from ebm_lib.noise import NoiseGenerator

    n_samples = 4
    sample_shape = (1, 8, 8)

    # Create a noise generator
    noise_gen = NoiseGenerator(size=(8, 8), batch_size=n_samples, device=ebm_model.device)

    # Create two samplers: one with the generator, one without
    sampler_with_gen = LangevinSampler(sampler_config, noise_generator=noise_gen)
    sampler_without_gen = LangevinSampler(sampler_config)

    # Run both for 0 steps to get the initial samples
    sampler_with_gen.k_steps = 0
    sampler_without_gen.k_steps = 0

    # Generate initial samples
    # The sampler with the generator should produce brown noise by default
    initial_samples_gen = sampler_with_gen.sample(ebm_model, n_samples, sample_shape)
    # The other sampler should produce uniform noise
    initial_samples_uniform = sampler_without_gen.sample(ebm_model, n_samples, sample_shape)

    # The two sets of initial samples should be different
    assert not torch.allclose(initial_samples_gen, initial_samples_uniform)

def test_sampler_with_noise_curriculum(ebm_model, sampler_config):
    """
    Tests that the sampler generates different initial noise based on the
    training progress when using a noise generator.
    """
    from ebm_lib.noise import NoiseGenerator

    n_samples = 4
    sample_shape = (1, 8, 8)
    noise_gen = NoiseGenerator(size=(8, 8), batch_size=n_samples, device=ebm_model.device)
    sampler = LangevinSampler(sampler_config, noise_generator=noise_gen)
    sampler.k_steps = 0 # We only want to inspect the initial samples

    # Noise at the beginning of training (progress=0.0) should be brown
    brown_noise = sampler.sample(ebm_model, n_samples, sample_shape, progress=0.0)

    # Noise at the end of training (progress=1.0) should be white
    white_noise = sampler.sample(ebm_model, n_samples, sample_shape, progress=1.0)

    # The two noise types should be statistically different
    assert not torch.allclose(brown_noise, white_noise)

    # A simple check: brown noise has more low-frequency components, so it should
    # be smoother and have a smaller variance of its pixel-wise difference.
    brown_diff_var = torch.var(torch.diff(brown_noise))
    white_diff_var = torch.var(torch.diff(white_noise))
    assert brown_diff_var < white_diff_var
