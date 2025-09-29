import pytest
import torch
import torch.nn as nn

from ebm_lib.models.autoencoder import SimpleConvAutoencoder
from ebm_lib.models.latent_ebm import LatentEBM
from ebm_lib.latent_ebm_module import LatentEBMLightningModule
from ebm_lib.samplers.langevin import LangevinSampler

@pytest.fixture
def autoencoder():
    return SimpleConvAutoencoder(input_shape=(3, 32, 32), latent_dim=64)

def test_autoencoder_forward(autoencoder):
    """Tests that the autoencoder's output shape matches the input shape."""
    input_tensor = torch.randn(4, 3, 32, 32)
    reconstructed_tensor = autoencoder(input_tensor)
    assert reconstructed_tensor.shape == input_tensor.shape

@pytest.fixture
def latent_ebm(autoencoder):
    """Provides a LatentEBM instance."""
    latent_net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Flatten(0))
    return LatentEBM(latent_energy_net=latent_net, encoder=autoencoder)

def test_latent_ebm_init(latent_ebm):
    """Tests that the autoencoder within the LatentEBM is frozen."""
    for param in latent_ebm.encoder.parameters():
        assert not param.requires_grad

def test_latent_ebm_forward(latent_ebm):
    """Tests that the forward pass takes an image and returns a scalar energy."""
    input_images = torch.randn(4, 3, 32, 32)
    energies = latent_ebm(input_images)
    assert energies.shape == (4,)

@pytest.fixture
def latent_module_components(latent_ebm):
    """Provides components for the LatentEBMLightningModule."""
    sampler = LangevinSampler(config={'k_steps': 2})
    optimizer_config = {'name': 'Adam', 'lr': 1e-4}
    return {
        "ebm_model": latent_ebm,
        "sampler": sampler,
        "optimizer_config": optimizer_config,
        "regularizers": [] # No regularizers for this test
    }

def test_latent_lightning_module_init(latent_module_components):
    """Tests initialization of the latent space training module."""
    module = LatentEBMLightningModule(**latent_module_components)
    assert isinstance(module.ebm_model, LatentEBM)

def test_latent_lightning_module_init_raises_error(latent_module_components):
    """Tests that it raises an error if not given a LatentEBM."""
    from ebm_lib.ebm import EBM
    # Replace the LatentEBM with a standard EBM
    latent_module_components['ebm_model'] = EBM(nn.Linear(10,1))
    with pytest.raises(TypeError):
        LatentEBMLightningModule(**latent_module_components)

def test_latent_training_step(latent_module_components):
    """Tests that a training step runs correctly in the latent space."""
    module = LatentEBMLightningModule(**latent_module_components)

    # Mock the trainer attribute
    from unittest.mock import Mock
    mock_trainer = Mock()
    mock_trainer.global_step = 10
    mock_trainer.max_steps = 100
    module.trainer = mock_trainer

    # A batch of images
    image_batch = torch.randn(4, 3, 32, 32)

    loss = module.training_step(image_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

    # Test backward pass
    loss.backward()
    # Check that the latent EBM's parameters have gradients, but the autoencoder's do not
    assert next(module.ebm_model.network.parameters()).grad is not None
    assert next(module.ebm_model.encoder.parameters()).grad is None

@pytest.mark.slow
def test_stable_diffusion_vae_integration():
    """
    An integration test to verify that a pre-trained VAE from `diffusers`
    can be loaded and used correctly within the LatentEBM framework.
    This test will download model weights and may be slow.
    """
    from diffusers import AutoencoderKL

    # 1. Load the pre-trained Stable Diffusion VAE
    try:
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-v1-4", subfolder="vae")
    except Exception as e:
        pytest.skip(f"Could not download VAE model, skipping test. Error: {e}")

    # 2. Create a wrapper to provide a simple .encode() method
    class VAEEncoderWrapper(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def encode(self, x):
            return self.vae.encode(x).latent_dist.mean

    encoder = VAEEncoderWrapper(vae)

    # 3. Define a simple EBM for the VAE's latent space
    # The latent space of this VAE is 4x32x32 for a 256x256 image
    latent_dim = 4 * 32 * 32
    latent_energy_net = nn.Sequential(nn.Linear(latent_dim, 1), nn.Flatten(0))

    # 4. Create the LatentEBM
    latent_ebm = LatentEBM(latent_energy_net=latent_energy_net, encoder=encoder)

    # 5. Test the forward pass
    image_batch = torch.randn(1, 3, 256, 256)
    energy = latent_ebm(image_batch)

    assert isinstance(energy, torch.Tensor)
    assert energy.shape == tuple() # Should be a scalar