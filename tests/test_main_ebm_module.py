import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM
from ebm_lib.samplers.langevin import LangevinSampler
from ebm_lib.regularizers.energy import L2EnergyRegularizer
from ebm_lib.lightning_module import EBMLightningModule

@pytest.fixture
def lightning_module_components():
    """Provides a standard set of components for initializing the EBMLightningModule."""
    ebm_network = nn.Linear(10, 1)
    ebm_model = EBM(ebm_network)
    sampler = LangevinSampler(config={'k_steps': 1, 'step_size': 0.1})
    optimizer_config = {'name': 'Adam', 'lr': 1e-4}
    regularizers = [L2EnergyRegularizer(config={'lambda_e': 0.1})]

    return {
        "ebm_model": ebm_model,
        "sampler": sampler,
        "optimizer_config": optimizer_config,
        "regularizers": regularizers
    }

def test_lightning_module_init(lightning_module_components):
    """Tests that the EBMLightningModule initializes correctly."""
    module = EBMLightningModule(**lightning_module_components)

    assert module.ebm_model is lightning_module_components["ebm_model"]
    assert module.sampler is lightning_module_components["sampler"]
    assert module.optimizer_config == lightning_module_components["optimizer_config"]
    assert len(module.regularizers) == 1
    assert isinstance(module.regularizers[0], L2EnergyRegularizer)

def test_lightning_module_configure_optimizers(lightning_module_components):
    """Tests the configure_optimizers method."""
    module = EBMLightningModule(**lightning_module_components)
    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults['lr'] == 1e-4

def test_lightning_module_training_step(lightning_module_components):
    """
    Tests that the training_step runs and returns a scalar loss tensor.
    This is a functional test to ensure the step logic executes without errors.
    """
    module = EBMLightningModule(**lightning_module_components)

    # Create a dummy batch of data
    batch = torch.randn(5, 10)

    # Run the training step
    loss = module.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 # Loss should be a scalar
    assert loss.requires_grad # Loss should have a grad_fn for backpropagation

def test_multi_scale_energy_calculation(lightning_module_components):
    """
    Tests that the multi-scale energy calculation correctly sums the
    energies from different scales.
    """
    # Use a mock EBM that always returns a constant energy of 1.0
    class ConstantEnergyEBM(EBM):
        def __init__(self):
            super().__init__(nn.Identity())
        def forward(self, x):
            return torch.ones(x.shape[0], device=x.device)

    # Replace the EBM in the components with our mock EBM
    lightning_module_components['ebm_model'] = ConstantEnergyEBM()

    # Initialize the module with 3 scales
    module = EBMLightningModule(**lightning_module_components, num_scales=3)

    # Use an image batch that can be downsampled multiple times
    image_batch = torch.randn(4, 3, 32, 32)

    # Calculate the multi-scale energy
    total_energy = module._calculate_multi_scale_energy(image_batch)

    # The mock EBM returns 1.0 for each scale. With 3 scales, the total
    # energy for each sample should be 1.0 * 3 = 3.0.
    expected_energy = torch.full((4,), 3.0)

    assert total_energy.shape == (4,)
    assert torch.allclose(total_energy, expected_energy)
