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
