import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM
from ebm_lib.regularizers.energy import L2EnergyRegularizer

class MockEBM(EBM):
    """A mock EBM that returns a fixed energy value for any input."""
    def __init__(self, energy_val: float):
        super().__init__(nn.Identity())
        self.energy_val = energy_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return a tensor of shape (batch_size,) with the fixed energy value.
        return torch.full((x.shape[0],), self.energy_val, device=x.device)

@pytest.fixture
def regularizer_config() -> dict:
    """Provides a standard configuration for the regularizer."""
    return {'lambda_e': 0.5}

def test_l2_energy_regularizer_init(regularizer_config):
    """Tests that the regularizer initializes with the correct lambda value."""
    regularizer = L2EnergyRegularizer(regularizer_config)
    assert regularizer.lambda_e == 0.5

def test_l2_energy_regularizer_loss_calculation(regularizer_config):
    """
    Tests that the loss is calculated correctly using a mock EBM that produces
    a constant, predictable energy.
    """
    regularizer = L2EnergyRegularizer(regularizer_config)

    # Create a mock EBM that always returns an energy of 2.0
    ebm_model = MockEBM(energy_val=2.0)

    pos_samples = torch.randn(10, 2)
    neg_samples = torch.randn(10, 2)

    loss = regularizer.calculate_loss(ebm_model, pos_samples, neg_samples)

    # positive_energy will be a tensor of [2., 2., ...], so mean(energy^2) is 4.0.
    # negative_energy will be a tensor of [2., 2., ...], so mean(energy^2) is 4.0.
    # The total penalty is (4.0 + 4.0).
    # The final loss is lambda * penalty.
    expected_loss = regularizer.lambda_e * (2.0**2 + 2.0**2)

    assert torch.isclose(loss, torch.tensor(expected_loss))

def test_l2_energy_regularizer_zero_lambda():
    """Tests that the loss is zero if lambda is zero."""
    config = {'lambda_e': 0.0}
    regularizer = L2EnergyRegularizer(config)
    ebm_model = MockEBM(energy_val=100.0) # High energy should not matter

    pos_samples = torch.randn(10, 2)
    neg_samples = torch.randn(10, 2)

    loss = regularizer.calculate_loss(ebm_model, pos_samples, neg_samples)

    assert torch.isclose(loss, torch.tensor(0.0))
