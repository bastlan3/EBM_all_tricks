import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM
from ebm_lib.regularizers.gradient import GradientPenaltyRegularizer

class LinearEBM(EBM):
    """
    An EBM with a linear energy function E(x) = w^T @ x. The gradient of this
    function with respect to x is constant and equal to w, which is ideal for
    testing the gradient penalty.
    """
    def __init__(self, weight: torch.Tensor):
        # The 'network' is used to hold the weight parameter so that autograd
        # can track it.
        super().__init__(nn.Linear(weight.numel(), 1, bias=False))
        # Set the weight of the linear layer to our desired gradient vector.
        self.network.weight.data = weight.view(1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape x to be (batch_size, -1) for the linear layer
        x_flat = x.view(x.shape[0], -1)
        return self.network(x_flat).squeeze(-1)

@pytest.fixture
def regularizer_config() -> dict:
    """Provides a standard configuration for the gradient penalty regularizer."""
    return {'lambda_gp': 2.0, 'target': 1.0}

def test_gradient_penalty_init(regularizer_config):
    """Tests that the regularizer initializes with the correct attributes."""
    regularizer = GradientPenaltyRegularizer(regularizer_config)
    assert regularizer.lambda_gp == 2.0
    assert regularizer.target == 1.0

def test_gradient_penalty_loss_calculation(regularizer_config):
    """
    Tests that the gradient penalty loss is calculated correctly using a linear
    EBM with a constant, known gradient.
    """
    regularizer = GradientPenaltyRegularizer(regularizer_config)

    # Create a linear EBM with a weight vector w. The gradient of E(x) is w.
    # We choose w such that its L2 norm is 3.0.
    weight = torch.tensor([1.0, 2.0, 2.0]) # Norm is sqrt(1+4+4) = 3.0
    ebm_model = LinearEBM(weight)

    # The interpolated samples don't matter since the gradient is constant.
    pos_samples = torch.randn(10, 3)
    neg_samples = torch.randn(10, 3)

    loss = regularizer.calculate_loss(ebm_model, pos_samples, neg_samples)

    # The gradient norm for this EBM is always ||w|| = 3.0.
    # The penalty per sample is (grad_norm - target)^2 = (3.0 - 1.0)^2 = 4.0.
    # The mean penalty is 4.0.
    # The final loss is lambda_gp * penalty = 2.0 * 4.0 = 8.0.
    expected_loss = regularizer_config['lambda_gp'] * ((torch.norm(weight) - regularizer_config['target'])**2)

    assert torch.isclose(loss, expected_loss, atol=1e-5)

def test_gradient_penalty_zero_lambda():
    """Tests that the loss is zero if lambda_gp is zero."""
    config = {'lambda_gp': 0.0}
    regularizer = GradientPenaltyRegularizer(config)

    weight = torch.tensor([10.0, 20.0]) # High gradient norm
    ebm_model = LinearEBM(weight)

    pos_samples = torch.randn(10, 2)
    neg_samples = torch.randn(10, 2)

    loss = regularizer.calculate_loss(ebm_model, pos_samples, neg_samples)

    assert torch.isclose(loss, torch.tensor(0.0))
