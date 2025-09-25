import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM
from ebm_lib.regularizers.score_matching import DenoisingScoreMatchingRegularizer

class QuadraticEBM(EBM):
    """
    An EBM with a simple quadratic energy function: E(x) = 0.5 * sum(w * x^2).
    The score is -grad(E(x)) = -w*x, which is easy to compute analytically.
    """
    def __init__(self, weight: torch.Tensor):
        super().__init__(nn.Identity())
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for broadcasting: (1, D)
        w = self.weight.view(1, -1)
        x_flat = x.view(x.shape[0], -1)
        # Energy = 0.5 * sum(w * x^2) over the feature dimension
        return 0.5 * torch.sum(w * x_flat**2, dim=1)

@pytest.fixture
def dsm_config() -> dict:
    """Provides a standard configuration for the DSM regularizer."""
    return {'lambda_dsm': 1.0, 'sigma': 0.1}

def test_dsm_regularizer_init(dsm_config):
    """Tests that the regularizer initializes correctly."""
    regularizer = DenoisingScoreMatchingRegularizer(dsm_config)
    assert regularizer.lambda_dsm == 1.0
    assert regularizer.sigma == 0.1

def test_dsm_loss_calculation(dsm_config):
    """
    Tests the DSM loss calculation with a simple EBM where the score function
    is known analytically.
    """
    regularizer = DenoisingScoreMatchingRegularizer(dsm_config)

    # Define a simple EBM with a known weight for the energy function
    weight = torch.tensor([2.0, 2.0])
    ebm_model = QuadraticEBM(weight)

    # Use a fixed data point and noise for a deterministic test
    positive_samples = torch.tensor([[1.0, 2.0]])
    noise = torch.tensor([[0.5, -0.5]]) * dsm_config['sigma']

    # Override the regularizer's internal noise generation method for a deterministic test
    regularizer._get_noise = lambda samples: noise

    # Calculate the loss using the regularizer
    loss = regularizer.calculate_loss(ebm_model, positive_samples, torch.empty(0))

    # --- Manual Calculation for Verification ---
    sigma = dsm_config['sigma']
    perturbed_samples = positive_samples + noise # [1.05, 1.95]

    # Expected model score: -w * x_tilde
    expected_model_score = -weight * perturbed_samples

    # Expected true score: -noise / sigma^2
    expected_true_score = -noise / (sigma**2)

    # Expected loss: 0.5 * ||s_model - s_true||^2
    expected_loss = 0.5 * torch.sum((expected_model_score - expected_true_score)**2)

    assert torch.isclose(loss, expected_loss, atol=1e-5)