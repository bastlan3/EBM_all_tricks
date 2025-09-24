import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM
from ebm_lib.samplers.differentiable_langevin import DifferentiableLangevinSampler
from ebm_lib.estimators.knn_entropy import KnnEntropyEstimator
from ebm_lib.corrected_cd.lightning_module import CorrectedCDLightningModule

@pytest.fixture
def corrected_cd_components():
    """Provides components needed to initialize the CorrectedCDLightningModule."""
    ebm_model = EBM(nn.Linear(10, 1))
    sampler = DifferentiableLangevinSampler(config={'k_steps': 2})
    knn_estimator = KnnEntropyEstimator(k=3)
    optimizer_config = {'name': 'Adam', 'lr': 1e-4}

    return {
        "ebm_model": ebm_model,
        "sampler": sampler,
        "knn_entropy_estimator": knn_estimator,
        "optimizer_config": optimizer_config
    }

def test_corrected_cd_init(corrected_cd_components):
    """Tests that the module initializes correctly."""
    module = CorrectedCDLightningModule(**corrected_cd_components)
    assert module.ebm_model is corrected_cd_components["ebm_model"]
    assert module.sampler is corrected_cd_components["sampler"]
    assert module.knn_entropy_estimator is corrected_cd_components["knn_entropy_estimator"]
    assert module.lambda_cd == 1.0 # Check default value

def test_corrected_cd_training_step(corrected_cd_components):
    """
    A functional test to ensure the training step runs and produces a valid loss
    with a backward pass.
    """
    module = CorrectedCDLightningModule(**corrected_cd_components)

    # Create a dummy batch of data
    batch = torch.randn(10, 10) # Batch of 10, 10-dim samples

    # Run the training step
    loss = module.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.requires_grad

    # Test backward pass
    try:
        loss.backward()
    except Exception as e:
        pytest.fail(f"loss.backward() raised an exception: {e}")

    # Check that model parameters have gradients
    grad_sum = sum(p.grad.sum() for p in module.ebm_model.parameters() if p.grad is not None)
    assert grad_sum != 0