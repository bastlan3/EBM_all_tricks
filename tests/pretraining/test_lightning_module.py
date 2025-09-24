import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM
from ebm_lib.heuristics.scorer import MixtureOfExpertsScorer
from ebm_lib.heuristics.experts import HeuristicExpert
from ebm_lib.pretraining.lightning_module import HeuristicPretrainer

class MockScorer(MixtureOfExpertsScorer):
    """A mock scorer that returns a fixed score."""
    def __init__(self):
        super().__init__(experts=[]) # No real experts needed

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        # Return a fixed target score for each image in the batch
        return torch.full((image_batch.shape[0],), 0.5, device=image_batch.device)

@pytest.fixture
def pretrainer_components():
    """Provides components needed to initialize the HeuristicPretrainer."""
    ebm_network = nn.Linear(10, 1) # A simple network for the EBM
    ebm_model = EBM(ebm_network)
    heuristic_scorer = MockScorer()
    optimizer_config = {'name': 'SGD', 'lr': 0.1}

    return {
        "ebm_model": ebm_model,
        "heuristic_scorer": heuristic_scorer,
        "optimizer_config": optimizer_config,
    }

def test_pretrainer_init(pretrainer_components):
    """Tests that the HeuristicPretrainer initializes correctly."""
    pretrainer = HeuristicPretrainer(**pretrainer_components)
    assert pretrainer.ebm_model is pretrainer_components["ebm_model"]
    assert pretrainer.heuristic_scorer is pretrainer_components["heuristic_scorer"]
    assert pretrainer.optimizer_config["name"] == "SGD"

def test_pretrainer_configure_optimizers(pretrainer_components):
    """Tests that the optimizer is configured correctly."""
    pretrainer = HeuristicPretrainer(**pretrainer_components)
    optimizer = pretrainer.configure_optimizers()

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults['lr'] == 0.1

def test_pretrainer_training_step(pretrainer_components):
    """
    Tests that a single training step executes and produces a valid loss.
    """
    pretrainer = HeuristicPretrainer(**pretrainer_components)

    # Create a dummy batch of images
    image_batch = torch.randn(4, 1, 10) # 4 samples, 1 channel, 10 features

    loss = pretrainer.training_step(image_batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 # Loss must be a scalar
    assert loss.requires_grad

    # Check the loss value
    # EBM energy will be some value, target score is 0.5. Loss should be > 0.
    assert loss.item() > 0

    # Test backward pass
    try:
        loss.backward()
    except Exception as e:
        pytest.fail(f"loss.backward() raised an exception: {e}")

    # Check that the EBM's parameters have gradients
    grad_sum = sum(p.grad.sum() for p in pretrainer.ebm_model.parameters() if p.grad is not None)
    assert grad_sum != 0
