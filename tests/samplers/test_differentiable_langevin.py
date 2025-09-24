import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM
from ebm_lib.samplers.differentiable_langevin import DifferentiableLangevinSampler

class LinearEBM(EBM):
    """A simple EBM with a single learnable parameter."""
    def __init__(self):
        super().__init__(network=nn.Linear(10, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def test_differentiable_sampler_retains_grad():
    """
    Tests the key property of the DifferentiableLangevinSampler: that the
    output samples remain attached to the computation graph.
    """
    sampler = DifferentiableLangevinSampler(config={'k_steps': 5})
    ebm_model = LinearEBM()

    initial_samples = torch.randn(4, 10)

    # Generate samples
    final_samples = sampler.sample(ebm_model, 4, (10,), initial_states=initial_samples)

    # The output should have a gradient function
    assert final_samples.grad_fn is not None

def test_gradient_flow_through_sampler():
    """
    Tests that gradients flow from a loss computed on the negative samples
    back through the sampler to the model's parameters.
    """
    sampler = DifferentiableLangevinSampler(config={'k_steps': 5})
    ebm_model = LinearEBM()

    # Ensure the model's parameter has no gradient initially
    assert ebm_model.network.weight.grad is None

    # Generate samples
    final_samples = sampler.sample(ebm_model, 4, (10,))

    # Compute a loss based on the output samples (e.g., the sampler energy loss)
    loss = ebm_model(final_samples).mean()

    # Backpropagate the loss
    loss.backward()

    # The model's parameter should now have a gradient
    assert ebm_model.network.weight.grad is not None
    assert torch.all(ebm_model.network.weight.grad != 0)