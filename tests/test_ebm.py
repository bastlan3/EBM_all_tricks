import pytest
import torch
import torch.nn as nn

from ebm_lib.ebm import EBM

@pytest.fixture
def simple_network() -> nn.Module:
    """A simple network that maps a 10-dim vector to a scalar energy."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

def test_ebm_initialization(simple_network):
    """Tests that the EBM wraps the network correctly."""
    ebm = EBM(network=simple_network)
    assert ebm.network is simple_network
    # Test that the device property works
    assert ebm.device.type == 'cpu'

def test_ebm_forward_pass(simple_network):
    """Tests that the forward pass returns a tensor of the correct shape."""
    ebm = EBM(network=simple_network)
    # Batch of 5 samples, each 10-dimensional
    input_tensor = torch.randn(5, 10)

    # The network outputs (5, 1). The EBM forward pass should return this.
    output = ebm(input_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (5, 1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for device test")
def test_ebm_device_movement(simple_network):
    """Tests that the EBM and its device property move to CUDA correctly."""
    ebm = EBM(network=simple_network)
    device = torch.device("cuda:0")
    ebm.to(device)

    assert ebm.device.type == 'cuda'
    # Check that a parameter of the network is also on the correct device
    assert next(ebm.network.parameters()).device.type == 'cuda'
