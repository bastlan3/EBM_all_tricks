import torch
import torch.nn as nn

class EBM(nn.Module):
    """
    A wrapper class for an energy-based model (EBM).
    The EBM is represented by a neural network that outputs a scalar energy value.
    """
    def __init__(self, network: nn.Module):
        """
        Args:
            network (nn.Module): The neural network that parameterizes the energy function.
        """
        super().__init__()
        self.network = network
        # Register a dummy buffer to easily and robustly get the device of the model.
        # This is important for models that might not have parameters.
        self.register_buffer('_dummy_buffer', torch.tensor(0.))

    @property
    def device(self) -> torch.device:
        """Returns the device the model is on."""
        return self._dummy_buffer.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the energy for a given input.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A tensor containing the scalar energy value for each input sample.
        """
        return self.network(x)
