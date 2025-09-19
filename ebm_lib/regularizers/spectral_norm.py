import torch.nn as nn

def add_spectral_norm(model: nn.Module) -> nn.Module:
    """
    Recursively applies spectral normalization to all Conv2d and Linear layers
    in a given model. This is an in-place modification.

    Args:
        model (nn.Module): The model (or a submodule) to which spectral
                           normalization will be applied.

    Returns:
        nn.Module: The model with spectral normalization hooks applied.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.utils.spectral_norm(module)
    return model
