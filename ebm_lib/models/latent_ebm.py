import torch
import torch.nn as nn

from ..ebm import EBM
from .autoencoder import SimpleConvAutoencoder

class LatentEBM(EBM):
    """
    An EBM wrapper that operates on the latent space of a frozen encoder.

    This module takes an image as input, passes it through a pre-trained,
    frozen encoder to get a latent representation, and then computes the
    energy of that latent vector using a separate EBM network.
    """
    def __init__(self, latent_energy_net: nn.Module, encoder: nn.Module):
        """
        Args:
            latent_energy_net (nn.Module): The network that computes energy from a latent vector.
            encoder (nn.Module): A pre-trained, frozen model with an `encode` method
                                 (e.g., a VAE from `diffusers` or a custom autoencoder).
        """
        # The 'network' of the parent EBM class is the latent energy network.
        super().__init__(network=latent_energy_net)

        if not hasattr(encoder, 'encode'):
            raise ValueError("The provided encoder must have an 'encode' method.")

        self.encoder = encoder
        # Freeze the encoder's parameters
        self.encoder.requires_grad_(False)

    def forward(self, x):
        """
        Computes the energy of an input. The method intelligently handles whether
        the input is an image or a latent vector based on its dimensionality.

        Args:
            x (torch.Tensor): A batch of images (4D tensor) or latent vectors (2D tensor).

        Returns:
            torch.Tensor: The energy of the latent representation.
        """
        if x.dim() == 4:
            # Input is an image batch, encode it first.
            with torch.no_grad():
                z = self.encoder.encode(x)
        elif x.dim() == 2:
            # Input is already a latent vector batch.
            z = x
        else:
            raise ValueError(f"LatentEBM expects a 4D image tensor or 2D latent tensor, but got {x.dim()}D")

        # Compute the energy of the latent vector using the EBM's network
        return self.network(z)