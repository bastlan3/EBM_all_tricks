import torch
from typing import Any

from .lightning_module import EBMLightningModule
from .models.latent_ebm import LatentEBM

class LatentEBMLightningModule(EBMLightningModule):
    """
    A specialized LightningModule for training an EBM on the latent space
    of a frozen autoencoder.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.ebm_model, LatentEBM):
            raise TypeError("The ebm_model for LatentEBMLightningModule must be a LatentEBM instance.")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Performs a training step in the latent space.

        It first encodes the input images into latent vectors and then performs
        the standard contrastive divergence training on those vectors.
        """
        # 1. Unpack images from the batch
        images = batch[0] if isinstance(batch, (list, tuple)) else batch

        # 2. Encode images to get positive samples in the latent space
        with torch.no_grad():
            positive_samples = self.ebm_model.encoder.encode(images)

        # The rest of the logic is identical to the standard EBMLightningModule,
        # but it will operate on latent vectors instead of images.
        # We can achieve this by creating a "fake" batch of positive latent samples.
        latent_batch = (positive_samples,)

        return super().training_step(latent_batch, batch_idx)