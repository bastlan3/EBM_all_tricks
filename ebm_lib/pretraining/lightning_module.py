import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Dict, Any

from ..ebm import EBM
from ..heuristics.scorer import MixtureOfExpertsScorer

class HeuristicPretrainer(pl.LightningModule):
    """
    A PyTorch Lightning module for pre-training an EBM by teaching it to
    mimic the output of a deterministic, heuristic-based image scorer.

    The training is a simple regression task where the EBM's energy output
    is trained to match the score from the heuristic scorer for a given
    augmented image. This allows the EBM to learn a reasonable energy
    surface before fine-tuning with contrastive divergence.
    """
    def __init__(self,
                 ebm_model: EBM,
                 heuristic_scorer: MixtureOfExpertsScorer,
                 optimizer_config: Dict[str, Any]):
        """
        Args:
            ebm_model (EBM): The energy-based model to be pre-trained.
            heuristic_scorer (MixtureOfExpertsScorer): The scorer that provides the
                                                       target regression values.
            optimizer_config (Dict[str, Any]): Configuration for the optimizer,
                                               e.g., {'name': 'Adam', 'lr': 1e-4}.
        """
        super().__init__()
        self.ebm_model = ebm_model
        self.heuristic_scorer = heuristic_scorer
        self.optimizer_config = optimizer_config
        # Save hyperparameters, ignoring the model and scorer modules
        self.save_hyperparameters(ignore=['ebm_model', 'heuristic_scorer'])

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Performs a single pre-training step on a batch of augmented images.

        Args:
            batch (torch.Tensor): A tensor of augmented images from the dataloader.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The calculated regression loss.
        """
        # The batch directly contains the augmented images
        augmented_images = batch

        # 1. Calculate the target scores from the heuristic experts.
        # Gradients are not needed for the scorer, as it acts as a fixed target.
        with torch.no_grad():
            heuristic_scores = self.heuristic_scorer(augmented_images).detach()

        # 2. Calculate the EBM's energy for the same images.
        # Squeeze the output to ensure it's (B,) to match the target shape.
        ebm_energies = self.ebm_model(augmented_images).squeeze()

        # 3. Compute the regression loss (Mean Squared Error).
        loss = F.mse_loss(ebm_energies, heuristic_scores)

        self.log('pretrain_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Sets up the optimizer based on the configuration dictionary.

        Raises:
            ValueError: If the specified optimizer is not found in `torch.optim`.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer_name = self.optimizer_config.get('name', 'Adam')
        lr = self.optimizer_config.get('lr', 1e-4)

        optimizer_class = getattr(torch.optim, optimizer_name, None)
        if optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")

        # self.parameters() will correctly include the parameters of self.ebm_model
        optimizer = optimizer_class(self.parameters(), lr=lr)

        return optimizer
