import pytorch_lightning as pl
import torch
from typing import Dict, Any

from ..ebm import EBM
from ..samplers.differentiable_langevin import DifferentiableLangevinSampler
from ..estimators.knn_entropy import KnnEntropyEstimator

class CorrectedCDLightningModule(pl.LightningModule):
    """
    A LightningModule that implements the "Corrected" Contrastive Divergence
    objective from Du et al. (2021). This objective corrects for the bias in
    the standard CD gradient by including terms for the sampler's energy and
    entropy.
    """
    def __init__(self,
                 ebm_model: EBM,
                 sampler: DifferentiableLangevinSampler,
                 knn_entropy_estimator: KnnEntropyEstimator,
                 optimizer_config: Dict[str, Any],
                 lambda_cd: float = 1.0,
                 lambda_energy: float = 1.0,
                 lambda_entropy: float = 0.1):
        """
        Args:
            ebm_model (EBM): The energy-based model.
            sampler (DifferentiableLangevinSampler): A sampler that allows for
                                                     backpropagation through its steps.
            knn_entropy_estimator (KnnEntropyEstimator): An estimator for the
                                                         sampler's entropy.
            optimizer_config (Dict[str, Any]): Configuration for the optimizer.
            lambda_cd (float): Weight for the standard CD loss.
            lambda_energy (float): Weight for the sampler energy loss.
            lambda_entropy (float): Weight for the sampler entropy loss.
        """
        super().__init__()
        self.ebm_model = ebm_model
        self.sampler = sampler
        self.knn_entropy_estimator = knn_entropy_estimator
        self.optimizer_config = optimizer_config

        self.lambda_cd = lambda_cd
        self.lambda_energy = lambda_energy
        self.lambda_entropy = lambda_entropy

        self.save_hyperparameters(ignore=['ebm_model', 'sampler', 'knn_entropy_estimator'])

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # 1. Unpack positive samples
        positive_samples = batch[0] if isinstance(batch, (list, tuple)) else batch

        # 2. Generate negative samples using the *differentiable* sampler.
        # The resulting `negative_samples` tensor is still attached to the graph.
        negative_samples = self.sampler.sample(
            self.ebm_model,
            n_samples=positive_samples.shape[0],
            sample_shape=positive_samples.shape[1:]
        )

        # 3. Calculate energies
        positive_energy = self.ebm_model(positive_samples).squeeze()
        negative_energy = self.ebm_model(negative_samples).squeeze()

        # --- Corrected CD Loss Components ---

        # Component 1: Standard CD Loss
        cd_loss = positive_energy.mean() - negative_energy.mean()
        self.log('train_loss/cd_loss', cd_loss)

        # Component 2: Sampler Energy Loss (the "missing" gradient term)
        # By backpropagating through this loss, we send gradients through the
        # MCMC chain, correcting the objective.
        energy_loss = negative_energy.mean()
        self.log('train_loss/energy_loss', energy_loss)

        # Component 3: Sampler Entropy Loss
        # We want to maximize the entropy of the sampler's distribution,
        # which is equivalent to minimizing the loss from our estimator.
        entropy_loss = self.knn_entropy_estimator(negative_samples)
        self.log('train_loss/entropy_loss', entropy_loss)

        # 4. Combine the losses
        total_loss = (self.lambda_cd * cd_loss +
                      self.lambda_energy * energy_loss +
                      self.lambda_entropy * entropy_loss)

        self.log('train_loss/total_loss', total_loss)

        return total_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_name = self.optimizer_config.get('name', 'Adam')
        lr = self.optimizer_config.get('lr', 1e-4)

        optimizer_class = getattr(torch.optim, optimizer_name, None)
        if optimizer_class is None:
            raise ValueError(f"Optimizer '{optimizer_name}' not found in torch.optim")

        optimizer = optimizer_class(self.parameters(), lr=lr)

        return optimizer