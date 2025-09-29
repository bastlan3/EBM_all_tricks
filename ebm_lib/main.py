import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import sys
import os

# Import components from our library
from ebm_lib.ebm import EBM
from ebm_lib.lightning_module import EBMLightningModule
from ebm_lib.samplers.langevin import ReplayBufferLangevinSampler
from ebm_lib.regularizers.energy import L2EnergyRegularizer
from ebm_lib.regularizers.gradient import GradientPenaltyRegularizer
from ebm_lib.regularizers.score_matching import DenoisingScoreMatchingRegularizer
from ebm_lib.regularizers.spectral_norm import add_spectral_norm
from ebm_lib.pretraining.data import HeuristicPretrainingDataModule
from ebm_lib.pretraining.lightning_module import HeuristicPretrainer
from ebm_lib.heuristics.experts import LaplacianVarianceExpert, HighFrequencyEnergyExpert
from ebm_lib.heuristics.scorer import MixtureOfExpertsScorer

class SimpleCNN(nn.Module):
    """A small CNN that can handle variable input sizes due to AdaptiveAvgPool2d."""
    def __init__(self, input_shape):
        super().__init__()
        c, h, w = input_shape
        self.main = nn.Sequential(
            nn.Conv2d(c, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.main(x).squeeze(-1)

def run_workflow(ckpt_path=None):
    """
    Demonstrates a full, two-stage workflow:
    1. Heuristic pre-training to initialize the EBM.
    2. Contrastive Divergence fine-tuning.
    """
    print("--- Starting Full EBM Training Workflow ---")

    # --- Shared Configuration ---
    image_shape = (3, 32, 32) # CIFAR10
    optimizer_config = {'name': 'Adam', 'lr': 1e-4}

    # --- Initialize Shared Components ---
    # Create two separate network instances for the online and target models
    online_network = SimpleCNN(input_shape=image_shape)
    add_spectral_norm(online_network)
    ebm_model = EBM(online_network)

    target_network = SimpleCNN(input_shape=image_shape)
    add_spectral_norm(target_network)
    target_ebm_model = EBM(target_network)

    # Initialize target model weights to be the same as the online model
    target_ebm_model.load_state_dict(ebm_model.state_dict())

    # --- Stage 1: Heuristic Pre-training ---
    print("\n--- STAGE 1: Heuristic Pre-training ---")

    # 1a. Set up the Data Pipeline for pre-training
    pretrain_data_module = HeuristicPretrainingDataModule(batch_size=128)

    # 1b. Set up the Heuristic Scorer
    experts = [LaplacianVarianceExpert(), HighFrequencyEnergyExpert()]
    heuristic_scorer = MixtureOfExpertsScorer(experts=experts)

    # 1c. Set up the Pre-training Lightning Module
    pretrainer_module = HeuristicPretrainer(
        ebm_model=ebm_model,
        heuristic_scorer=heuristic_scorer,
        optimizer_config=optimizer_config
    )

    # 1d. Configure and run the pre-training Trainer
    pretrain_trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=2,
        limit_train_batches=50, # Limit for a quick demonstration
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True
    )
    pretrain_trainer.fit(model=pretrainer_module, datamodule=pretrain_data_module)
    print("--- Heuristic Pre-training Finished ---")

    # --- Stage 2: Contrastive Divergence Fine-tuning ---
    print("\n--- STAGE 2: Contrastive Divergence Fine-tuning ---")

    # The `ebm_model` object has now been pre-trained. We can use it directly.

    # 2a. Set up the Sampler for CD training
    sampler = ReplayBufferLangevinSampler(config={'buffer_size': 10000})

    # 2b. Set up Regularizers
    regularizers = [
        L2EnergyRegularizer(config={'lambda_e': 0.01}),
        DenoisingScoreMatchingRegularizer(config={'lambda_dsm': 0.1, 'sigma': 0.01})
    ]

    # 2c. Set up the main EBM Lightning Module
    finetune_module = EBMLightningModule(
        ebm_model=ebm_model,
        target_ebm_model=target_ebm_model,
        sampler=sampler,
        optimizer_config=optimizer_config,
        regularizers=regularizers,
        ema_decay=0.999 # Use EMA for stable fine-tuning
    )

    # 2d. Configure and run the fine-tuning Trainer
    finetune_trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=2,
        limit_train_batches=50,
        enable_checkpointing=True, # Enable checkpoints for the final model
        default_root_dir="ebm_checkpoints",
        logger=False,
        enable_progress_bar=True
    )

    # Use the same data module, but the LightningModule will now use the samples
    # for CD training instead of heuristic regression.
    finetune_trainer.fit(model=finetune_module, datamodule=pretrain_data_module, ckpt_path=ckpt_path)
    print("--- CD Fine-tuning Finished ---")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run the EBM training workflow.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume fine-tuning from."
    )
    args = parser.parse_args()

    run_workflow(ckpt_path=args.ckpt_path)

    # --- Example for Latent Space EBM Training (Commented Out) ---
    # This shows how you would set up the latent space training using a
    # powerful, pre-trained VAE from Stable Diffusion.
    """
    from diffusers import AutoencoderKL
    from ebm_lib.models.latent_ebm import LatentEBM
    from ebm_lib.latent_ebm_module import LatentEBMLightningModule

    # 1. Load the pre-trained Stable Diffusion VAE
    # This will download the model from the Hugging Face Hub on first run.
    print("Loading pre-trained Stable Diffusion VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")

    # The VAE's encode method returns a distribution. We need a simple encoder
    # that returns the mean of that distribution as a latent vector.
    class VAEEncoderWrapper(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def encode(self, x):
            # The VAE returns a distribution object, we take the mean for a deterministic latent vector
            return self.vae.encode(x).latent_dist.mean

    encoder = VAEEncoderWrapper(vae)

    # 2. Define an EBM that operates on the latent space
    # The latent space of this VAE is 4 channels, H/8, W/8
    latent_dim = 4 * (image_shape[1] // 8) * (image_shape[2] // 8)
    latent_energy_net = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    # 3. Create the LatentEBM wrapper
    latent_ebm = LatentEBM(latent_energy_net, encoder)

    # 4. Use the LatentEBMLightningModule for training
    # The sampler will now operate on the latent space
    latent_sampler = LangevinSampler(config={'k_steps': 20})
    latent_module = LatentEBMLightningModule(
        ebm_model=latent_ebm,
        sampler=latent_sampler,
        optimizer_config={'name': 'Adam', 'lr': 1e-4}
    )

    # 5. Train on a standard image dataloader, resizing to the VAE's expected input size
    data_module = HeuristicPretrainingDataModule(batch_size=32, image_size=256)
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model=latent_module, datamodule=data_module)
    """