import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import sys
import os

# Use relative imports, as this script is part of the ebm_lib package
from .ebm import EBM
from .lightning_module import EBMLightningModule
from .samplers.langevin import ReplayBufferLangevinSampler
from .regularizers.energy import L2EnergyRegularizer
from .regularizers.gradient import GradientPenaltyRegularizer
from .regularizers.spectral_norm import add_spectral_norm

# 1. Define a simple EBM network (e.g., a small CNN for image data)
class SimpleCNN(nn.Module):
    """A small CNN to be used as the energy function."""
    def __init__(self, input_shape):
        super().__init__()
        c, h, w = input_shape
        self.main = nn.Sequential(
            nn.Conv2d(c, 16, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 4, 2, 1), # -> h/2, w/2
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), # -> h/4, w/4
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * (h // 4) * (w // 4), 1) # Output a single energy value
        )

    def forward(self, x):
        # The output of the network is the scalar energy.
        # Squeeze the last dimension to get a tensor of shape (batch_size,).
        return self.main(x).squeeze(-1)

def run_example():
    """
    An example script demonstrating how to configure and run EBM training.
    """
    print("--- Starting EBM Training Library Example ---")

    # --- Configuration ---
    image_shape = (1, 32, 32)
    batch_size = 32

    # --- 1. Create a dummy dataset ---
    # Create 10 batches of random data to simulate a small dataset.
    print("Creating dummy dataset...")
    dummy_data = torch.randn(batch_size * 10, *image_shape)
    dataset = TensorDataset(dummy_data)
    # Set num_workers based on platform for compatibility
    num_workers = 4 if sys.platform != 'win32' else 0
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # --- 2. Define the EBM network and model ---
    print("Initializing model components...")
    # The user is responsible for creating the network architecture.
    ebm_network = SimpleCNN(input_shape=image_shape)

    # (Optional) Apply spectral norm for training stability.
    add_spectral_norm(ebm_network)

    # Wrap the network in our EBM class.
    # PyTorch Lightning will handle moving this to the correct device.
    ebm_model = EBM(ebm_network)

    # --- 3. Configure and select the SAMPLER ---
    sampler_config = {
        'k_steps': 40,
        'step_size': 1e-4,
        'noise_scale': 0.005,
        'buffer_size': 10000,
        'replay_probability': 0.95
    }
    sampler = ReplayBufferLangevinSampler(sampler_config)

    # --- 4. Configure and select a LIST of REGULARIZERS ---
    regularizers_config = {
        'l2_energy': {'lambda_e': 0.1},
        'grad_penalty': {'lambda_gp': 1.0}
    }
    regularizers = [
        L2EnergyRegularizer(regularizers_config['l2_energy']),
        GradientPenaltyRegularizer(regularizers_config['grad_penalty'])
    ]

    # --- 5. Configure Optimizer and instantiate the Lightning Module ---
    optimizer_config = {'name': 'Adam', 'lr': 1e-4}

    ebm_lightning_module = EBMLightningModule(
        ebm_model=ebm_model,
        sampler=sampler,
        optimizer_config=optimizer_config,
        regularizers=regularizers
    )

    # --- 6. Configure the PyTorch Lightning TRAINER ---
    # This is the only part that needs to change for different hardware.
    # We'll run on CPU for a few steps for this example.
    print("Configuring PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        accelerator='auto', # Automatically selects CPU, GPU, or other accelerator
        max_epochs=1,
        # Limit steps for a quick demonstration
        limit_train_batches=10,
        enable_checkpointing=False,
        logger=False, # Disable logging for this simple example
        enable_progress_bar=True
    )

    print("\n--- Starting Training ---")
    # --- 7. Start training! ---
    trainer.fit(model=ebm_lightning_module, train_dataloaders=data_loader)
    print("--- Training Finished ---")

if __name__ == '__main__':
    # Add a check for CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s).")
    else:
        print("CUDA not available. Running on CPU.")

    run_example()
