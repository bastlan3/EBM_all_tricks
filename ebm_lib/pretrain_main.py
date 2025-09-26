import torch
import pytorch_lightning as pl

# Import components from our library
from ebm_lib.ebm import EBM
from ebm_lib.main import SimpleCNN, ResNet18EBM# Reuse the CNN from the main example
from ebm_lib.heuristics.experts import LaplacianVarianceExpert, HighFrequencyEnergyExpert
from ebm_lib.heuristics.scorer import MixtureOfExpertsScorer
from ebm_lib.pretraining.data import HeuristicPretrainingDataModule
from ebm_lib.pretraining.lightning_module import HeuristicPretrainer

def run_pretraining_example():
    """
    An example script demonstrating how to run the heuristic-based EBM pre-training.
    """
    print("--- Starting EBM Heuristic Pre-training Example ---")

    # --- 1. Set up the Data Pipeline ---
    print("Initializing data module...")
    # This will download CIFAR10 and apply heavy augmentations
    data_module = HeuristicPretrainingDataModule(batch_size=64)

    # --- 2. Set up the EBM ---
    print("Initializing EBM...")
    ebm_network = ResNet18EBM(input_shape=(3, 32, 32)) # CIFAR10 has 3 channels
    ebm_model = EBM(ebm_network)

    # --- 3. Set up the Heuristic Scorer ---
    print("Initializing heuristic scorer...")
    # Create a list of the expert modules to use
    experts = [
        LaplacianVarianceExpert(),
        HighFrequencyEnergyExpert()
    ]
    # Combine them into the Mixture-of-Experts scorer
    heuristic_scorer = MixtureOfExpertsScorer(experts=experts)

    # --- 4. Set up the Pre-training Lightning Module ---
    print("Initializing Pre-training Lightning Module...")
    optimizer_config = {'name': 'Adam', 'lr': 1e-4}

    pretrainer_module = HeuristicPretrainer(
        ebm_model=ebm_model,
        heuristic_scorer=heuristic_scorer,
        optimizer_config=optimizer_config
    )

    # --- 5. Configure and run the Trainer ---
    print("Configuring PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        accelerator='auto',
        max_epochs=10,
        # Limit steps for a quick demonstration run
        limit_train_batches=2000,
        enable_checkpointing=True,
        logger=False,
        enable_progress_bar=True
    )

    print("\n--- Starting Pre-training ---")
    trainer.fit(model=pretrainer_module, datamodule=data_module)
    print("--- Pre-training Finished ---")

if __name__ == '__main__':
    run_pretraining_example()
