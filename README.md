# A Modular EBM Research Library

This project provides an extensive, modular, and combinatorial Python library for training Energy-Based Models (EBMs). It is built on PyTorch and PyTorch Lightning to be scalable and support training on CPU, single GPU, multiple GPUs (DDP), and TPUs seamlessly.

The core design principle is to allow researchers to easily mix and match training components (samplers, regularizers) to evaluate their combined effects.

## Core Features

- **Built on PyTorch Lightning**: Abstracts away boilerplate code for device management, distributed training, and mixed-precision, allowing you to focus on the EBM logic.
- **Modular Components**:
    - **Samplers**: A suite of MCMC samplers to generate negative samples.
        - `LangevinSampler`
        - `ReplayBufferLangevinSampler` (Improves mixing by re-using past samples)
        - `MALASampler` (Metropolis-Adjusted Langevin Algorithm)
        - `HMCSampler` (Hamiltonian Monte Carlo)
        - `ParallelTemperingSampler` (Replica Exchange MCMC)
    - **Regularizers**: A set of regularizers to stabilize training and improve model performance.
        - `L2EnergyRegularizer`: Penalizes large energy values.
        - `GradientPenaltyRegularizer`: Encourages the model to be 1-Lipschitz.
        - `add_spectral_norm`: A utility function to apply spectral normalization to the EBM's network.
- **Fully Tested**: The library includes a comprehensive test suite using `pytest` to ensure the correctness of each component.

## Installation

To use the library and run the example, you need to install the required dependencies.

```bash
pip install torch pytorch-lightning pytest
```

## Usage

The main entry point for training is the `EBMLightningModule`. You configure your EBM network, sampler, and regularizers as separate components and pass them to the Lightning module. The `pytorch_lightning.Trainer` then handles all the hardware-specific training loops.

Below is an example of how to set up and run an experiment.

```python
# main.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

# Import all the components from our library
from ebm_lib.ebm import EBM
from ebm_lib.lightning_module import EBMLightningModule
from ebm_lib.samplers.langevin import ReplayBufferLangevinSampler
from ebm_lib.regularizers.energy import L2EnergyRegularizer
from ebm_lib.regularizers.gradient import GradientPenaltyRegularizer
from ebm_lib.regularizers.spectral_norm import add_spectral_norm

# 1. Define a simple EBM network
class SimpleCNN(nn.Module):
    # ... (network definition) ...

# 2. Define the EBM network and model
ebm_network = SimpleCNN(input_shape=(1, 32, 32))
add_spectral_norm(ebm_network) # Optional
ebm_model = EBM(ebm_network)

# 3. Configure the SAMPLER
sampler_config = {
    'k_steps': 40,
    'step_size': 1e-4,
    'noise_scale': 0.005,
    'buffer_size': 10000,
    'replay_probability': 0.95
}
sampler = ReplayBufferLangevinSampler(sampler_config)

# 4. Configure a LIST of REGULARIZERS
regularizers = [
    L2EnergyRegularizer(config={'lambda_e': 0.1}),
    GradientPenaltyRegularizer(config={'lambda_gp': 1.0})
]

# 5. Configure Optimizer and instantiate the Lightning Module
optimizer_config = {'name': 'Adam', 'lr': 1e-4}
ebm_lightning_module = EBMLightningModule(
    ebm_model=ebm_model,
    sampler=sampler,
    optimizer_config=optimizer_config,
    regularizers=regularizers
)

# 6. Configure the PyTorch Lightning TRAINER
# Run on 4 GPUs
# trainer = pl.Trainer(accelerator='gpu', devices=4, strategy='ddp', max_epochs=100)
# Run on CPU for debugging
trainer = pl.Trainer(accelerator='cpu', max_epochs=1, limit_train_batches=10)

# 7. Start training
# (Create a dummy DataLoader 'data_loader' for the example)
trainer.fit(model=ebm_lightning_module, train_dataloaders=data_loader)
```

## Running the Example

An executable example script is provided at `ebm_lib/main.py`. To run it, navigate to the project's root directory and execute it as a module:

```bash
python3 -m ebm_lib.main
```

## Running Tests

The project includes a full suite of unit tests. To run the tests, navigate to the project's root directory and run `pytest`:

```bash
pytest tests/
```

---

## Heuristic Pre-training

In addition to standard contrastive divergence training, this library provides a novel method for pre-training EBMs using a set of deterministic, heuristic-based "experts".

### Concept

The core idea is to first train the EBM to approximate a "classic" understanding of image quality or noise. Instead of learning from data alone, the EBM learns to solve a regression task: its energy output for an image should match a score produced by a `MixtureOfExpertsScorer`. This scorer uses traditional signal processing methods (like Laplacian variance and FFT analysis) to evaluate image properties.

This pre-training step can guide the EBM to learn a useful energy surface before being fine-tuned with more expensive MCMC-based methods.

### Components

- **`HeuristicExpert`**: Base class for any traditional image evaluation algorithm. Two are provided: `LaplacianVarianceExpert` and `HighFrequencyEnergyExpert`.
- **`MixtureOfExpertsScorer`**: Combines the (normalized) scores from multiple experts into a single target value.
- **`HeuristicPretrainingDataModule`**: A PyTorch Lightning DataModule that provides a stream of heavily augmented images (noise, blur, crops, etc.) to train on.
- **`HeuristicPretrainer`**: A LightningModule that orchestrates the regression task.

### Running the Pre-training Example

An example script is provided at `ebm_lib/pretrain_main.py`. To run it, navigate to the project's root directory and execute it as a module:

```bash
python3 -m ebm_lib.pretrain_main
```
