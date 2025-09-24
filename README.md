# A Modular Research Library for Energy-Based Models

This project provides an extensive, modular, and combinatorial Python library for training Energy-Based Models (EBMs). It is built on PyTorch and PyTorch Lightning to be scalable and support training on CPU, single GPU, multiple GPUs (DDP), and TPUs seamlessly.

The core design principle is to allow researchers to easily mix and match training components (samplers, regularizers, and advanced objectives) to evaluate their combined effects.

## Installation

First, ensure you have a recent version of PyTorch installed that matches your hardware (e.g., CUDA version). Then, install the library and its other dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install dependencies
pip install -e .
pip install pytest # For running tests
```

A `requirements.txt` file is also provided for reproducibility.

## Core Concepts

The library is built around a few key components:

- **`EBM`**: A simple wrapper around a `torch.nn.Module` that defines the energy function.
- **`Sampler`**: A base class for MCMC methods used to generate negative samples.
- **`Regularizer`**: A base class for techniques that add a penalty to the loss function to stabilize training.
- **`EBMLightningModule`**: The main `pytorch_lightning.LightningModule` that orchestrates standard contrastive divergence training.

---

## A Guide to EBM Training Techniques

This guide explains the theory and implementation of the various state-of-the-art EBM training techniques included in this library.

### Category 1: The Core Objective (Contrastive Divergence)

Standard EBM training is performed with the `EBMLightningModule`. It uses a **short-run MCMC** sampler (like `LangevinSampler`) to generate negative samples and minimizes the **Contrastive Divergence (CD)** loss: `E(positive) - E(negative)`.

```python
# Basic Usage (see ebm_lib/main.py for a full example)
from ebm_lib.lightning_module import EBMLightningModule

# 1. Create your EBM, Sampler, and Regularizers
ebm_model = ...
sampler = ...
regularizers = [...]

# 2. Initialize the Lightning Module
ebm_lightning_module = EBMLightningModule(
    ebm_model=ebm_model,
    sampler=sampler,
    regularizers=regularizers,
    optimizer_config={'name': 'Adam', 'lr': 1e-4}
)

# 3. Train with the PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model=ebm_lightning_module, train_dataloaders=...)
```

### Category 2: Improving MCMC Sampling

#### Technique: Persistent CD (Replay Buffer)
- **Theory**: Instead of starting MCMC chains from random noise every time, we store the samples from the end of the last training step in a buffer and use them to initialize the chains for the next step. This is vastly more efficient.
- **Implementation**: Use the `ReplayBufferLangevinSampler`. A small fraction of samples are periodically re-initialized from noise to prevent mode collapse, controlled by `replay_probability`.

```python
from ebm_lib.samplers.langevin import ReplayBufferLangevinSampler

sampler_config = {
    'k_steps': 40,
    'buffer_size': 10000,
    'replay_probability': 0.95 # 5% of chains are re-initialized from noise
}
sampler = ReplayBufferLangevinSampler(sampler_config)
```

#### Technique: Data Augmentation as an MCMC Move
- **Theory**: Applying a data augmentation (e.g., a flip or rotation) inside the MCMC chain can be seen as a powerful "jump" proposal that helps the sampler explore different modes of the data distribution.
- **Implementation**: Use the `AugmentedLangevinSampler` and pass it a `torchvision` transform.

```python
from ebm_lib.samplers.augmented_langevin import AugmentedLangevinSampler
import torchvision.transforms.v2 as transforms

# Define a transform to be used as an MCMC move
mcmc_transform = transforms.RandomHorizontalFlip(p=1.0)

sampler_config = {'k_steps': 20, 'augment_interval': 5}
sampler = AugmentedLangevinSampler(sampler_config, transform=mcmc_transform)
```

### Category 3: Stabilization and Regularization

#### Technique: Spectral Normalization
- **Theory**: Constrains the Lipschitz constant of the EBM network by normalizing the weights of its layers. This is crucial for preventing the sampler from becoming unstable due to exploding gradients.
- **Implementation**: Use the `add_spectral_norm` utility on your network *before* passing it to the `EBM` wrapper.

```python
from ebm_lib.regularizers.spectral_norm import add_spectral_norm

ebm_network = SimpleCNN(...)
add_spectral_norm(ebm_network) # Apply in-place
ebm_model = EBM(ebm_network)
```

#### Technique: L2 Regularization on Energy
- **Theory**: Adds a penalty proportional to `E(x)²` to the loss. This keeps the absolute energy values from drifting and causing numerical instability.
- **Implementation**: Add the `L2EnergyRegularizer` to the list of regularizers.

```python
from ebm_lib.regularizers.energy import L2EnergyRegularizer
regularizers = [L2EnergyRegularizer(config={'lambda_e': 0.1})]
```

#### Technique: Gradient Clipping
- **Theory**: A last-resort safety measure to prevent outlier batches from causing destructively large gradient updates.
- **Implementation**: This is a built-in feature of the `pytorch_lightning.Trainer`.

```python
# Clip the gradient norm to a maximum value of 1.0
trainer = pl.Trainer(gradient_clip_val=1.0)
```

### Category 4: Advanced Architectures and Objectives

#### Technique: Multi-Scale Energy Function
- **Theory**: Defines the total energy as a sum of energies computed on multiple resolutions of the input image. This forces the network to learn features that are coherent across scales, improving global structure.
- **Implementation**: Set `num_scales` when initializing the `EBMLightningModule`. Your EBM network must be able to handle variable-sized inputs (e.g., by using an `AdaptiveAvgPool2d` layer before the final linear layer).

```python
# The network in `ebm_model` must support variable input sizes
module = EBMLightningModule(..., num_scales=3)
```

#### Technique: Corrected CD Objective (Du et al., 2021)
- **Theory**: Corrects a known bias in the standard CD gradient by including two extra terms: one that backpropagates through the MCMC sampler and another that maximizes the sampler's entropy. This leads to a more stable and accurate approximation of the true maximum likelihood gradient.
- **Implementation**: This advanced objective requires a new set of components.
    1. A **differentiable sampler** (`DifferentiableLangevinSampler`).
    2. An **entropy estimator** (`KnnEntropyEstimator`).
    3. The **`CorrectedCDLightningModule`**, which combines everything.

```python
# See ebm_lib/corrected_cd/ for the full implementation
from ebm_lib.corrected_cd.lightning_module import CorrectedCDLightningModule
from ebm_lib.samplers.differentiable_langevin import DifferentiableLangevinSampler
from ebm_lib.estimators.knn_entropy import KnnEntropyEstimator

differentiable_sampler = DifferentiableLangevinSampler(...)
knn_estimator = KnnEntropyEstimator(k=5)

# Use the dedicated Lightning module for this objective
corrected_cd_module = CorrectedCDLightningModule(
    ebm_model=ebm_model,
    sampler=differentiable_sampler,
    knn_entropy_estimator=knn_estimator,
    ...
)
```

### Category 5: Pre-training with Heuristics

This library also includes a novel pipeline for pre-training an EBM to predict a "common sense" noise score from a set of deterministic signal processing experts.

- **See the `Heuristic Pre-training` section and the `ebm_lib/pretrain_main.py` script for details.**

---

## Running Examples and Tests

- **Standard Training**: `python3 -m ebm_lib.main`
- **Heuristic Pre-training**: `python3 -m ebm_lib.pretrain_main`
- **Run All Tests**: `pytest`