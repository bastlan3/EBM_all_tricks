# EBM Library Hyperparameter Guide

This document provides a quick reference for the hyperparameters of the various components in this library, along with recommended starting values from the literature.

## Samplers

### `ebm_lib.langevin.LangevinSampler`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `k_steps` | `int` | Number of MCMC steps to run. | 20-60 for training, can be higher for sampling. |
| `step_size` | `float` | Step size (learning rate) for the Langevin dynamics update. | Varies greatly. Start with 1e-4 to 1e-5. |
| `noise_scale` | `float` | Scale of the Gaussian noise added at each step. | Typically small, e.g., 0.005 to 0.01. |

### `ebm_lib.langevin.MALASampler`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `k_steps` | `int` | Number of MCMC steps to run. | 20-60 for training, can be higher for sampling. |
| `step_size` | `float` | Step size (learning rate) for the Langevin proposal. | Varies greatly. Start with 1e-4 to 1e-5. |
| `noise_scale` | `float` | Scale of the Gaussian noise for the Langevin proposal. | Typically small, e.g., 0.005 to 0.01. |

### `ebm_lib.langevin.ReplayBufferLangevinSampler`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `k_steps` | `int` | Number of MCMC steps to run. | 20-60 for training, can be higher for sampling. |
| `step_size` | `float` | Step size (learning rate) for the Langevin dynamics update. | Varies greatly. Start with 1e-4 to 1e-5. |
| `noise_scale` | `float` | Scale of the Gaussian noise added at each step. | Typically small, e.g., 0.005 to 0.01. |
| `buffer_size` | `int` | Maximum number of samples to store in the replay buffer. | 10000 |
| `replay_probability` | `float` | Probability of re-initializing a chain from the buffer vs. from noise. | 0.95 (i.e., 5% refresh rate from noise) |

### `ebm_lib.differentiable_langevin.DifferentiableLangevinSampler`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `k_steps` | `int` | Number of MCMC steps to run. Gradients will flow back through all steps. | 5-20. Fewer steps are common due to computational cost. |
| `step_size` | `float` | Step size for the Langevin dynamics update. | Varies greatly. Start with 1e-4 to 1e-5. |
| `noise_scale` | `float` | Scale of the Gaussian noise added at each step. | Typically small, e.g., 0.005 to 0.01. |

### `ebm_lib.hmc.HMCSampler`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `n_leapfrog_steps` | `int` | Number of leapfrog steps to simulate Hamiltonian dynamics. | 10-20. More steps allow for more distant proposals. |
| `step_size` | `float` | Step size (epsilon) for the leapfrog integrator. | Highly sensitive. Requires tuning. Start small, e.g., 0.05-0.1. |

### `ebm_lib.parallel_tempering.ParallelTemperingSampler`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `n_chains` | `int` | Number of parallel chains (temperatures) to run. | 8-16 |
| `betas` | `Tuple[float] or None` | Tuple of inverse temperatures. If None, a linear schedule is used. | None (to use the default linear schedule) or a geometric schedule. |
| `swap_interval` | `int` | Number of MCMC steps between swap proposals. | 5-10 |
| `k_steps` | `int` | Number of internal Langevin steps per sample() call. | 10-20 |
| `step_size` | `float` | Step size for the internal Langevin sampler. | 1e-4 to 1e-5 |
| `noise_scale` | `float` | Noise scale for the internal Langevin sampler. | 0.005 |

### `ebm_lib.augmented_langevin.AugmentedLangevinSampler`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `k_steps` | `int` | Number of MCMC steps to run. | 20-60 for training, can be higher for sampling. |
| `step_size` | `float` | Step size (learning rate) for the Langevin dynamics update. | Varies greatly. Start with 1e-4 to 1e-5. |
| `noise_scale` | `float` | Scale of the Gaussian noise added at each step. | Typically small, e.g., 0.005 to 0.01. |
| `augment_interval` | `int` | Number of MCMC steps between applying the augmentation transform. | 5-10 |

## Regularizers

### `ebm_lib.energy.L2EnergyRegularizer`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `lambda_e` | `float` | Weight of the L2 energy penalty. | 0.1 to 1.0 |

### `ebm_lib.gradient.GradientPenaltyRegularizer`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `lambda_gp` | `float` | Weight of the gradient penalty. | 1.0 to 10.0 |
| `target` | `float` | The target value for the gradient norm. | 1.0 |

### `ebm_lib.score_matching.DenoisingScoreMatchingRegularizer`

| Hyperparameter | Type    | Description                                                   | Recommended Value(s)                               |
|----------------|---------|---------------------------------------------------------------|----------------------------------------------------|
| `lambda_dsm` | `float` | Weight of the Denoising Score Matching loss. | 1.0 is a common starting point. |
| `sigma` | `float` | Standard deviation of the Gaussian noise added to the data. | 0.01 to 0.1 |
