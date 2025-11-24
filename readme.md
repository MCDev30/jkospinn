# JKO-SPINN: Learning SDE models and identifying physical parameters with Score-Based PINNs and JKO

---

## Overview

**JKO-SPINN** is a modern implementation of an innovative algorithm for learning Stochastic Differential Equations (SDEs) from noisy trajectories. By combining the power of score-based neural networks, physical knowledge integration via Stochastic Differential Equations (SDEs), and advanced optimization techniques inspired by the Jordan-Kinderlehrer-Otto (JKO) scheme, this code enables joint inference of:

- **the unknown drift field (score)**
- **the physical parameters of the process**
- **from partially observed and noisy data**

The approach is based on solving a mathematically motivated variational problem, with, at the core of the model, a physics-informed neural network (PINN) that learns a score function adapting both to the data and to the underlying physics using differential operators (scores, divergences, Hutchinson trace, etc.).

---

## Main Features

- Generic SDE simulator (Euler-Maruyama)
- Support for Ornstein-Uhlenbeck and double-well processes
- Automatic generation of reference trajectories for evaluation
- Score-based neural network with Fourier features and Swish activation
- Hybrid loss combining:  
  - **Score Matching** (data-driven)
  - **Physics** (PINN, residual differential operator)
  - **L2 guidance on true parameters (optional)**
- Joint optimization of network & physical parameters
- Fully traceable results (losses, convergence curves, relative errors, etc.)
- Experiments to assess robustness (sparsity, ablations) and stability (multi-initialization)
- Scientific visualizations: curves, barplots, summary tables…

---

## User Guide

### 1. Prerequisites

- JAX (`jax`, `jaxlib`)
- NumPy, Matplotlib, Seaborn, tqdm
- optax
- scipy (for stats)

All code is GPU-compatible if JAX is available on your machine.

### 2. Code Structure

- `Config`: general configuration of hyperparameters and experiment settings
- `OUProcess` & `DoubleWellProcess`: toy SDE definitions
- `SDESimulator`: generic SDE simulation (Euler-Maruyama)
- `generate_data`: generation of synthetic, noisy datasets
- `score_network`, `init_network`: score-based PINN architecture with Fourier features
- Physics operators: drift, divergences, Hutchinson trace
- Loss functions: data (DSM), physics (PINN), parameter guidance
- Training loop: `train_jko_spinn`
- Result visualization and analysis

### 3. Running Experiments

Typical experiments:

- **EXP 1: Ornstein-Uhlenbeck Process**  
  *Learning the θ, μ, σ parameters of the OU process from noisy trajectories*

- **EXP 2: Double-Well Process**  
  *Inferring parameter α and noise of a double-well potential*

- **EXP 3: Comparative analysis & robustness**  
  *Comparison with alternative solvers*

Each experiment generates performance summaries: relative errors for each parameter, convergence, loss curves, and visualizations.

### 4. Results and Visualization

Dedicated visualization functions allow:

- Monitoring convergence of physical parameters
- Visualizing loss curves
- Summary tables and relative error barplots (%)
- Additional analyses: data sparsity, ablations on Hutchinson sample count, λ_physics, etc.

---

Parameters, data generation, and SDE choice are fully configurable via the `Config` class.

---

## Algorithmic Highlights

- Joint optimization of PINN & SDE physical parameters by gradient descent (AdamW/Optax)
- Score-based learning using *denoising score matching* (DSM)
- Imposing physics through a differential operator (JKO-PINN residual)
- Use of Hutchinson’s trace for efficient score divergence computation
- Stability: early stopping, history, multi-restarts

---

## Authors

Developed by:

**Charbel MAMLANKOU, Jamal ADETOLA & Wilfrid HOUEDANOU** .  
Corresponding author: charbelzeusmamlankou@gmail.com
