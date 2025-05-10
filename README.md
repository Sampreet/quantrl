# QuantRL: Quantum Control using Reinforcement Learning

![Latest Version](https://img.shields.io/badge/version-0.0.9-red?style=for-the-badge)

> A library of modules to interface deterministic and stochastic quantum models for reinforcement learning.

### Key Features!

* Quickly interface environments for Reinforcement Learning using Stable-Baselines3.
* Run multiple environments in parallel using vectorized inheritable classes.
* Support for deterministic and stochastic linear environments.
* Live visualization and learning curves.

### What's New in v0.0.x

* Initialize environments with any of the three backends: NumPy, PyTorch and JAX.
* Solve IVPs for the popular libraries `TorchDiffEq` and `Diffrax`.
* Asynchronous cache-dump to speed up environment evolution.
* Updated stochastic environment for fast Wiener processes.
* Added support for measurement noise in observations.
* Callback to save best mean reward.

## Installation

[QuantRL](https://github.com/sampreet/quantrl) requires `Python 3.12+`, preferably installed via the [Anaconda distribution](https://www.anaconda.com/download).
The toolbox primarily relies on `gymnasium` (for single environments) and `stable-baselines3` (for vectorized environments).
All of its dependencies can be installed using:

```bash
python -m pip install numpy scipy matplotlib tqdm pillow pandas gymnasium stable-baselines3
```

Additionally, to avail the PyTorch or JAX backends, the latest version of these framework (for both CPU and GPU) should be installed (preferably in different `conda` environments) using in their official documentations: [PyTorch docs](https://pytorch.org/get-started/locally/) and [JAX docs](https://jax.readthedocs.io/en/latest/installation.html).
After successful installation, the corresponding libraries (`torchdiffeq` for PyTorch and `diffrax` for JAX) can be installed using PIP as:

```bash
pip install torchdiffeq
```

or,

```bash
pip install jax
pip install diffrax
```

To install JAX with GPU support, use `jax[cuda12]`.

***Note: JAX-GPU support for Windows and MacOS is still limited but it runs well in WSL2.***

Finally, to install the latest version of `quantrl`, execute:

```bash
pip install git+https://github.com/sampreet/quantrl.git
```
