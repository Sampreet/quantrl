# QuantRL: Quantum Control using Reinforcement Learning

![Latest Version](https://img.shields.io/badge/version-0.0.10-red?style=for-the-badge)

> A backend-agnostic library of modules to interface
deterministic and stochastic quantum models for reinforcement learning.

### Key Features!

* Quickly interface environments with any of the three backends:
NumPy, PyTorch and JAX.
* Run multiple RL environments in parallel using
vectorized inheritable classes.
* Evolve deterministic and stochastic environments
with asynchronous saves.
* Visualize evolutions and plot learning curves seamlessly.

### What's New!

* Support for NumPy 2.x.x.
* ``'tsit5'`` solver in PyTorch.

For a complete list of changes, see [CHANGELOG.md](CHANGELOG.md).

## Installation

[QuantRL](https://github.com/sampreet/quantrl) requires `Python 3.12+`,
preferably installed via the
[Anaconda distribution](https://www.anaconda.com/download).
It's base dependencies can be installed using:

```bash
python -m pip install numpy scipy matplotlib tqdm rich pillow pandas
```

The default backend for the library uses vanilla NumPy and Scipy.
To avail the JAX or PyTorch backends, the latest version
of these framework (CPU or GPU) should be installed
(preferably in different `conda` environments)
using in their official documentations:
[JAX docs](https://jax.readthedocs.io/en/latest/installation.html) and
[PyTorch docs](https://pytorch.org/get-started/locally/).
After successful installation, the corresponding libraries
(`diffrax` for JAX and `torchdiffeq` for PyTorch) can be installed using PIP.

For the CPU versions, use:

```bash
python -m pip install torch torchdiffeq jax diffrax
```

For the GPU versions with CUDA 12 support, use:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install torchdiffeq "jax[cuda12]" diffrax
```

***Note: JAX-GPU support for Windows and MacOS
is still limited but it runs well in WSL2.***

QuantRL primarily relies on `gymnasium` (for single environments)
and `stable-baselines3` (for vectorized environments).

These can be installed using:

```bash
python -m pip install gymnasium stable-baselines3
```

Finally, to install the latest version of `quantrl`, execute:

```bash
pip install git+https://github.com/sampreet/quantrl.git
```
