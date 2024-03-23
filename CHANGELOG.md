# Changelog

## 2024/03/23 - 00 - v0.0.7 - Updated Stochastic Environment
* Added ``iterate_i`` method to support single loops in `quantrl.backends` package.
* Updated ``quantrl.envs.stochastic.LinearEnv`` with support for JAX loops.
* Removed iterative solvers from `quantrl.solvers` package.
* Updated `README` and `setup`.

## 2024/03/22 - 00 - v0.0.7 - Updated FileIO
* Updated `quantrl.backends` package:
    * Renamed occurences of ``tensor`` to ``typed`` and ``array`` to ``numpy``.
    * Renamed ``lib`` attribute to ``library`` for the backend library.
    * Added ``numpy`` option to ``dtype_from_str`` method.
    * Added ``is_typed``, ``generator`` and ``integers`` methods.
* Updated `quantrl.envs` package:
    * Added ``data_idxs`` option for selective storage.
    * Renamed ``disk_cache_size`` to ``cache_dump_interval``.
    * Removed ``preserve_dtype`` option. All backend data-types are now preserved.
    * Updated default value of ``observation_space_range`` to ``[-1e12, 1e12]`` and added ``check_truncation`` method to check if observations are out of bounds..
    * Added ``t_ssz`` attribute for the actual step size.
    * Added dedicated variables for caching, saving and plotting in vectorized environments.
* Updated `quantrl.io.FileIO` class:
    * Renamed ``disk_cache_size`` to ``cache_dump_interval``.
    * Removed option to update cache in parts and added ``dump_part`` method to directly dump part data of batches.
    * Added threading to support asynchronous cache dump.
* Templated iterative solvers to `quantrl.solvers` package.
* Updated `README` and `setup`.

## 2024/03/18 - 00 - v0.0.6 - PyTorch and JAX Backends
* Added `quantrl.backends` package:
    * Added abstract base class `BaseBackend` for different backends in `base` module.
    * Interfaced JAX, PyTorch and NumPy backends in `jax`, `torch` and `numpy` modules.
* Updated `quantrl.envs.base` module:
    * Created and documented abstract methods for `BaseEnv` class.
    * Renamed ``_step`` method to ``_update_observations``.
    * Interfaced backends in `BaseEnv`, `BaseGymEnv` and `BaseSB3Env` classes.
    * Added parameter ``preserve_dtype`` to preserve the interfaced tensor data-type in `BaseEnv` class.
    * Removed truncation information from ``step_async`` in `BaseSB3Env` class.
* Updated `quantrl.envs.deterministic` module:
    * Interfaced backends in `LinearizedHOEnv` and `LinearizedHOVecEnv` classes.
    * Added template methods for ``get_A``, ``get_D`` and ``get_mode_rates`` and updated documentation.
* Interfaced backends and removed MCQT support in `quantrl.envs.stochastic.LinearEnv` class.
* Added `base` solver module and interfaces for three different IVP solvers (`jax.DiffraxIVPSolver`, `torch.TorchDiffEqIVPSolver` and `numpy.SciPyIVPSolver` classes) in `quantrl.solvers` package.
* Updated `README` and `setup`.

## 2024/03/01 - 00 - v0.0.5 - Vectorized Environments
* Updated `quantrl.envs.base` module:
    * ``reset_Observations``, ``get_Properties`` and ``get_Reward`` methods of all classes are renamed to ``reset_observations``, ``get_properties`` and ``get_reward``, with ``reset_observations`` returning the initial observations.
    * Removed `n_trajectories` parameter and added `n_properties` parameter in all classes.
    * Added `file_prefix` parameter and `action_steps` attribute to `BaseEnv` class.
    * Renamed `file_prefix` attribute to `file_path_prefix` for the complete path.
    * Updated the workflow of `step` methods of all classes.
    * `plot_learning_curve` method of `BaseEnv` class takes `n_episodes` parameters for running average.
    * `replay_trajectories` and `close` methods of `BaseEnv` class require `n_episodes` parameter.
    * `close` method of `BaseEnv` class takes an additional `save_replay` parameter.
    * Added `save` parameter to `close` method of `BaseGymEnv` class.
    * Added `close` parameter to `evolve` method of `BaseGymEnv` class.
    * Added `BaseSB3Env` class to support Stable-Baselines3 vectorized environments.
* Updated `quantrl.envs.deterministic` module:
    * Removed `n_trajectories` and added `n_properties` parameters to `LinearizedHOEnv` class.
    * Removed `actions` parameter from `LinearizedHOEnv._step` method.
    * Added `LinearizedHOVecEnv` class to interface vectorized environments.
* Updated `quantrl.envs.stochastic.LinearEnv` class:
    * Removed `n_trajectories` and added `n_properties` parameters.
    * Removed `actions` parameter from `_step` and `_step_wiener` methods.
* Updated `quantrl.io.FileIO` class:
    * Added option to initialize with `data_shape` parameter.
    * Renamed `max_cache_size` parameter to `disk_cache_size`.
    * Added `dump_cache` parameter in `close` method.
* Added option to save plots in `save_dir` directory in `quantrl.plotters.TrajectoryPlotter` class.
* Updated `README`, `requirements` and `setup`.

## 2024/02/20 - 00 - v0.0.4 - Implemented kwargs
* Updated `quantrl.envs.base.BaseGymEnv` class:
    * Implemented keyword arguments in initialization.
    * Properties are saved if the child method exists.
* Minor fixes to `quantrl.envs.deterministic.LinearizedHOEnv` class.
* Updated Wiener process for drift matrix in `quantrl.envs.stochastic.LinearEnv` class.
* Reversed the sequence of arguments in `quantrl.solvers.differential.SciPyIVPSolver` class.
* Update tick options in `quantrl.plotter` module.
* Updated `README` and `setup`.


## 2024/02/10 - 00 - v0.0.3 - Added Wiener Processes
* Restructured `quantrl.envs.rl` module:
    * Created `quantrl.envs.base` module with `BaseGymEnv`.
    * Created `quantrl.envs.deterministic` module with `LinearizedHOEnv`.
    * Created `quantrl.envs.stochastic` module with `HOEnv`.
* Buffer fixes to `quantrl.solvers.differential.SciPyIVPSolver`.
* Fixed cache dump and underflow in `quantrl.io.FileIO`.
* Added options to make gifs and save plots in `quantrl.plotters.TrajectoryPlotter`.
* Updated `README` and `setup`.

## 2024/01/16 - 00 - v0.0.2 - DDE Support
* Replaced `quantrl.envs.BaseEnv` with `quantrl.envs.rl`.
* Added `quantrl.solvers.differential` module with DDE support.
* Renamed `quantrl.solvers.QCMSolver` to `quantrl.solvers.measure`.
* Replaced `quantrl.log` with `quantrl.io`.
* Replaced `quantrl.utils` with `quantrl.plotters`.
* Added `README`, `requirements` and `setup`.

## 2023/05/20 - 00 - v0.0.1 - Initial Commit
* Added `quantrl.envs.BaseEnv` module.
* Added `quantrl.solvers.QCMSolver` module.
* Added `quantrl.log` and `quantrl.utils` module.
* Added `CHANGELOG`, `README` and `setup`.