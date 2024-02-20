# Changelog

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