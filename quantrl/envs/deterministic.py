#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with deterministic environments."""

__name__    = 'quantrl.envs.deterministic'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-02-17"

# dependencies
import numpy as np

# quantrl modules
from .base import BaseGymEnv
from ..solvers.differential import SciPyIVPSolver

# TODO: Implement TorchIVPSolver

class LinearizedHOEnv(BaseGymEnv):
    """Class to interface deterministic linearized harmonic oscillator environments.

    Initializes ``dim_corrs``, ``num_corrs``, ``is_A_constant`` and ``is_D_constant``.

    The interfaced environment requires ``default_params`` dictionary defined before initializing the parent class.
    The interfaced environment needs to implement ``reset_Observations``, ``get_Properties`` and ``get_Reward`` methods.
    Refer to **Notes** of :class:`quantrl.envs.base.BaseGymEnv` for their implementations.
    The ``func`` method can be opted to call ``get_mode_rates`` for the classical mode amplitudes, ``get_A`` for the Jacobian of the quantum fluctuation quadratures and ``get_D`` for the noise correlations.
    Refer to **Notes** below for their implementations.

    Parameters
    ----------
    name: str
        Name of the environment.
    desc: str
        Description of the environment.
    params: dict
        Parameters of the environment.
    n_trajectories: int
        Total number of trajectories.
    t_norm_max: float
        Maximum time for each trajectory in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    t_norm_mul: float
        Multiplier to revert the normalization.
    n_observations: tuple
        Total number of observations.
    n_actions: tuple
        Total number of observations.
    action_maximums: list
        Maximum values of each action.
    action_interval: int
        Interval at which the actions are updated. Must be positive.
    solver_type: str, optional
        Solver to evolve each time step. Options are ``'torch'`` for TorchDiffEq-based ODE/DDE solvers using :class:`quantrl.solvers.differential.TorchIVPSolver` and ``'scipy'`` for SciPy-based ODE/DDE solvers using :class:`quantrl.solvers.differential.SciPyIVPSolver`. Default is ``'scipy'``.
    dir_prefix: str, default='data'
        Prefix of the directory where the data will be stored.
    kwargs: dict, optional
        Keyword arguments. Refer to the ``kwargs`` parameter of :class:`quantrl.envs.base.BaseGymEnv` for available options. Additional options are:
        ============    ================================================
        key             value
        ============    ================================================
        ode_method      (*str*) method used to solve the ODEs/DDEs. Available options are ``'BDF'``, ``'DOP853'``, ``'LSODA'``, ``'Radau'``, ``'RK23'``, ``'RK45'``, ``'dop853'``, ``'dopri5'``, ``'lsoda'``, ``'zvode'`` and ``'vode'``. Default is ``'vode'``.
        ode_atol        (*float*) absolute tolerance of the ODE/DDE solver. Default is ``1e-12``.
        ode_rtol        (*float*) relative tolerance of the ODE/DDE solver. Default is ``1e-6``.
        ============    ================================================

    Notes
    -----
        The following optional methods follow a strict formatting:
            ================    ================================================
            method              returns
            ================    ================================================
            get_A               the drift matrix with shape ``(num_quads, num_quads)``, formatted as ``get_A(t, modes, args)``, where ``modes`` are the mode amplitudes at normalized time ``t`` and ``args`` is a list containing the array of actions, the control function formatted as ``func_control(t)`` and the delay function formatted as ``func_delay(t)``.
            get_D               the noise matrix with shape ``(num_quads, num_quads)``. It follows the same formatting as ``get_A``.
            get_mode_rates      the rate of change of the modes with shape ``(num_modes, )``. It follows the same formatting as ``get_A``.
            ================    ================================================
    """

    default_params = dict()
    """dict: Default parameters of the environment."""

    default_ode_solver_params = dict(
        ode_method='vode',
        ode_atol=1e-12,
        ode_rtol=1e-6
    )
    """dict: Default parameters of the ODE solver."""

    def __init__(self,
        name:str,
        desc:str,
        params:dict,
        num_modes:int,
        num_quads:int,
        n_trajectories:int,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_observations:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        solver_type:str='scipy',
        dir_prefix:str='data',
        **kwargs
    ):
        """Class constructor for LinearizedHOEnv."""

        # set constants
        self.name = name
        self.desc = desc
        self.num_modes = num_modes
        self.dim_corrs = (num_quads, num_quads)
        self.num_corrs = num_quads**2
        self.is_A_constant = False
        self.is_D_constant = False
        # set parameters
        self.params = dict()
        for key in self.default_params:
            self.params[key] = params.get(key, self.default_params[key])
        # update keyword arguments
        for key in self.default_ode_solver_params:
            kwargs[key] = kwargs.get(key, self.default_ode_solver_params[key])
        # set matrices
        self.A = np.zeros(self.dim_corrs, dtype=np.float_)
        self.D = np.zeros(self.dim_corrs, dtype=np.float_)

        # initialize parent
        super().__init__(
            n_trajectories=n_trajectories,
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            n_observations=n_observations,
            n_actions=n_actions,
            action_maximums=action_maximums,
            action_interval=action_interval,
            dir_prefix=(dir_prefix if dir_prefix != 'data' else ('data/' + self.name.lower()) + '/env') + '_' + '_'.join([
                str(self.params[key]) for key in self.params
            ]),
            **kwargs
        )

        # initialize solver
        self.solver_type = solver_type
        if 'torch' in self.solver_type:
            return NotImplementedError
        else:
            self.solver = SciPyIVPSolver(
                func=self.func,
                solver_params={
                    't_min': self.T[0],
                    't_max': self.T[-1],
                    't_dim': self.t_dim,
                    'ode_method': kwargs['ode_method'],
                    'ode_atol': kwargs['ode_atol'],
                    'ode_rtol': kwargs['ode_rtol'],
                    'ode_is_stiff': False,
                    'ode_step_dim': self.action_interval
                },
                func_controls=getattr(self, 'func_controls', None),
                has_delay=self.has_delay,
                func_delay=getattr(self, 'func_delay', None),
                t_delay=self.t_delay
            )

        # initialize buffers
        if self.num_modes != 0:
            self.mode_rates_real = np.zeros(2 * self.num_modes, dtype=np.float_)
        if self.num_corrs != 0:
            self.matmul_0 = np.empty(self.dim_corrs, dtype=np.float_)
            self.matmul_1 = np.empty(self.dim_corrs, dtype=np.float_)
            self.sum_0 = np.empty(self.dim_corrs, dtype=np.float_)
            self.sum_1 = np.empty(self.dim_corrs, dtype=np.float_)
        self.y_rates = np.empty(2 * self.num_modes + self.num_corrs, dtype=np.float_)

    def _step(self,
        actions
    ):
        """Method to implement one step.

        Parameters
        ----------
        actions: :class:`numpy.ndarray`
            Array of actions at current time.
        """

        # solve coupled ODEs/DDEs using TorchIVPSolver
        if self.solver_type == 'torch':
            return NotImplementedError
        # solve coupled ODEs/DDEs using SciPyIVPSolver
        else:
            return self.solver.step_ivp(
                y0=self.Observations[-1],
                T_step=self.T_step,
                params=actions
            )

    def func(self,
        t,
        y,
        args
    ):
        r"""Wrapper function for the rates of change of the real-valued modes and correlations.

        The variables are casted to real.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        y: :class:`numpy.ndarray`
            Real-valued modes and flattened correlations. First ``num_modes`` elements contain the real parts of the modes, the next ``num_modes`` elements contain the imaginary parts of the modes, and the last ``num_corrs`` elements contain the correlations. When ``num_modes`` is ``0``, only the correlations are included. When ``num_corrs`` is ``0``, only the modes are included.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        rates: :class:`numpy.ndarray`
            Rates of change of the real-valued modes and flattened correlations.
        """

        # extract frequently used variables
        if self.num_modes != 0:
            modes = y[:self.num_modes] + 1.0j * y[self.num_modes:2 * self.num_modes]
            # get real-valued mode rates
            self.y_rates[:2 * self.num_modes] = self.get_mode_rates_real(
                t=t,
                modes_real=y[:2 * self.num_modes],
                args=args
            )
        else:
            modes=None

        if self.num_corrs != 0:
            corrs = np.reshape(y[2 * self.num_modes:], self.dim_corrs)

            # get drift matrix
            self.A = self.A if self.is_A_constant else self.get_A(
                t=t,
                modes=modes,
                args=args
            )

            # get noise matrix
            self.D = self.D if self.is_D_constant else self.get_D(
                t=t,
                modes=modes,
                args=args
            )

            # get flattened correlation rates
            self.y_rates[2 * self.num_modes:] = np.add(np.add(np.matmul(self.A, corrs, out=self.matmul_0), np.matmul(corrs, self.A.transpose(), out=self.matmul_1), out=self.sum_0), self.D, out=self.sum_1).ravel()

        return self.y_rates

    def get_mode_rates_real(self,
        t,
        modes_real,
        args
    ):
        """Method to obtain the real-valued mode rates from real-valued modes.

        Requires the system method ``get_mode_rates``. Refer to **Notes** for its implementation.

        Parameters
        ----------
        t: float
            Time at which the drift matrix is calculated.
        modes_real: :class:`numpy.ndarray`
            Real-valued modes.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        mode_rates_real: :class:`numpy.ndarray`
            Real-valued rates for each mode.
        """

        # handle null
        if getattr(self, 'get_mode_rates', None) is None:
            return self.mode_rates_real

        # get complex-valued mode rates
        mode_rates = self.get_mode_rates(
            t=t,
            modes=modes_real[:self.num_modes] + 1.0j * modes_real[self.num_modes:],
            args=args
        )

        # set real-valued mode rates
        self.mode_rates_real[:self.num_modes] = np.real(mode_rates)
        self.mode_rates_real[self.num_modes:] = np.imag(mode_rates)

        return self.mode_rates_real