#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""Module with deterministic environments."""

__name__    = 'quantrl.envs.deterministic'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-01-16"

# dependencies
import numpy as np

# quantrl modules
from .base import BaseGymEnv
from ..solvers.differential import SciPyIVPSolver

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
    num_modes: int
        Number of classical mode amplitudes in the environment.
    num_quads: int
        Number of quantum fluctuation quadratures in the environment.
    n_trajectories: int
        Total number of trajectories.
    t_norm_max: float
        Maximum time for each trajectory in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    t_norm_mul: float
        Multiplier to revert the normalization.
    action_maxs: list
        Maximum values of each action.
    actions_interval: int, optional
        Interval at which the actions are updated. Must be positive. Default is ``1``.
    reward_max: float, optional
        Maximum value of reward (implemented in children). Default is ``1.0``.
    reward_noise: float, optional
        Noise in the reward function (implemented in children). Default is ``0.0``.
    action_space_range: list, optional
        Range of the actions obtained from the network. The output is scaled by the corresponding action multiplier. Default is ``[-1, 1]``.
    observation_space_range: list, optional
        Range of the observations. Default is ``[-1e6, 1e6]``.
    observation_space_shape: tuple, optional
        Shape of the observations. Default is ``(10, )``.
    save_properties: bool, optional
        Option to save additional properties for each time step. Requires ``get_properties()`` method implemented in children.
    solver_type: str, optional
        Solver to evolve each time step. Options are ``'torchdiffeq'`` for TorchDiffEq-based ODE/DDE solvers using :class:`quantrl.solvers.differential.TorchDiffEqIVPSolver` and ``'scipy'`` for SciPy-based ODE/DDE solvers using :class:`quantrl.solvers.differential.SciPyIVPSolver`. Default is ``'scipy'``.
    ode_method: str, optional
        Method used to solve the ODEs/DDEs. Available options are ``'BDF'``, ``'DOP853'``, ``'LSODA'``, ``'Radau'``, ``'RK23'``, ``'RK45'``, ``'dop853'``, ``'dopri5'``, ``'lsoda'``, ``'zvode'`` and ``'vode'``. Default is ``'vode'``.
    ode_atol: float, optional
        Absolute tolerance of the ODE/DDE solver. Default is ``1e-12``.
    ode_rtol: float, optional
        Relative tolerance of the ODE/DDE solver. Default is ``1e-6``.
    plot: bool, optional
        Option to plot the trajectories using :class:`quantrl.plotters.TrajectoryPlotter`. Default is ``True``.
    plot_interval: int, optional
        Number of trajectories after which the plots are updated. Must be non-negative. If ``0``, the plots are plotted after each step.
    plot_idxs: list, optional
        Indices of the data values required to plot at each time step. Default is ``[-1]`` for the cummulative reward.
    axes_args: list, optional
        Lists of axis properties. The first element of each is the ``x_label``, the second is ``y_label``, the third is ``[y_limit_min, y_limit_max]`` and the fourth is ``y_scale``. Default is ``[['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']]``.
    axes_lines_max: int, optional
        Maximum number of lines to display in each plot. Higher numbers slow down the run. Default is ``100``.
    axes_cols: int, optional
        Number of columns in the figure. Default is ``3``.
    dir_prefix: str, optional
        Prefix of the directory where the data will be stored. Default is ``'data'``.
    max_trajectories_per_file: int, optional
        Maximum number of trajectory data to save per file. Default is ``100``.

    Notes
    -----
        The following optional methods follow a strict formatting:
            ================    ================================================
            method              returns
            ================    ================================================
            get_A               the drift matrix, formatted as ``get_A(t, modes, args)``, where ``modes`` are the mode amplitudes at time ``t`` and ``args`` is a list containing the delay function formatted as ``func_delay(t)``, the control function formatted as ``func_control(t)`` and the array of actions.
            get_D               the noise matrix. It follows the same formatting as ``get_A``.
            get_mode_rates      the rate of change of the modes. It follows the same formatting as ``get_A``.
            ================    ================================================
    """

    default_params = dict()
    """dict: Default parameters of the environment."""

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
        action_maxs:list,
        action_interval:int=1,
        has_delay:bool=False,
        reward_max:float=1.0,
        reward_noise:float=0.0,
        action_space_range:list=[-1, 1],
        observation_space_range:list=[-1e6, 1e6],
        observation_space_shape:tuple=(10, ),
        save_properties:bool=False,
        solver_type:str='scipy',
        ode_method:str='vode',
        ode_atol:float=1e-12,
        ode_rtol:float=1e-6,
        plot:bool=True,
        plot_interval:int=1,
        plot_idxs:list=[-1],
        axes_args:list=[
            ['$t / t_{0}$', '$\\tilde{R}$', [np.sqrt(10) * 1e-1, np.sqrt(10) * 1e6], 'log']
        ],
        axes_lines_max:int=100,
        axes_cols:int=3,
        dir_prefix:str='data/env',
        max_trajectories_per_file:int=100
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
            
        # initialize parent
        super().__init__(
            n_trajectories=n_trajectories,
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            action_maxs=action_maxs,
            action_interval=action_interval,
            has_delay=has_delay,
            reward_max=reward_max,
            reward_noise=reward_noise,
            action_space_range=action_space_range,
            observation_space_range=observation_space_range,
            observation_space_shape=observation_space_shape,
            save_properties=save_properties,
            plot=plot,
            plot_interval=plot_interval,
            plot_idxs=plot_idxs,
            axes_args=axes_args,
            axes_lines_max=axes_lines_max,
            axes_cols=axes_cols,
            dir_prefix=(dir_prefix if dir_prefix != 'data/env' else ('data/' + self.name.lower())) + '_' + '_'.join([
                str(self.params[key]) for key in self.params
            ]),
            max_trajectories_per_file=max_trajectories_per_file
        )

        # initialize solver
        self.solver_type = solver_type
        if 'torchdiffeq' in self.solver_type:
            return NotImplementedError
        else:
            self.solver = SciPyIVPSolver(
                func=self.func,
                solver_params={
                    't_min': self.T[0],
                    't_max': self.T[-1],
                    't_dim': self.t_dim,
                    'ode_method': ode_method,
                    'ode_atol': ode_atol,
                    'ode_rtol': ode_rtol,
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
            self.A = np.zeros(self.dim_corrs, dtype=np.float_)
            self.D = np.zeros(self.dim_corrs, dtype=np.float_)
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

        # Wiener processes
        if self.solver_type == 'torchdiffeq':
            return NotImplementedError
        # coupled ODEs/DDEs
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
            Real-valued modes and flattened correlations. First ``num_modes`` elements contain the real parts of the modes, the next ``num_modes`` elements contain the imaginary parts of the modes and optionally, the last ``num_corrs`` elements contain the correlations.
        args: tuple
            Delay function, control function and constant parameters.

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
            Delay function, control function and constant parameters.
        
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