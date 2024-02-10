#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""Module with stochastic environments."""

__name__    = 'quantrl.envs.stochastic'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-02-10"

# dependencies
import numpy as np

# quantrl modules
from .base import BaseGymEnv

# TODO: Implement MCQT

class HOEnv(BaseGymEnv):
    """Class to interface stochastic harmonic oscillator environments.
    
    The interfaced environment requires ``default_params`` dictionary defined before initializing the parent class.
    The interfaced environment needs to implement ``reset_Observations``, ``get_Properties`` and ``get_Reward`` methods.
    Refer to **Notes** of :class:`quantrl.envs.base.BaseGymEnv` for their implementations.
    The ``_step_wiener`` method requires ``get_M`` for the evolution matrices and the ``get_noise`` for the noise values.
    Refer to **Notes** below for their implementations.

    Parameters
    ----------
    params: dict
        Parameters of the environment.
    name: str
        Name of the environment.
    desc: str
        Description of the environment.
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
        Solver to evolve each time step. Options are ``'mcqt'`` for Monte-Carlo quantum trajectories and ``'wiener'`` for Weiner processes. Default is ``'wiener'``.
    ode_method: str, optional
        Method used to solve the ODEs/DDEs if ``solver_type`` is ``'mcqt'``. Available options are ``'BDF'``, ``'DOP853'``, ``'LSODA'``, ``'Radau'``, ``'RK23'``, ``'RK45'``, ``'dop853'``, ``'dopri5'``, ``'lsoda'``, ``'zvode'`` and ``'vode'``. Default is ``'vode'``.
    ode_atol: float, optional
        Absolute tolerance of the ODE/DDE solver. Default is ``1e-12``.
    ode_rtol: float, optional
        Relative tolerance of the ODE/DDE solver. Default is ``1e-6``.
    plot: bool, optional
        Option to plot the trajectories using :class:`quantrl.plotters.TrajectoryPlotter`. Default is ``True``.
    plot_interval: int, optional
        Number of trajectories after which the plots are updated. Must be non-negative. If ``0``, the plots are plotted after each step.
    plot_idxs: list, optional
        Positions of the data array to plot at each time step. Default is ``[-1]`` for the cummulative reward.
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
        The following required methods follow a strict formatting:
            ================    ================================================
            method              returns
            ================    ================================================
            get_M               the evolution matrix with shape ``(n_observations, n_observations)``, formatted as ``get_M(i, params)``, where ``params`` is the array of actions at the ``i``-th time step.
            get_noise           the noise values with shape ``(n_observations, )``. It follows the same formatting as ``get_M``.
            ================    ================================================
    """

    # attribute
    default_params = dict()
    """dict: Default parameters of the environment."""

    def __init__(self,
        name:str,
        desc:str,
        params:dict,
        n_trajectories:int,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        action_maxs:list,
        action_interval:int=1,
        has_delay:bool=False,
        reward_max:float=1.0,
        reward_noise:float=0.0,
        action_space_range:list=[-1.0, 1.0],
        observation_space_range:list=[-1e6, 1e6],
        observation_space_shape:tuple=(10, ),
        save_properties:bool=False,
        solver_type:str='wiener',
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
        """Class constructor for HOEnv."""

        # set constants
        self.name = name
        self.desc = desc
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
        if 'mcqt' in self.solver_type:
            return NotImplementedError
        # else:
        #     self.W_is = np.sqrt(self.t_norm_ssz) * np.random.normal(loc=0.0, scale=1.0, size=(self.t_dim))

        # initialize buffer
        self._Y = None

    def _step(self,
        actions
    ):
        """Method to implement one step.

        Parameters
        ----------
        actions: :class:`numpy.ndarray`
            Array of actions at current time.
        """

        # Monte-Carlo quantum trajectories
        if self.solver_type == 'mcqt':
            return NotImplementedError
        # Wiener processes
        else:
            return self._step_wiener(
                actions=actions
            )
    
    def _step_wiener(self,
        actions
    ):
        """Method to implement one step of a Wiener process.

        Parameters
        ----------
        actions: :class:`numpy.ndarray`
            Array of actions at current time.

        Returns
        -------
        Observations: :class:`numpy.ndarray`
            Observations for the action interval.
        """

        # increment observations
        Observations = np.empty((len(self.T_step), self.n_observations), dtype=np.float_)
        Observations[0] = self.Observations[-1]
        for i in range(1, len(self.T_step)):
            # get evolution matrix
            M_i = self.get_M(
                i=self.t_idx + i,
                params=actions
            )
            # get noise
            n_i = self.get_noise(
                i=self.t_idx + i,
                params=actions
            )
            Observations[i] = M_i.dot(Observations[i - 1]) + n_i

        return Observations