#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with stochastic environments."""

__name__    = 'quantrl.envs.stochastic'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-02-17"

# dependencies
import numpy as np

# quantrl modules
from .base import BaseGymEnv

# TODO: Implement MCQT
# TODO: Implement delay function

class LinearEnv(BaseGymEnv):
    """Class to interface stochastic linear environments.

    The interfaced environment requires ``default_params`` dictionary defined before initializing the parent class.
    The interfaced environment needs to implement ``reset_Observations``, ``get_Properties`` and ``get_Reward`` methods.
    Refer to **Notes** of :class:`quantrl.envs.base.BaseGymEnv` for their implementations.
    The ``_step_wiener`` method requires ``get_M`` for the evolution matrices and the ``get_noise`` for the noise values.
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
        Solver to evolve each time step. Options are ``'mcqt'`` for Monte-Carlo quantum trajectories and ``'wiener'`` for Weiner processes. Default is ``'wiener'``.
    dir_prefix: str, default='data'
        Prefix of the directory where the data will be stored.
    kwargs: dict, optional
        Keyword arguments. Refer to the ``kwargs`` parameter of :class:`quantrl.envs.base.BaseGymEnv` for available options. Additional options are:
        ============    ================================================
        key             value
        ============    ================================================
        ode_method      (*str*) method used to solve the ODEs/DDEs if ``solver_type`` is ``'mcqt'``. Available options are ``'BDF'``, ``'DOP853'``, ``'LSODA'``, ``'Radau'``, ``'RK23'``, ``'RK45'``, ``'dop853'``, ``'dopri5'``, ``'lsoda'``, ``'zvode'`` and ``'vode'``. Default is ``'vode'``.
        ode_atol        (*float*) absolute tolerance of the ODE/DDE solver. Default is ``1e-12``.
        ode_rtol        (*float*) relative tolerance of the ODE/DDE solver. Default is ``1e-6``.
        ============    ================================================

    Notes
    -----
        The following required methods follow a strict formatting:
            ============    ================================================
            method          returns
            ============    ================================================
            get_A           the drift matrix with shape ``(n_observations, n_observations)``, formatted as ``get_A(t, args)``, where ``args`` is a list containing the array of actions at normalized time ``t``, the control function formatted as ``func_control(t)`` and the delay function formatted as ``func_delay(t)``..
            get_noises      the prefixes for the noise vector with shape ``(n_observations, )``. It follows the same formatting as ``get_A``.
            ============    ================================================
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
        n_trajectories:int,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_observations:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        solver_type:str='wiener',
        dir_prefix:str='data',
        **kwargs
    ):
        """Class constructor for LinearEnv."""

        # set constants
        self.name = name
        self.desc = desc
        # set parameters
        self.params = dict()
        for key in self.default_params:
            self.params[key] = params.get(key, self.default_params[key])
        # update keyword arguments
        for key in self.default_ode_solver_params:
            kwargs[key] = kwargs.get(key, self.default_ode_solver_params[key])
        # set matrices
        self.I = np.eye(n_observations, dtype=np.float_)
        self.A = np.zeros((n_observations, n_observations), dtype=np.float_)

        # initialize BaseGymEnv
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
        if 'mcqt' in self.solver_type:
            return NotImplementedError

    def _step(self,
        actions
    ):
        """Method to implement one step.

        Parameters
        ----------
        actions: :class:`numpy.ndarray`
            Actions at current time.
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
            Actions at current time.

        Returns
        -------
        Observations: :class:`numpy.ndarray`
            Observations for the action interval.
        """

        # increment observations
        self.Ws = np.sqrt(self.t_norm_ssz) * np.random.normal(loc=0.0, scale=1.0, size=(self.T_step.shape[0]))
        Observations = np.empty((self.T_step.shape[0], self.n_observations), dtype=np.float_)
        Observations[0] = self.Observations[-1]
        for i in range(1, self.T_step.shape[0]):
            # get drift matrix
            M_i = self.I + self.get_A(
                t=self.T_norm[self.t_idx + i],
                args=[actions, None, None]
            ) * self.t_norm_ssz
            # get noise
            n_i = self.get_noises(
                t=self.T_norm[self.t_idx + i],
                args=[actions, None, None]
            )
            Observations[i] = M_i.dot(Observations[i - 1]) + n_i * self.Ws[i]

        return Observations