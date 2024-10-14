#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with stochastic environments."""

__name__    = 'quantrl.envs.stochastic'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-10-14"

# dependencies
import numpy as np

# quantrl modules
from ..backends.context_manager import get_backend_instance
from .base import BaseGymEnv

# TODO: Add delay feature
# TODO: Release memory

class LinearEnv(BaseGymEnv):
    """Class to interface stochastic linear environments using Wiener increments.

    Initializes ``A`` and ``is_A_constant``.
    The interfaced environment requires ``default_params`` dictionary defined before initializing the parent class.

    The interfaced environment needs to implement ``reset_states`` and ``get_reward`` methods.
    Additionally, the ``get_properties`` method should be overridden if ``n_properties`` is non-zero.
    Refer to **Notes** of :class:`quantrl.envs.base.BaseEnv` for their implementations.

    The ``func`` method requires ``get_A`` for the Jacobian of the states and the ``get_noise_prefixes`` for the noise values.

    Parameters
    ----------
    name: str
        Name of the environment.
    desc: str
        Description of the environment.
    params: dict
        Parameters of the environment.
    t_norm_max: float
        Maximum time for each episode in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    t_norm_mul: float
        Multiplier to revert the normalization.
    n_observations: tuple
        Total number of observations.
    n_properties: int
        Total number of properties.
    n_actions: tuple
        Total number of observations.
    action_maximums: list
        Maximum values of each action.
    action_interval: int
        Interval at which the actions are updated. Must be positive.
    data_idxs: list
        Indices of the data to store into the ``data`` attribute. The indices can be selected from the complete set of values at each point of time (total ``1 + n_actions + n_observations + n_properties + 1`` elements in the same order, where the first element is the time and the last element is the reward).
    backend_library: str, default='numpy'
        Solver to use for each step. Options are ``'torch'`` for PyTorch-based solvers, ``'jax'`` for JAX-based solvers and ``'numpy'`` for NumPy/SciPy-based solvers.
    backend_precision: str, default='double'
        Precision of the numerical values in the backend. Options are ``'single'`` and ``'double'``.
    backend_device: str, default='cuda'
        Device to run the solver. Options are ``'cpu'`` and ``'cuda'``.
    dir_prefix: str, default='data'
        Prefix of the directory where the data will be stored.
    kwargs: dict, optional
        Keyword arguments. Refer to the ``kwargs`` parameter of :class:`quantrl.envs.base.BaseEnv` for available options.
    """

    default_params = {}
    """dict: Default parameters of the environment."""

    backend_libraries = ['torch', 'jax', 'numpy']
    """list: Available backend libraries."""

    def __init__(self,
        name:str,
        desc:str,
        params:dict,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_observations:int,
        n_properties:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        data_idxs:list,
        backend_library:str='numpy',
        backend_precision:str='double',
        backend_device:str='gpu',
        dir_prefix:str='data',
        **kwargs
    ):
        """Class constructor for LinearEnv."""

        # validate arguments
        assert backend_library in self.backend_libraries, f"parameter ``solver_type`` should be one of ``{self.backend_libraries}``"

        # select backend
        backend = get_backend_instance(
            library=backend_library,
            precision=backend_precision,
            device=backend_device
        )

        # set constants
        self.name = name
        self.desc = desc

        # set parameters
        self.params = {}
        for key, _ in self.default_params.items():
            self.params[key] = params.get(key, self.default_params[key])
        # set buffers
        self.Ws = None
        self.I = backend.eye(
            N=n_observations,
            dtype='real'
        )
        self.A = backend.zeros(
            shape=(n_observations, n_observations),
            dtype='real'
        )
        self.is_A_constant = False

        # initialize BaseGymEnv
        super().__init__(
            backend=backend,
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            n_observations=n_observations,
            n_properties=n_properties,
            n_actions=n_actions,
            action_maximums=action_maximums,
            action_interval=action_interval,
            data_idxs=data_idxs,
            dir_prefix=(dir_prefix if dir_prefix != 'data' else ('data/' + self.name.lower()) + '/env') + '_' + '_'.join([
                str(val) for _, val in self.params.items()
            ]),
            file_prefix='lin_env',
            **kwargs
        )

        # initialize buffers
        self.add_0 = self.backend.empty(
            shape=(self.n_observations, ),
            dtype='real'
        )
        self.matmul_0 = self.backend.empty(
            shape=(self.n_observations, ),
            dtype='real'
        )

    def reset(self,
        seed:float=None,
        options:dict=None
    ):
        # update Wiener noises
        self.Ws = np.sqrt(self.t_ssz) * self.backend.normal(
            generator=self.backend.generator(
                seed=self.seed
            ),
            shape=(self.shape_T[0] - 1, self.n_observations),
            mean=0.0,
            std=1.0,
            dtype='real'
        )

        # return observations
        return super().reset(
            seed=seed,
            options=options
        )

    def _update_states(self):
        # initialize states
        _States = self.backend.jit_update(
            tensor=self.States,
            indices=0,
            values=self.States[-1]
        )

        # iterate and return
        return self.backend.iterate_i(
            func=self.func,
            iterations_i=self.backend.shape(
                tensor=self.T_step
            )[0] - 1,
            Y=_States,
            args=(self.actions, None, None)
        )

    def func(self,
        i,
        Y,
        args:tuple
    ):
        """Method to obtain the rates of change of the real-valued variables.

        Parameters
        ----------
        i: int
            Index of the iteration.
        Y: Any
            Real-valued variables.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        rates: Any
            Rates of change of the real-valued modes and flattened correlations with shape ``(2 * num_modes + num_corrs, )``.
        """

        # get drift matrix
        M_i = self.I + self.get_A(
            t_idx=self.t_idx + i,
            args=args
        ) * self.t_ssz
        # get noise prefixes
        n_i = self.get_noise_prefixes(
            t_idx=self.t_idx + i,
            args=args
        )
        # get updated states
        values = self.backend.jit_add(
            tensor_0=self.backend.jit_matmul(
                tensor_0=M_i,
                tensor_1=Y[i],
                out=self.matmul_0
            ),
            tensor_1=n_i * self.Ws[self.t_idx + i],
            out=self.add_0
        )
        return self.backend.jit_update(
            tensor=Y,
            indices=i + 1,
            values=values
        )

    def get_A(self,
        t_idx:int,
        args:tuple
    ):
        """Method to obtain the Jacobian of the states.

        Parameters
        ----------
        t_idx: int
            Index of the time at which the values are calculated.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        A: Any
            Jacobian of the states with shape ``(n_observations, n_observations)``.
        """

        raise NotImplementedError

    def get_noise_prefixes(self,
        t_idx:int,
        args:tuple
    ):
        """Method to obtain the noise prefixes for each state.

        Parameters
        ----------
        t_idx: int
            Index of the time at which the values are calculated.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        noise_prefixes: Any
            Noise prefixes for each observation with shape ``(n_observations, )``.
        """

        raise NotImplementedError
