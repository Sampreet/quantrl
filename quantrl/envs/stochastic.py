#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with stochastic environments."""

__name__    = 'quantrl.envs.stochastic'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-03-18"

# dependencies
import numpy as np

# quantrl modules
from .base import BaseGymEnv

# TODO: Separate iterative solver to support JAX
# TODO: Add MCQT
# TODO: Add delay feature

class LinearEnv(BaseGymEnv):
    """Class to interface stochastic linear environments using Wiener increments.

    Initializes ``A`` and ``is_A_constant``.
    The interfaced environment requires ``default_params`` dictionary defined before initializing the parent class.

    The interfaced environment needs to implement ``reset_observations`` and ``get_reward`` methods.
    Additionally, the ``get_properties`` method should be overridden if ``n_properties`` is non-zero.
    Refer to **Notes** of :class:`quantrl.envs.base.BaseEnv` for their implementations.

    The ``_update_observations`` method requires ``get_A`` for the Jacobian of the observations and the ``get_noise`` for the noise values.

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

    default_params = dict()
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
        backend_library:str='numpy',
        backend_precision:str='double',
        backend_device:str='cuda',
        dir_prefix:str='data',
        **kwargs
    ):
        """Class constructor for LinearEnv."""

        # validate arguments
        assert backend_library in self.backend_libraries, "parameter ``solver_type`` should be one of ``{}``".format(self.backend_libraries)

        # select backend
        if 'torch' in backend_library:
            from ..backends.torch import TorchBackend
            backend = TorchBackend(
                precision=backend_precision,
                device=backend_device
            )
        elif 'jax' in backend_library:
            from ..backends.jax import JaxBackend
            backend = JaxBackend(
                precision=backend_precision
            )
        else:
            from ..backends.numpy import NumPyBackend
            backend = NumPyBackend(
                precision=backend_precision
            )

        # set constants
        self.name = name
        self.desc = desc

        # set parameters
        self.params = dict()
        for key in self.default_params:
            self.params[key] = params.get(key, self.default_params[key])
        # set matrices
        self.I = backend.eye(
            rows=n_observations,
            cols=None,
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
            dir_prefix=(dir_prefix if dir_prefix != 'data' else ('data/' + self.name.lower()) + '/env') + '_' + '_'.join([
                str(self.params[key]) for key in self.params
            ]),
            file_prefix='lin_env',
            **kwargs
        )

        # initialize solver
        self.seeds = np.random.default_rng().integers(low=0, high=1000, size=self.shape_T, endpoint=False, dtype=np.int32)

        # initialize buffers
        self.matmul_0 = self.backend.empty(
            shape=(self.n_observations, ),
            dtype='real'
        )

    def _update_observations(self):
        # frequently used variables
        _shape = self.backend.shape(
            tensor=self.T_step
        )
        self.Ws = self.backend.normal(
            shape=_shape,
            mean=0.0,
            std=np.sqrt(self.t_norm_ssz),
            seed=int(self.seeds[self.t_idx]),
            dtype='real'
        )

        # increment observations
        Observations = tuple()
        Observations += (self.Observations[-1] + 0.0, )
        for i in range(1, _shape[0]):
            # get drift matrix
            M_i = self.I + self.get_A(
                t=self.T_norm[self.t_idx + i],
                args=[self.actions, None, None]
            ) * self.t_norm_ssz
            # get noise prefixes
            n_i = self.get_noise_prefixes(
                t=self.T_norm[self.t_idx + i],
                args=[self.actions, None, None]
            )
            # update observations
            Observations += (self.backend.matmul(
                tensor_0=M_i,
                tensor_1=Observations[i - 1],
                out=self.matmul_0
            ) + n_i * self.Ws[i], )

        return self.backend.stack(
            tensors=Observations,
            axis=0,
            out=self.Observations
        )

    def get_A(self,
        t,
        args
    ):
        """Method to obtain the Jacobian of the observations.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        A: Any
            Jacobian of the observations with shape ``(n_observations, n_observations)``.
        """

        raise NotImplementedError

    def get_noise_prefixes(self,
        t,
        args
    ):
        """Method to obtain the noise prefixes for each observations.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        noise_prefixes: Any
            Noise prefixes for each observation with shape ``(n_observations, )``.
        """

        raise NotImplementedError