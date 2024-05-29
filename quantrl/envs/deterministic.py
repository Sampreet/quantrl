#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with deterministic environments."""

__name__    = 'quantrl.envs.deterministic'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-05-29"

# quantrl modules
from .base import BaseGymEnv, BaseSB3Env

# TODO: ABC for common processes

class LinearizedHOEnv(BaseGymEnv):
    """Class to interface deterministic linearized harmonic oscillator environments.

    Initializes ``dim_corrs``, ``num_corrs``, ``A``, ``D``, ``is_A_constant``, ``is_D_constant`` and ``solver``.
    The interfaced environment requires ``default_params`` dictionary defined before initializing the parent class.

    The interfaced environment needs to implement ``reset_states`` and ``get_reward`` methods.
    Additionally, the ``get_properties`` method should be overridden if ``n_properties`` is non-zero.
    Refer to **Notes** of :class:`quantrl.envs.base.BaseEnv` for their implementations.
    The default ``func`` method can be used to call ``get_mode_rates`` for rates of change of the classical mode amplitudes, ``get_A`` for the Jacobian of the quantum fluctuation quadratures and ``get_D`` for the quantum noise correlations by overriding the corresponding methods.

    Parameters
    ----------
    name: str
        Name of the environment.
    desc: str
        Description of the environment.
    params: dict
        Parameters of the environment.
    num_modes: int
        Number of classical mode amplitudes.
    num_quads: int
        Number of quantum fluctuation quadratures.
    t_norm_max: float
        Maximum time for each episode in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    t_norm_mul: float
        Multiplier to revert the normalization.
    n_observations: int
        Total number of observations.
    n_properties: int
        Total number of properties.
    n_actions: int
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
        Keyword arguments. Refer to the ``kwargs`` parameter of :class:`quantrl.envs.base.BaseEnv` for available options. Additional options are:

        ============    ================================================
        key             value
        ============    ================================================
        ode_method      (*str*) method used to solve the ODEs/DDEs. Available options are ``'dopri8'``, ``'dopri5'``, ``'bosh3'``, ``'fehlberg2'`` and ``'adaptive_huen'`` for a TorchDiffEq-based solver, ``'dopri8'``, ``'dopri5'`` and ``'tsit5'`` for a Diffrax-based solver and ``'BDF'``, ``'DOP853'``, ``'LSODA'``, ``'Radau'``, ``'RK23'``, ``'RK45'``, ``'dop853'``, ``'dopri5'``, ``'lsoda'``, ``'zvode'`` and ``'vode'`` for a SciPy-based solver. Default is ``'vode'``.
        ode_atol        (*float*) absolute tolerance of the ODE/DDE solver. Default is ``1e-9``.
        ode_rtol        (*float*) relative tolerance of the ODE/DDE solver. Default is ``1e-6``.
        ============    ================================================
    """

    default_params = dict()
    """dict: Default parameters of the environment."""

    default_ode_solver_params = dict(
        ode_method='vode',
        ode_atol=1e-9,
        ode_rtol=1e-6
    )
    """dict: Default parameters of the ODE solver."""

    backend_libraries = ['torch', 'jax', 'numpy']
    """list: Available backend libraries."""

    def __init__(self,
        name:str,
        desc:str,
        params:dict,
        num_modes:int,
        num_quads:int,
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
        backend_device:str='cuda',
        dir_prefix:str='data',
        **kwargs
    ):
        """Class constructor for LinearizedHOEnv."""

        # validate arguments
        assert backend_library in self.backend_libraries, "parameter ``solver_type`` should be one of ``{}``".format(self.backend_libraries)

        # select backend
        if 'torch' in backend_library:
            from ..backends.torch import TorchBackend
            from ..solvers.torch import TorchDiffEqIVPSolver as IVPSolverClass
            backend = TorchBackend(
                precision=backend_precision,
                device=backend_device
            )
        elif 'jax' in backend_library:
            from ..backends.jax import JaxBackend
            from ..solvers.jax import DiffraxIVPSolver as IVPSolverClass
            backend = JaxBackend(
                precision=backend_precision
            )
        else:
            from ..backends.numpy import NumPyBackend
            from ..solvers.numpy import SciPyIVPSolver as IVPSolverClass
            backend = NumPyBackend(
                precision=backend_precision
            )

        # set constants
        self.name = name
        self.desc = desc
        self.num_modes = num_modes
        self.dim_corrs = (num_quads, num_quads)
        self.num_corrs = num_quads**2

        # set parameters
        self.params = dict()
        for key in self.default_params:
            self.params[key] = params.get(key, self.default_params[key])
        # update keyword arguments
        for key in self.default_ode_solver_params:
            kwargs[key] = kwargs.get(key, self.default_ode_solver_params[key])
        # set matrices
        self.A = backend.zeros(
            shape=self.dim_corrs,
            dtype='real'
        )
        self.D = backend.zeros(
            shape=self.dim_corrs,
            dtype='real'
        )
        self.is_A_constant = False
        self.is_D_constant = False

        # initialize parent
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
            dir_prefix=(dir_prefix if dir_prefix != 'data' else ('data/' + self.name.lower()) + '/env'),
            file_prefix='lho_env',
            **kwargs
        )

        # initialize solver
        self.solver = IVPSolverClass(
            func=self.func,
            y_0=self.States[-1],
            T=self.T,
            solver_params={
                'method': kwargs['ode_method'],
                'atol': kwargs['ode_atol'],
                'rtol': kwargs['ode_rtol'],
                'is_stiff': False,
                'step_interval': self.action_interval
            },
            func_controls=getattr(self, 'func_controls', None),
            has_delay=self.has_delay,
            func_delay=getattr(self, 'func_delay', None),
            delay_interval=self.action_interval,
            backend=self.backend
        )

        # initialize buffers
        if self.num_modes != 0:
            self.mode_rates_real = self.backend.zeros(
                shape=(2 * self.num_modes, ),
                dtype='real'
            )
        if self.num_corrs != 0:
            self.matmul_0 = self.backend.empty(
                shape=self.dim_corrs,
                dtype='real'
            )
            self.matmul_1 = self.backend.empty(
                shape=self.dim_corrs,
                dtype='real'
            )
            self.sum_0 = self.backend.empty(
                shape=self.dim_corrs,
                dtype='real'
            )
            self.sum_1 = self.backend.empty(
                shape=self.dim_corrs,
                dtype='real'
            )
        self.y_rates = self.backend.empty(
            shape=(2 * self.num_modes + self.num_corrs, ),
            dtype='real'
        )

    def _update_states(self):
        return self.solver.step(
            T_step=self.T_step,
            y_0=self.States[-1],
            params=self.actions
        )

    def func(self,
        t,
        y,
        args:tuple
    ):
        """Method to obtain the rates of change of the real-valued modes and correlations.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        y: Any
            Real-valued modes and flattened correlations with shape ``(2 * num_modes + num_corrs, )``. First ``num_modes`` elements contain the real parts of the modes, the next ``num_modes`` elements contain the imaginary parts of the modes, and the last ``num_corrs`` elements contain the correlations. When ``num_modes`` is ``0``, only the correlations are included. When ``num_corrs`` is ``0``, only the modes are included.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        rates: Any
            Rates of change of the real-valued modes and flattened correlations with shape ``(2 * num_modes + num_corrs, )``.
        """

        # extract frequently used variables
        if self.num_modes != 0:
            modes = y[:self.num_modes] + 1.0j * y[self.num_modes:2 * self.num_modes]
            # get real-valued mode rates
            _mode_rates_real = self.get_mode_rates_real(
                t=t,
                modes_real=y[:2 * self.num_modes],
                args=args
            )
            if self.num_corrs == 0:
                return _mode_rates_real
        else:
            modes=None

        if self.num_corrs != 0:
            corrs = self.backend.reshape(
                tensor=y[2 * self.num_modes:],
                shape=self.dim_corrs
            )

            # get drift matrix
            A = self.A if self.is_A_constant else self.get_A(
                t=t,
                modes=modes,
                args=args
            )

            # get noise matrix
            D = self.D if self.is_D_constant else self.get_D(
                t=t,
                modes=modes,
                args=args
            )

            # get flattened correlation rates
            _corr_rates_flat = self.backend.flatten(
                tensor=self.backend.add(
                    tensor_0=self.backend.add(
                        tensor_0=self.backend.matmul(
                            tensor_0=A,
                            tensor_1=corrs,
                            out=self.matmul_0
                        ),
                        tensor_1=self.backend.matmul(
                            tensor_0=corrs,
                            tensor_1=self.backend.transpose(
                                tensor=A,
                                axis_0=0,
                                axis_1=1
                            ),
                            out=self.matmul_1
                        ),
                        out=self.sum_0
                    ),
                    tensor_1=D,
                    out=self.sum_1
                )
            )

            if self.num_modes == 0:
                return _corr_rates_flat

        return self.backend.concatenate(
            tensors=(
                _mode_rates_real,
                _corr_rates_flat
            ),
            axis=0,
            out=self.y_rates
        )

    def get_A(self,
        t,
        modes,
        args:tuple
    ):
        """Method to obtain the Jacobian of quantum fluctuation quadratures.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        modes: Any
            Classical mode amplitudes with shape ``(num_modes, )``.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        A: Any
            Jacobian of quantum fluctuation quadratures with shape ``(num_quads, num_quads)``.
        """

        raise NotImplementedError

    def get_D(self,
        t,
        modes,
        args:tuple
    ):
        """Method to obtain the quantum noise correaltions.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        modes: Any
            Classical mode amplitudes with shape ``(num_modes, )``.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        D: Any
            Quantum noise correlations with shape ``(num_quads, num_quads)``.
        """

        raise NotImplementedError

    def get_mode_rates(self,
        t,
        modes,
        args
    ):
        """Method to obtain the rates of change of the classical mode amplitudes.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        modes: Any
            Classical mode amplitudes with shape ``(num_modes, )``.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        D: Any
            Rates of change of the classical mode amplitudes with shape ``(num_modes, )``.
        """

        raise NotImplementedError

    def get_mode_rates_real(self,
        t,
        modes_real,
        args:tuple
    ):
        """Method to obtain the real-valued rates of change of the classical mode amplitudes.

        The interfaced environment needs to implement the ``get_mode_rates`` method.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        modes_real: Any
            Real-valued classical mode amplitudes with shape ``(2 * num_modes, )``.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        mode_rates_real: Any
            Real-valued rates of change of the classcial mode amplitudes with shape ``(2 * num_modes, )``.
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

        # return real-valued mode rates
        return self.backend.concatenate(
            tensors=(
                self.backend.real(
                    tensor=mode_rates
                ),
                self.backend.imag(
                    tensor=mode_rates
                )
            ),
            axis=0,
            out=self.mode_rates_real
        )

class LinearizedHOVecEnv(BaseSB3Env):
    """Class to interface deterministic linearized harmonic oscillator vectorized environments.

    Initializes ``dim_corrs``, ``num_corrs``, ``A``, ``D``, ``is_A_constant``, ``is_D_constant`` and ``solver``.
    The interfaced environment requires ``default_params`` dictionary defined before initializing the parent class.
    The paramter ``n_envs`` overrides the ``cache_dump_interval`` parameter.
    For massively parallel models, the ``data_idxs`` parameter should be carefully selected to initialize the ``data`` attribute with shape ``(n_envs, t_dim, n_data_idxs)``.
    In such cases, it is advisable to use the JAX backend with the ``cache_all_data`` paramter to ``False``.

    The interfaced environment needs to implement ``reset_states`` and ``get_reward`` methods.
    Additionally, the ``get_properties`` method should be overridden if ``n_properties`` is non-zero.
    Refer to **Notes** of :class:`quantrl.envs.base.BaseEnv` for their implementations.
    The default ``func`` method can be used to call ``get_mode_rates`` for rates of change of the classical mode amplitudes, ``get_A`` for the Jacobian of the quantum fluctuation quadratures and ``get_D`` for the quantum noise correlations by overriding the corresponding methods.

    Parameters
    ----------
    name: str
        Name of the environment.
    desc: str
        Description of the environment.
    params: dict
        Parameters of the environment.
    num_modes: int
        Number of classical mode amplitudes.
    num_quads: int
        Number of quantum fluctuation quadratures.
    t_norm_max: float
        Maximum time for each episode in normalized units.
    t_norm_ssz: float
        Normalized time stepsize.
    t_norm_mul: float
        Multiplier to revert the normalization.
    n_envs: int
        Number of environments to run in parallel. This value overrides the optional ``cache_dump_interval`` parameter.
    n_observations: int
        Total number of observations.
    n_properties: int
        Total number of properties.
    n_actions: int
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
    kwargs: dict
        Keyword arguments. Refer to the ``kwargs`` parameter of :class:`quantrl.envs.base.BaseEnv` for available options. Additional options are:

        ============    ================================================
        key             value
        ============    ================================================
        ode_method      (*str*) method used to solve the ODEs/DDEs. Available options are ``'dopri8'``, ``'dopri5'``, ``'bosh3'``, ``'fehlberg2'`` and ``'adaptive_huen'`` for a TorchDiffEq-based solver, ``'dopri8'``, ``'dopri5'`` and ``'tsit5'`` for a Diffrax-based solver and ``'BDF'``, ``'DOP853'``, ``'LSODA'``, ``'Radau'``, ``'RK23'``, ``'RK45'``, ``'dop853'``, ``'dopri5'``, ``'lsoda'``, ``'zvode'`` and ``'vode'`` for a SciPy-based solver. Default is ``'vode'``.
        ode_atol        (*float*) absolute tolerance of the ODE/DDE solver. Default is ``1e-9``.
        ode_rtol        (*float*) relative tolerance of the ODE/DDE solver. Default is ``1e-6``.
        ============    ================================================
    """

    default_params = dict()
    """dict: Default parameters of the environment."""

    default_ode_solver_params = dict(
        ode_method='vode',
        ode_atol=1e-9,
        ode_rtol=1e-6
    )
    """dict: Default parameters of the ODE solver."""

    backend_libraries = ['torch', 'jax', 'numpy']
    """list: Available backend libraries."""

    def __init__(self,
        name:str,
        desc:str,
        params:dict,
        num_modes:int,
        num_quads:int,
        t_norm_max:float,
        t_norm_ssz:float,
        t_norm_mul:float,
        n_envs:int,
        n_observations:int,
        n_properties:int,
        n_actions:int,
        action_maximums:list,
        action_interval:int,
        data_idxs:list,
        backend_library:str='numpy',
        backend_precision:str='double',
        backend_device:str='cuda',
        dir_prefix:str='data',
        **kwargs
    ):
        """Class constructor for LinearizedHOEnv."""

        # validate arguments
        assert backend_library in self.backend_libraries, "parameter ``solver_type`` should be one of ``{}``".format(self.backend_libraries)

        # select backend
        if 'torch' in backend_library:
            from ..backends.torch import TorchBackend
            from ..solvers.torch import TorchDiffEqIVPSolver as IVPSolverClass
            backend = TorchBackend(
                precision=backend_precision,
                device=backend_device
            )
        elif 'jax' in backend_library:
            from ..backends.jax import JaxBackend
            from ..solvers.jax import DiffraxIVPSolver as IVPSolverClass
            backend = JaxBackend(
                precision=backend_precision
            )
        else:
            from ..backends.numpy import NumPyBackend
            from ..solvers.numpy import SciPyIVPSolver as IVPSolverClass
            backend = NumPyBackend(
                precision=backend_precision
            )

        # set constants
        self.name = name
        self.desc = desc
        self.num_modes = num_modes
        self.dim_corrs = (num_quads, num_quads)
        self.num_corrs = num_quads**2

        # set parameters
        self.params = dict()
        for key in self.default_params:
            self.params[key] = params.get(key, self.default_params[key])
        # update keyword arguments
        for key in self.default_ode_solver_params:
            kwargs[key] = kwargs.get(key, self.default_ode_solver_params[key])
        # set matrices
        self.A = backend.zeros(
            shape=(n_envs, *self.dim_corrs),
            dtype='real'
        )
        self.D = backend.zeros(
            shape=(n_envs, *self.dim_corrs),
            dtype='real'
        )
        self.is_A_constant = False
        self.is_D_constant = False

        # initialize BaseVecEnv
        super().__init__(
            backend=backend,
            t_norm_max=t_norm_max,
            t_norm_ssz=t_norm_ssz,
            t_norm_mul=t_norm_mul,
            n_envs=n_envs,
            n_observations=n_observations,
            n_properties=n_properties,
            n_actions=n_actions,
            action_maximums=action_maximums,
            action_interval=action_interval,
            data_idxs=data_idxs,
            dir_prefix=(dir_prefix if dir_prefix != 'data' else ('data/' + self.name.lower()) + '/env'),
            file_prefix='lho_vec_env',
            **kwargs
        )

        # initialize solver
        self.solver = IVPSolverClass(
            func=self.func,
            y_0=self.States[-1],
            T=self.T,
            solver_params={
                'method': kwargs['ode_method'],
                'atol': kwargs['ode_atol'],
                'rtol': kwargs['ode_rtol'],
                'is_stiff': False,
                'step_interval': self.action_interval
            },
            func_controls=getattr(self, 'func_controls', None),
            has_delay=self.has_delay,
            func_delay=getattr(self, 'func_delay', None),
            delay_interval=self.action_interval,
            backend=self.backend
        )

        # initialize buffers
        if self.num_modes != 0:
            self.mode_rates_real = self.backend.zeros(
                shape=(self.n_envs, 2 * self.num_modes),
                dtype='real'
            )
        if self.num_corrs != 0:
            self.matmul_0 = self.backend.empty(
                shape=(self.n_envs, *self.dim_corrs),
                dtype='real'
            )
            self.matmul_1 = self.backend.empty(
                shape=(self.n_envs, *self.dim_corrs),
                dtype='real'
            )
            self.sum_0 = self.backend.empty(
                shape=(self.n_envs, *self.dim_corrs),
                dtype='real'
            )
            self.sum_1 = self.backend.empty(
                shape=(self.n_envs, *self.dim_corrs),
                dtype='real'
            )
        self.y_rates = self.backend.empty(
            shape=(self.n_envs, 2 * self.num_modes + self.num_corrs),
            dtype='real'
        )

    def _update_states(self):
        return self.solver.step(
            T_step=self.T_step,
            y_0=self.States[-1],
            params=self.actions
        )

    def func(self,
        t,
        y,
        args:tuple
    ):
        r"""Wrapper function for the rates of change of the real-valued modes and correlations.

        The variables are cast to real.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        y: Any
            Real-valued modes and flattened correlations with shape ``(2 * num_modes + num_corrs, )``. First ``num_modes`` elements contain the real parts of the modes, the next ``num_modes`` elements contain the imaginary parts of the modes, and the last ``num_corrs`` elements contain the correlations. When ``num_modes`` is ``0``, only the correlations are included. When ``num_corrs`` is ``0``, only the modes are included.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        rates: Any
            Rates of change of the real-valued modes and flattened correlations with shape ``(2 * num_modes + num_corrs, )``.
        """

        # extract frequently used variables
        if self.num_modes != 0:
            modes = y[:, :self.num_modes] + 1.0j * y[:, self.num_modes:2 * self.num_modes]
            # get real-valued mode rates
            _mode_rates_real = self.get_mode_rates_real(
                t=t,
                modes_real=y[:, :2 * self.num_modes],
                args=args
            )
            if self.num_corrs == 0:
                return _mode_rates_real
        else:
            modes=None

        if self.num_corrs != 0:
            corrs = self.backend.reshape(
                tensor=y[:, 2 * self.num_modes:],
                shape=(self.n_envs, *self.dim_corrs)
            )

            # get drift matrix
            A = self.A if self.is_A_constant else self.get_A(
                t=t,
                modes=modes,
                args=args
            )

            # get noise matrix
            D = self.D if self.is_D_constant else self.get_D(
                t=t,
                modes=modes,
                args=args
            )

            # get flattened correlation rates
            _corr_rates_flat = self.backend.reshape(
                tensor=self.backend.add(
                    tensor_0=self.backend.add(
                        tensor_0=self.backend.matmul(
                            tensor_0=A,
                            tensor_1=corrs,
                            out=self.matmul_0
                        ),
                        tensor_1=self.backend.matmul(
                            tensor_0=corrs,
                            tensor_1=self.backend.transpose(
                                tensor=A,
                                axis_0=1,
                                axis_1=2
                            ),
                            out=self.matmul_1
                        ),
                        out=self.sum_0),
                    tensor_1=D,
                    out=self.sum_1
                ),
                shape=(self.n_envs, self.num_corrs)
            )

            if self.num_modes == 0:
                return _corr_rates_flat

        return self.backend.concatenate(
            tensors=(
                _mode_rates_real,
                _corr_rates_flat
            ),
            axis=1,
            out=self.y_rates
        )

    def get_A(self,
        t,
        modes,
        args:tuple
    ):
        """Method to obtain the Jacobian of quantum fluctuation quadratures.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        modes: Any
            Classical mode amplitudes with shape ``(n_envs, num_modes)``.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        A: Any
            Jacobian of quantum fluctuation quadratures with shape ``(n_envs, num_quads, num_quads)``.
        """

        raise NotImplementedError

    def get_D(self,
        t,
        modes,
        args:tuple
    ):
        """Method to obtain the quantum noise correaltions.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        modes: Any
            Classical mode amplitudes with shape ``(n_envs, num_modes)``.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        D: Any
            Quantum noise correlations with shape ``(n_envs, num_quads, num_quads)``.
        """

        raise NotImplementedError

    def get_mode_rates(self,
        t,
        modes,
        args:tuple
    ):
        """Method to obtain the rates of change of the classical mode amplitudes.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        modes: Any
            Classical mode amplitudes with shape ``(n_envs, num_modes)``.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        D: Any
            Rates of change of the classical mode amplitudes with shape ``(n_envs, num_modes)``.
        """

        raise NotImplementedError

    def get_mode_rates_real(self,
        t,
        modes_real,
        args:tuple
    ):
        """Method to obtain the real-valued mode rates from real-valued modes.

        The interfaced environment needs to implement the ``get_mode_rates`` method.

        Parameters
        ----------
        t: float
            Time at which the values are calculated.
        modes_real: Any
            Real-valued classical mode amplitudes with shape ``(n_envs, 2 * num_modes)``.
        args: tuple
            Actions, control function and delay function.

        Returns
        -------
        mode_rates_real: Any
            Real-valued rates of change of the classcial mode amplitudes with shape ``(n_envs, 2 * num_modes)``.
        """

        # handle null
        if getattr(self, 'get_mode_rates', None) is None:
            return self.mode_rates_real

        # get complex-valued mode rates
        mode_rates = self.get_mode_rates(
            t=t,
            modes=modes_real[:, :self.num_modes] + 1.0j * modes_real[:, self.num_modes:],
            args=args
        )

        # return real-valued mode rates
        return self.backend.concatenate(
            tensors=(
                self.backend.real(
                    tensor=mode_rates
                ),
                self.backend.imag(
                    tensor=mode_rates
                )
            ),
            axis=1,
            out=self.mode_rates_real
        )