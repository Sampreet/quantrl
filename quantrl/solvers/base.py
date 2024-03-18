#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with base classes to interface different solvers."""

__name__    = 'quantrl.solvers.base'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-03-17"

# dependencies
from abc import ABC, abstractmethod
from tqdm import tqdm

# quantrl modules
from ..backends.base import BaseBackend

class BaseIVPSolver(ABC):
    """ODE and DDE solver backend for initial-value problems.

    The inherited classes should contain the ``solver_methods`` attribute listing all the available methods of the corresponding solver.

    Currently, the module only supports a single delay interval.

    Parameters
    ----------
    func: callable
        ODE/DDE function in the format ``func(t, y, args)``.
        The first element of ``args`` contains the constant parameters, the second element contains the function for the controls and the third contains the delay function.
    solver_params: dict
        Parameters of the solver.
        Currently supported options are:
            ============    ====================================================
            key             value
            ============    ====================================================
            method          (*str*) method used to solve the ODEs. Refer to the documentation of the inherited solvers. Default is ``'vode'`` from :class:`quantrl.solvers.numpy.SciPyIVPSolver`.
            atol            (*float*) absolute tolerance of the integrator. Default is ``1e-12``.
            rtol            (*float*) relative tolerance of the integrator. Default is ``1e-9``.
            is_stiff        (*bool*) option to select whether the integration is a stiff problem or a non-stiff one. Default is ``False``.
            step_dim        (*bool*) number of steps to jump during integration. Higher values give faster results. Default is ``10``.
            ============    ====================================================
    func_controls: callable
        Function for the controls in the format ``func_controls(t)``.
    has_delay: bool
        Option to solve DDEs.
    func_delay: callable
        History function for first delay step in the format ``func_delay(t)``.
        This function is then internally replaced by the interpolated function for the subsequent steps.
    delay_interval: int
        Interval of the delay.

    ..note: In the presence of delay, the parameter ``'step_dim'`` is overriden by the delay interval.
    """

    default_solver_params = {
        'method': 'vode',
        'atol': 1e-12,
        'rtol': 1e-9,
        'is_stiff': False,
        'step_dim': 10
    }
    """dict: Default parameters of the solver."""

    def __init__(self,
        func,
        y0,
        T,
        solver_params:dict,
        func_controls,
        has_delay:bool,
        func_delay,
        delay_interval:int,
        backend:BaseBackend
    ):
        """Class constructor for BaseIVPSolver."""

        # set attributes
        self.func = func
        self.y0 = y0
        self.T = T
        self.func_controls = func_controls
        self.has_delay = has_delay
        self.func_delay = func_delay
        self.delay_interval = delay_interval
        self.backend = backend

        # validate attributes
        assert self.func_delay is not None if self.has_delay else True, "delay function cannot be ``None`` if parameter ``has_delay`` is ``True``"

        # frequently used variables
        self.shape_y0 = self.get_shape(
            tensor=self.y0
        )
        self.shape_T = self.get_shape(
            tensor=self.T
        )

        # set params
        self.solver_params = dict()
        for key in self.default_solver_params:
            self.solver_params[key] = solver_params.get(key, self.default_solver_params[key])
        # override step dimension with delay interval if DDE
        if self.has_delay and self.delay_interval != 0:
            self.solver_params['step_dim'] = self.delay_interval

        # validate params
        assert self.solver_params['method'] in self.solver_methods, "parameter ``method`` should be one of ``{}``".format(self.solver_methods.keys)
        assert type(self.solver_params['step_dim']) is int and self.solver_params['step_dim'] < self.shape_T[0], "parameters ``step_dim`` should be an integer with a value less than the total number of steps"
    
    @abstractmethod
    def integrate(self,
        y0,
        T_step,
        params
    ):
        """Method to take one integration step.
        
        Parameters
        ----------
        y0: Any
            Initial values of the variables.
        T_step: Any
            Times at which the results are returned.
        params: Any
            Parameters to pass to the function.
        
        Returns
        -------
        Y: Any
            Values of the variables at the given points of time.
        """

        raise NotImplementedError
    
    @abstractmethod
    def interpolate(self,
        T,
        Y
    ):
        """Method to take one interpolation step.
        
        Parameters
        ----------
        T: Any
            Times at which the values are calculated.
        Y: Any
            Values at the given times.

        Returns
        -------
        func: callable
            Interpolated functions.
        """

        raise NotImplementedError

    def get_tensor(self,
        array
    ):
        """Method to convert to a tensor.
        
        Parameters
        ----------
        array: Any
            Array.
        
        Returns
        -------
        tensor: Any
            Tensor
        """

        return self.backend.convert_to_tensor(
            array=array
        )
    
    def get_shape(self,
        tensor
    ) -> tuple:
        """Method to obtain the shape of a tensor.
        
        Parameters
        ----------
        tensor: Any
            Tensor.
        
        Returns
        -------
        shape: tuple
            Shape of the tensor.
        """

        return self.backend.shape(
            tensor=tensor
        )
    
    def get_buffer(self,
        tensor_0,
        tensor_1
    ):
        """Method to create an empty tensor from two tensors.
        
        Parameters
        ----------
        tensor_0: Any
            First tensor.
        tensor_1: Any
            Second tensor.
        
        Returns
        -------
        Y: Any
            Empty tensor with shape ``(*shape_0, *shape_1)``.
        """

        return self.backend.empty(
            shape=(
                *self.backend.shape(
                    tensor=tensor_0
                ),
                *self.backend.shape(
                    tensor=tensor_1
                )
            ),
            dtype='real')

    def step(self,
        y0,
        T_step,
        params
    ):
        """Method to take one integration step.
        
        Parameters
        ----------
        y0: Any
            Initial values of the variables.
        T_step: Any
            Times at which the results are returned.
        params: Any
            Parameters to pass to the function.

        Returns
        -------
        Y: Any
            Values of the variables at the given points of time.
        """

        # integrate
        _Y = self.integrate(
            y0=y0,
            T_step=T_step,
            params=params
        )

        # update delay function
        if self.has_delay:
            self.func_delay = self.interpolate(
                T=T_step,
                Y=_Y
            )

        return _Y

    def solve_ivp(self,
        y0,
        params,
        show_progress
    ):
        """Module to solve the IVP.

        Parameters
        ----------
        y0: Any
            Initial values of the variables.
        params: Any
            Parameters to pass to the function.
        show_progress: bool
            Option to show the progress.

        Returns
        -------
        Y: Any
            Values of the variables at all points of time.
        """

        # frequently used variables
        _step_dim = self.solver_params['step_dim']

        # initialize results
        Y = self.get_buffer(
            T_0=self.T_eval,
            T_1=y0
        )
        Y[0] = y0

        # evolve
        for i in tqdm(
            range(_step_dim, self.shape_T[0], _step_dim),
            desc="Progress",
            leave=False,
            mininterval=0.5,
            disable=not show_progress
        ):
            Y[i - _step_dim + 1:i + 1] = self.step(
                y0=Y[i - _step_dim],
                T_step=self.T[i - _step_dim:i + 1],
                params=params
            )[1:]

        return Y