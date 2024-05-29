#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with base classes to interface different solvers."""

__name__    = 'quantrl.solvers.base'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-05-29"

# dependencies
from abc import ABC, abstractmethod
from tqdm import tqdm

# quantrl modules
from ..backends.base import BaseBackend

# TODO: Support for JAX in solve_ivp

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

            ================    ====================================================
            key                 value
            ================    ====================================================
            method              (*str*) method used to solve the ODEs. Refer to the documentation of the inherited solvers. Default is ``'vode'`` from :class:`quantrl.solvers.numpy.SciPyIVPSolver`.
            atol                (*float*) absolute tolerance of the integrator. Default is ``1e-12``.
            rtol                (*float*) relative tolerance of the integrator. Default is ``1e-9``.
            is_stiff            (*bool*) option to select whether the integration is a stiff problem or a non-stiff one. Default is ``False``.
            step_interval       (*bool*) number of steps to jump during integration. Higher values give faster results. Default is ``10``.
            ================    ====================================================
    func_controls: callable
        Function for the controls in the format ``func_controls(t)``.
    has_delay: bool
        Option to solve DDEs.
    func_delay: callable
        History function for first delay step in the format ``func_delay(t)``.
        This function is then internally replaced by the interpolated function for the subsequent steps.
    delay_interval: int
        Interval of the delay.

    ..note: In the presence of delay, the parameter ``'step_interval'`` is overriden by the delay interval.
    """

    default_solver_params = {
        'method': 'vode',
        'atol': 1e-12,
        'rtol': 1e-9,
        'is_stiff': False,
        'step_interval': 10,
        'complex': False
    }
    """dict: Default parameters of the solver."""

    def __init__(self,
        func,
        y_0,
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
        self.y_0 = y_0
        self.T = T
        self.func_controls = func_controls
        self.has_delay = has_delay
        self.func_delay = func_delay
        self.delay_interval = delay_interval
        self.backend = backend

        # validate attributes
        assert self.func_delay is not None if self.has_delay else True, "delay function cannot be ``None`` if parameter ``has_delay`` is ``True``"

        # frequently used variables
        self.shape_y = self.backend.shape(
            tensor=self.y_0
        )
        self.shape_T = self.backend.shape(
            tensor=self.T
        )

        # set params
        self.solver_params = dict()
        for key in self.default_solver_params:
            self.solver_params[key] = solver_params.get(key, self.default_solver_params[key])
        # override step dimension with delay interval if DDE
        if self.has_delay and self.delay_interval != 0:
            self.solver_params['step_interval'] = self.delay_interval

        # validate params
        assert self.solver_params['method'] in self.solver_methods, "parameter ``method`` should be one of ``{}``".format(self.solver_methods)
        assert type(self.solver_params['step_interval']) is int and self.solver_params['step_interval'] < self.shape_T[0], "parameter ``step_interval`` should be an integer with a value less than the total number of steps"

        # step constants
        self.step_interval = self.solver_params['step_interval']
    
    @abstractmethod
    def integrate(self,
        T_step,
        y_0,
        params
    ):
        """Method to take one integration step.
        
        Parameters
        ----------
        T_step: Any
            Times at which the results are returned.
        y_0: Any
            Initial values of the variables.
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
        T_step,
        Y
    ):
        """Method to take one interpolation step.
        
        Parameters
        ----------
        T_step: Any
            Times at which the results are returned.
        Y: Any
            Values at the given times.

        Returns
        -------
        func: callable
            Interpolated functions.
        """

        raise NotImplementedError

    def step(self,
        T_step,
        y_0,
        params
    ):
        """Method to take one integration and interpolation step.
        
        Parameters
        ----------
        T_step: Any
            Times at which the results are returned.
        y_0: Any
            Initial values of the variables.
        params: Any
            Parameters to pass to the function.

        Returns
        -------
        Y: Any
            Values of the variables at the given points of time.
        """

        # integrate
        _Y = self.integrate(
            T_step=T_step,
            y_0=y_0,
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
        y_0,
        params,
        show_progress
    ):
        """Module to solve the IVP.

        Parameters
        ----------
        y_0: Any
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

        # initialize results
        Y = self.backend.empty(
            shape=(
                *self.backend.shape(
                    tensor=self.T
                ),
                *self.backend.shape(
                    tensor=y_0
                )
            ),
            dtype='real'
        )
        Y[0] = y_0

        # evolve
        for i in tqdm(
            range(self.step_interval, self.shape_T[0], self.step_interval),
            desc="Solving",
            leave=False,
            mininterval=0.5,
            disable=not show_progress
        ):
            Y[i - self.step_interval + 1:i + 1] = self.step(
                T_step=self.T[i - self.step_interval:i + 1],
                y_0=Y[i - self.step_interval],
                params=params
            )[1:]

        return Y