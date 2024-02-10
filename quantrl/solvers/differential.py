#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
"""Module to solve differential equations."""

__name__    = 'quantrl.solvers.differential'
__authors__ = ["Sampreet Kalita"]
__created__ = "2023-04-25"
__updated__ = "2024-01-16"

# dependencies
from scipy.interpolate import splev, splrep
from tqdm import tqdm
import numpy as np
import scipy.integrate as si

# TODO: Implement TorchDiffEqSolver

class SciPyIVPSolver(object):
    """ODE and DDE solver using SciPy-based methods for initial-value problems.

    Currently, the module only supports a single delay interval.
    
    Parameters
    ----------
    func: callable
        ODE/DDE function in the format ``func(t, y, args)``.
        The first element of ``args`` contains the delay function, the second element contains the function for the controls and the third contains the constant parameters.
    solver_params: dict
        Parameters of the solver.
        Currently supported options are:
            ========        ====================================================
            key             value
            ========        ====================================================
            't_min'         (*float*) minimum time at which integration starts. Default is ``0.0``.
            't_max'         (*float*) maximum time at which integration stops. Default is ``1000.0``.
            't_dim'         (*int*) number of values from ``'t_max'`` to ``'t_min'``, both inclusive. Default is ``10001``.
            'ode_method'    (*str*) method used to solve the ODEs. Available options are ``'BDF'``, ``'DOP853'``, ``'LSODA'``, ``'Radau'``, ``'RK23'``, ``'RK45'``, ``'dop853'``, ``'dopri5'``, ``'lsoda'``, ``'zvode'`` and ``'vode'`` (fallback). Refer to SciPy documentation for details of each method. Default is ``'vode'``.
            'ode_atol'      (*float*) absolute tolerance of the integrator. Default is ``1e-12``.
            'ode_rtol'      (*float*) relative tolerance of the integrator. Default is ``1e-6``.
            'ode_is_stiff'  (*bool*) option to select whether the integration is a stiff problem or a non-stiff one. Default is ``False``.
            'ode_step_dim'  (*bool*) number of steps to jump during integration. Higher values give faster results. Default is ``10``.
            ========        ====================================================
    func_controls: callable, default=None
        Function for the controls in the format ``func_controls(t)``.
    has_delay: bool, default=True
        Option to solve DDEs.
    func_delay: callable, default=None
        History function for first delay step in the format ``func_delay(t)``.
        This function is then internally replaced by the interpolated function for the subsequent steps.
    t_delay: float, default=0.0
        Exact value of delay time.

    ..note: In the presence of delay, the parameter ``'ode_step_dim'`` is overriden by the delay interval.
    """

    # attributes
    scipy_new_methods = ['BDF', 'DOP853', 'LSODA', 'Radau', 'RK23', 'RK45']
    """list: New Python-based methods availabile in :class:`scipy.integrate`."""
    scipy_old_methods = ['dop853', 'dopri5', 'lsoda', 'vode', 'zvode']
    """list: Old FORTRAN-based methods availabile in :class:`scipy.integrate`."""

    default_solver_params = {
        't_min': 0.0,
        't_max': 1000.0,
        't_dim': 10001,
        'ode_method': 'vode',
        'ode_atol': 1e-8,
        'ode_rtol': 1e-6,
        'ode_is_stiff':False,
        'ode_step_dim': 10
    }
    """dict: Default parameters of the solver."""
    
    def __init__(self,
        func,
        solver_params:dict,
        func_controls=None,
        has_delay:bool=False,
        func_delay=None,
        t_delay:float=0.0
    ):
        """Class constructor for SciPyIVPSolver."""

        # set functions
        self.func = func
        self.func_controls = func_controls
        self.func_delay = func_delay

        # set params
        self.solver_params = dict()
        for key in self.default_solver_params:
            self.solver_params[key] = solver_params.get(key, self.default_solver_params[key])
        # validate params
        assert self.solver_params['ode_method'] in self.scipy_old_methods or self.solver_params['ode_method'] in self.scipy_new_methods
        assert type(self.solver_params['ode_step_dim']) is int and self.solver_params['ode_step_dim'] < self.solver_params['t_dim']

        # initialize FORTRAN-based solver
        if self.solver_params['ode_method'] in self.scipy_old_methods:
            self.integrator = si.ode(self.func)
            self.integrator.set_integrator(
                name=self.solver_params['ode_method'],
                atol=self.solver_params['ode_atol'],
                rtol=self.solver_params['ode_rtol'],
                method='bdf' if self.solver_params['ode_is_stiff'] else 'adams'
            )

        # set up delay
        self.has_delay = has_delay
        assert self.func_delay is not None if self.has_delay else True
        # override evaluation times to integral multiple of delay times
        if self.has_delay and t_delay != 0.0:
            self.t_delay = t_delay
            divisor = int((self.solver_params['t_max'] - self.solver_params['t_min']) / self.t_delay)
            self.t_eval_max = self.t_delay * divisor + self.solver_params['t_min']
            self.t_eval_dim = self.solver_params['t_dim']
            self.ode_step_dim = int((self.solver_params['t_dim'] - 1) / divisor)
        # old steps
        else:
            self.t_eval_max = self.solver_params['t_max']
            self.t_eval_dim = self.solver_params['t_dim']
            self.ode_step_dim = self.solver_params['ode_step_dim']
        # set times
        self.T_eval = np.linspace(self.solver_params['t_min'], self.t_eval_max, self.t_eval_dim, dtype=np.float_)
        
    def step_ivp(self,
        y0,
        T_step,
        params=None
    ):
        """Module to take one step.
        
        Parameters
        ----------
        y0: float
            Initial values of the variables.
        T_step: :class:`numpy.ndarray`
            Times at which the results are returned.
        params: list
            Parameters to pass to the function.
        
        Returns
        -------
        Y: :class:`numpy.ndarray`
            Values of the variables at the given points of time.
        """

        # step arguments
        args = [self.func_delay, self.func_controls, params]

        # FORTRAN-based solvers
        if self.solver_params['ode_method'] in self.scipy_old_methods:
            _Y = np.empty((len(T_step), len(y0)), dtype=np.float_)
            _Y[0] = y0
            self.integrator.set_initial_value(
                y=y0,
                t=T_step[0]
            )
            self.integrator.set_f_params(args)
            for i in range(1, len(T_step)):
                _Y[i] = self.integrator.integrate(T_step[i])
        # Python-based methods
        else:
            _Y = np.transpose(si.solve_ivp(
                fun=self.func,
                y0=y0,
                t_span=[T_step[0], T_step[-1]],
                t_eval=T_step,
                method=self.solver_params['ode_method'],
                atol=self.solver_params['ode_atol'],
                rtol=self.solver_params['ode_rtol'],
                args=(args, )
            ).y)

        # update delay function
        if self.has_delay:
            b_spline = [splrep(T_step, _Y[:, j]) for j in range(_Y.shape[1])]
            self.func_delay = lambda t: np.array([splev(t, b_spline[j]) for j in range(_Y.shape[1])])

        return _Y
    
    def solve_ivp(self,
        y0,
        params=None,
        show_progress=False
    ):
        """Module to take one step.
        
        Parameters
        ----------
        y0: float
            Initial values of the variables.
        params: list
            Parameters to pass to the function.
        show_progress: bool
            Option to show the progress.
        
        Returns
        -------
        Y: :class:`numpy.ndarray`
            Values of the variables at all points of time.
        """

        # initialize results
        Y = np.empty((self.t_eval_dim, len(y0)), dtype=np.float_)
        Y[0] = y0

        # evolve
        for i in tqdm(
            range(self.ode_step_dim, self.t_eval_dim, self.ode_step_dim),
            desc="Progress (time)",
            leave=False,
            mininterval=0.5,
            disable=not show_progress
        ):
            Y[i - self.ode_step_dim + 1:i + 1] = self.step_ivp(
                y0=Y[i - self.ode_step_dim],
                T_step=self.T_eval[i - self.ode_step_dim:i + 1],
                params=params
            )[1:]

        return Y