#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface NumPy-based solvers."""

__name__    = 'quantrl.solvers.numpy'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-03-23"

# dependencies
from scipy.interpolate import splev, splrep
import scipy.integrate as si

# quantrl modules
from ..backends.numpy import NumPyBackend
from .base import BaseIVPSolver

class SciPyIVPSolver(BaseIVPSolver):
    """ODE and DDE solver using SciPy-based methods for initial-value problems.

    Available methods are ``'BDF'``, ``'DOP853'``, ``'LSODA'``, ``'Radau'``, ``'RK23'``, ``'RK45'``, ``'dop853'``, ``'dopri5'``, ``'lsoda'``, ``'zvode'`` and ``'vode'``.
    Refer to :class:`quantrl.backends.base.BaseIVPSolver` for its implementation.
    """

    # attributes
    scipy_new_methods = ['BDF', 'DOP853', 'LSODA', 'Radau', 'RK23', 'RK45']
    """list: New Python-based methods availabile in :class:`scipy.integrate`."""
    scipy_old_methods = ['dop853', 'dopri5', 'lsoda', 'vode', 'zvode']
    """list: Old FORTRAN-based methods availabile in :class:`scipy.integrate`."""
    solver_methods = scipy_new_methods + scipy_old_methods
    """list: SciPy-based methods availabile in :class:`scipy.integrate`."""

    def __init__(self,
        func,
        y_0,
        T,
        solver_params:dict,
        func_controls=None,
        has_delay:bool=False,
        func_delay=None,
        delay_interval:int=0,
        backend:NumPyBackend=None
    ):
        # initialize BaseIVPSolver
        super().__init__(
            func=func,
            y_0=y_0,
            T=T,
            solver_params=solver_params,
            func_controls=func_controls,
            has_delay=has_delay,
            func_delay=func_delay,
            delay_interval=delay_interval,
            backend=backend if backend is not None else NumPyBackend(
                precision='double'
            )
        )

        # flatten function for integration
        self.func_flat = self.func
        self.is_y_flat = True
        if len(self.shape_y) > 1:
            self.func_flat = lambda t, y, args: self.backend.flatten(
                tensor=self.func(
                    t=t,
                    y=self.backend.reshape(
                        tensor=y,
                        shape=self.shape_y
                    ),
                    args=args
                )
            )
            self.is_y_flat = False

        # initialize FORTRAN-based solver
        if self.solver_params['method'] in self.scipy_old_methods:
            # initialize solver
            self.integrator = si.ode(self.func_flat)
            self.integrator.set_integrator(
                name=self.solver_params['method'],
                atol=self.solver_params['atol'],
                rtol=self.solver_params['rtol'],
                method='bdf' if self.solver_params['is_stiff'] else 'adams'
            )

    def integrate(self,
        T_step,
        y_0,
        params=None
    ):
        # convert to tensor
        y_0 = self.backend.convert_to_typed(
            tensor=y_0
        )

        # flatten
        y_0_flat = y_0
        if not self.is_y_flat:
            y_0_flat = self.backend.flatten(
                tensor=y_0
            )

        # integrate
        _Y_flat = self.integrate_flat(
            y_0_flat=y_0_flat,
            T_step=T_step,
            args=[params, self.func_controls, self.func_delay]
        )

        # reshape
        return self.backend.reshape(
            tensor=_Y_flat,
            shape=(len(T_step), *self.shape_y)
        )

    def integrate_flat(self,
        T_step,
        y_0_flat,
        args:tuple
    ):
        """Method to take one integration step.
        
        Parameters
        ----------
        T_step: Any
            Times at which the results are returned.
        y_0_flat: Any
            Flattened initial values of the variables.
        args: tuple
            Actions, control function and delay function.
        
        Returns
        -------
        Y: Any
            Values of the variables at the given points of time.
        """

        # integrate using FORTRAN-based solvers
        if self.solver_params['method'] in self.scipy_old_methods:
            _Y_flat = self.backend.empty(
                shape=(
                    *self.backend.shape(
                        tensor=T_step
                    ),
                    *self.backend.shape(
                        tensor=y_0_flat
                    )
                ),
                dtype='real'
            )
            _Y_flat[0] = y_0_flat
            self.integrator.set_initial_value(
                y=y_0_flat,
                t=T_step[0]
            )
            self.integrator.set_f_params(args)
            for i in range(1, len(T_step)):
                _Y_flat[i] = self.integrator.integrate(T_step[i])
        
        # integrate using Python-based solvers
        else:
            _Y_flat = self.backend.transpose(
                tensor=si.solve_ivp(
                    fun=self.func_flat,
                    y0=y_0_flat,
                    t_span=[T_step[0], T_step[-1]],
                    t_eval=T_step,
                    method=self.solver_params['method'],
                    atol=self.solver_params['atol'],
                    rtol=self.solver_params['rtol'],
                    args=(args, )
                ).y
            )

        return _Y_flat
    
    def interpolate(self,
        T_step,
        Y
    ):
        _shape = self.backend.shape(
            tensor=Y
        )[1]
        b_spline = [splrep(T_step, Y[:, j]) for j in range(_shape)]
        return lambda t: [splev(t, b_spline[j]) for j in range(_shape)]