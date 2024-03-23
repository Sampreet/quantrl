#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface JAX-based solvers."""

__name__    = 'quantrl.solvers.jax'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-03-23"

# dependencies
import jax
import diffrax as dfx

# quantrl modules
from ..backends.jax import JaxBackend
from .base import BaseIVPSolver

# TODO: Implement interpolation

class DiffraxIVPSolver(BaseIVPSolver):
    """ODE and DDE solver using Diffrax-based methods for initial-value problems.

    Available methods are ``'dopri8'``, ``'dopri5'``, and ``'tsit5'``.
    Refer to :class:`quantrl.backends.base.BaseIVPSolver` for its implementation.
    """

    # attributes
    solver_methods = ['dopri5', 'dopri8', 'tsit5']
    """list: Diffrax-based methods."""

    def __init__(self,
        func,
        y_0,
        T,
        solver_params:dict,
        func_controls=None,
        has_delay:bool=False,
        func_delay=None,
        delay_interval:int=0,
        backend:JaxBackend=None
    ):
        # initialize BaseIVPSolver
        super().__init__(
            func=jax.jit(func),
            y_0=y_0,
            T=T,
            solver_params=solver_params,
            func_controls=jax.jit(func_controls) if func_controls is not None else None,
            has_delay=has_delay,
            func_delay=jax.jit(func_delay) if func_delay is not None else None,
            delay_interval=delay_interval,
            backend=backend if backend is not None else JaxBackend(
                precision='double'
            )
        )

        # initialize solver
        self.term = dfx.ODETerm(self.func)
        self.solver = {
            'dopri5': dfx.Dopri5,
            'dopri8': dfx.Dopri8,
            'tsit5': dfx.Tsit5
        }.get(self.solver_params['method'], dfx.Dopri5)()

    def integrate(self,
        T_step,
        y_0,
        params=None
    ):
        # convert to tensor
        y_0 = self.backend.convert_to_typed(
            tensor=y_0
        )
    
        # integrate
        return dfx.diffeqsolve(
            terms=self.term,
            solver=self.solver,
            t0=T_step[0],
            t1=T_step[-1],
            dt0=T_step[1] - T_step[0],
            y0=y_0,
            args=[params, self.func_controls, self.func_delay],
            saveat=dfx.SaveAt(ts=T_step),
            stepsize_controller=dfx.PIDController(
                atol=self.solver_params['atol'],
                rtol=self.solver_params['rtol']
            ) 
        ).ys
    
    def interpolate(self,
        T_step,
        Y
    ):
        raise NotImplementedError