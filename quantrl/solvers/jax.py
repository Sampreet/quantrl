#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface JAX-based solvers."""

__name__    = 'quantrl.solvers.jax'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-03-17"

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
        y0,
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
            func=func,
            y0=y0,
            T=T,
            solver_params=solver_params,
            func_controls=func_controls,
            has_delay=has_delay,
            func_delay=func_delay,
            delay_interval=delay_interval,
            backend=backend if backend is not None else JaxBackend(
                precision='double'
            )
        )

        # initialize solver
        self.term = dfx.ODETerm(jax.jit(self.func))
        self.solver = {
            'dopri5': dfx.Dopri5,
            'dopri8': dfx.Dopri8,
            'tsit5': dfx.Tsit5
        }.get(self.solver_params['method'], dfx.Dopri5)()

    def integrate(self,
        y0,
        T_step,
        params=None
    ):
        # convert to tensor
        y0 = self.get_tensor(
            array=y0
        )
    
        # integrate
        return dfx.diffeqsolve(
            terms=self.term,
            solver=self.solver,
            t0=T_step[0],
            t1=T_step[-1],
            dt0=T_step[1] - T_step[0],
            y0=y0,
            args=[params, self.func_controls, self.func_delay],
            saveat=dfx.SaveAt(ts=T_step),
            stepsize_controller=dfx.PIDController(
                atol=self.solver_params['atol'],
                rtol=self.solver_params['rtol']
            ) 
        ).ys
    
    def interpolate(self,
        T, Y
    ):
        raise NotImplementedError