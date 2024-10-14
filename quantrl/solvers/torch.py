#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface PyTorch-based solvers."""

__name__    = 'quantrl.solvers.torch'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-10-14"

# dependencies
from torchdiffeq import odeint

# quantrl modules
from ..backends.torch import TorchBackend
from .base import BaseIVPSolver

# TODO: Implement interpolation

class TorchDiffEqIVPSolver(BaseIVPSolver):
    """ODE and DDE solver using TorchDiffEq-based methods for initial-value problems.

    Available methods are ``'dopri8'``, ``'dopri5'``, ``'bosh3'``, ``'fehlberg2'`` and ``'adaptive_huen'``.
    Refer to :class:`quantrl.backends.base.BaseIVPSolver` for its implementation.
    """

    # attributes
    solver_methods = ['dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_huen']
    """list: TorchDiffEq-based methods."""

    def __init__(self,
        func,
        y_0,
        T,
        solver_params:dict,
        func_controls=None,
        has_delay:bool=False,
        func_delay=None,
        delay_interval:int=0,
        backend:TorchBackend=None
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
            backend=backend if backend is not None else TorchBackend(
                precision='double',
                device='cuda'
            )
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

        # integrate
        return odeint(
            func=lambda t, y: self.func(t, y, [params, self.func_controls, self.func_delay]),
            y0=y_0,
            t=T_step,
            atol=self.solver_params['atol'],
            rtol=self.solver_params['rtol'],
            method=self.solver_params['method'],
            options={}
        )

    def interpolate(self,
        T_step,
        Y
    ):
        raise NotImplementedError
