#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface JAX backend."""

__name__    = 'quantrl.solvers.context_manager'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-10-09"
__updated__ = "2025-05-11"

# quantrl modules
from .base import BaseIVPSolver

SOLVERS_IVP = {}

def get_IVP_solver(
        library:str
    ) -> BaseIVPSolver:
    """Method to obtain an IVP solver class.
    
    Parameters
    ----------
    library: str
        Name of the library. Options are ``'jax'``, ``'torch'`` and ``'numpy'``.

    Returns
    -------
    IVPSolver: :class:`quantrl.solvers.base.BaseIVPSolver`
        The IVP solver class.
    """
    if library in SOLVERS_IVP:
        return SOLVERS_IVP[library]

    if 'jax' in library.lower():
        try:
            from .jax import DiffraxIVPSolver
            SOLVERS_IVP['jax'] = DiffraxIVPSolver
            library = 'jax'
            return SOLVERS_IVP[library]
        # use PyTorch if JAX is not installed
        except ImportError:
            print("JAX not installed, defaulting to PyTorch")
            library = 'torch'

    if 'torch' in library.lower():
        from .torch import TorchDiffEqIVPSolver
        SOLVERS_IVP['torch'] = TorchDiffEqIVPSolver
        library = 'torch'
        return SOLVERS_IVP[library]

    assert 'numpy' in library.lower(), "parameter ``library`` can be either ``'jax'`, ``'torch'`` or ``'numpy'``"
    from .numpy import SciPyIVPSolver
    SOLVERS_IVP['numpy'] = SciPyIVPSolver
    library = 'numpy'
    return SOLVERS_IVP[library]
