#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface JAX backend."""

__name__    = 'quantrl.solvers.context_manager'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-10-09"
__updated__ = "2024-10-14"

# quantrl modules
from .base import BaseIVPSolver

IVP_SOLVERS = {}

def get_IVP_solver(
        library:str
    ) -> BaseIVPSolver:
    """Method to obtain an IVP solver class.
    
    Parameters
    ----------
    library: str
        Name of the library. Options are ``'jax'``, ``'numpy'`` and ``'torch'``.

    Returns
    -------
    IVPSolver: :class:`quantrl.solvers.base.BaseIVPSolver`
        The IVP solver class.
    """
    if library in IVP_SOLVERS:
        return IVP_SOLVERS[library]
    if 'jax' in library.lower():
        from .jax import DiffraxIVPSolver
        IVP_SOLVERS['jax'] = DiffraxIVPSolver
        library = 'jax'
    elif 'torch' in library.lower():
        from .torch import TorchDiffEqIVPSolver
        IVP_SOLVERS['torch'] = TorchDiffEqIVPSolver
        library = 'torch'
    else:
        assert 'numpy' in library.lower(), 'parameter `library` can be either `"jax"`, `"numpy"` or `"pytorch"`'
        from .numpy import SciPyIVPSolver
        IVP_SOLVERS['numpy'] = SciPyIVPSolver
        library = 'numpy'
    return IVP_SOLVERS[library]
