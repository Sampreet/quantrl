#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface JAX backend."""

__name__    = 'quantrl.backends.context_manager'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-10-09"
__updated__ = "2024-10-13"

# quantrl modules
from .base import BaseBackend

BACKEND_INSTANCES = {}

# TODO: validate arguments
def get_backend_instance(
        library:str,
        precision:str='double',
        device:str='gpu'
    ) -> BaseBackend:
    """Method to obtain an instantiated backend.
    
    Parameters
    ----------
    library: str
        Name of the library. Options are ``'jax'``, ``'numpy'`` and ``'torch'``.
    precision: str, default='double'
        Precision of the numerical values in the backend. Options are ``'single'`` and ``'double'``.
    device: str, default='gpu'
        Device for the backend. Options are ``'cpu'`` and ``'gpu'``.

    Returns
    -------
    backend: :class:`quantrl.backends.base.BaseBackend`
        The instantiated backend.
    """
    if library in BACKEND_INSTANCES:
        return BACKEND_INSTANCES[library]
    if 'jax' in library.lower():
        from .jax import JaxBackend
        BACKEND_INSTANCES['jax'] = JaxBackend(precision=precision)
        library = 'jax'
    elif 'torch' in library.lower():
        from .torch import TorchBackend
        BACKEND_INSTANCES['torch'] = TorchBackend(precision=precision, device=device)
        library = 'torch'
    else:
        assert 'numpy' in library.lower(), 'parameter `library` can be either `"jax"`, `"numpy"` or `"pytorch"`'
        from .numpy import NumPyBackend
        BACKEND_INSTANCES['numpy'] = NumPyBackend(precision=precision)
        library = 'numpy'
    return BACKEND_INSTANCES[library]
