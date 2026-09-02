#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface JAX backend."""

__name__    = 'quantrl.backends.context_manager'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-10-09"
__updated__ = "2025-05-29"

# quantrl modules
from .base import BaseBackend

INSTANCES_BACKEND = {}

def get_instance_backend(
        library:str,
        precision:str='double',
        device:str='cuda',
) -> BaseBackend:
    """Method to obtain an instantiated backend.
    
    Parameters
    ----------
    library: str
        Name of the library for the backend.
        Options are ``'jax'``, ``'torch'`` and ``'numpy'``.
    precision: str, default='double'
        Precision of the numerical values in the backend.
        Options are ``'single'`` and ``'double'``.
    device: str, default='cuda'
        Device for the backend.
        Options are ``'cpu'`` and ``'cuda'``.

    Returns
    -------
    Backend: :class:`quantrl.backends.base.BaseBackend`
        The instantiated backend.
    """

    if library in INSTANCES_BACKEND:
        return INSTANCES_BACKEND[library]

    if 'jax' in library.lower():
        try:
            from .jax import JAXBackend
            INSTANCES_BACKEND['jax'] = JAXBackend(
                precision=precision,
            )
            library = 'jax'
            return INSTANCES_BACKEND[library]
        # use PyTorch if JAX is not installed
        except ImportError:
            print("JAX not installed, defaulting to PyTorch")
            library = 'torch'

    if 'torch' in library.lower():
        from .torch import TorchBackend
        INSTANCES_BACKEND['torch'] = TorchBackend(
            precision=precision,
            device=device,
        )
        library = 'torch'
        return INSTANCES_BACKEND[library]

    assert 'numpy' in library.lower(), \
        "parameter ``library`` can be either ``'jax'``, ``'torch'`` or ``'numpy'``"
    from .numpy import NumPyBackend
    INSTANCES_BACKEND['numpy'] = NumPyBackend(
        precision=precision,
    )
    library = 'numpy'
    return INSTANCES_BACKEND[library]
