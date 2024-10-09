#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface JAX backend."""

__name__    = 'quantrl.backends.context_manager'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-10-09"
__updated__ = "2024-10-09"

# quantrl modules
from .base import BaseBackend

BACKEND_INSTANCES = dict()

def get_backend_instance(
        library:str,
        precision:str='double',
        device:str='cuda'
    ) -> BaseBackend:
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