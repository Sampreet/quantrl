#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface JAX backend."""

__name__    = 'quantrl.backends.jax'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-03-18"

# dependencies
import numpy as np
import jax
import jaxlib.xla_extension as jxe
import jax.numpy as jnp

# quantrl modules
from .base import BaseBackend

# TODO: Implement buffers

class JaxBackend(BaseBackend):
    def __init__(self,
        precision:str='double'
    ):  
        # initialize BaseBackend
        super().__init__(
            lib=jnp,
            tensor_type=jxe.ArrayImpl,
            precision=precision
        )

        # enable 64-bit mode
        if self.precision == 'double':
            jax.config.update('jax_enable_x64', True)

        def numpy_transpose(
            tensor,
            axis_0:int=None,
            axis_1:int=None
        ):
            _shape = self.shape(
                tensor=tensor
            )
            _axes = np.arange(len(_shape))
            if axis_0 is not None and axis_1 is not None:
                _axes[axis_1] = axis_0 % len(_shape)
                _axes[axis_0] = axis_1 % len(_shape)
                return np.transpose(tensor, axes=_axes)
            else:
                return self.convert_to_tensor(
                    array=tensor
                ).T

        self.jit_transpose = jax.jit(
            fun=numpy_transpose,
            static_argnames=('axis_0', 'axis_1')
        )

        self.jit_repeat = jax.jit(
            fun=lambda tensor, repeats, axis: jnp.repeat(tensor, repeats, axis),
            static_argnames=('repeats', 'axis')
        )

        self.jit_add = jax.jit(
            fun=lambda tensor_0, tensor_1, out: jnp.add(tensor_0, tensor_1),
            donate_argnames='out'
        )

        self.jit_matmul = jax.jit(
            fun=lambda tensor_0, tensor_1, out: jnp.matmul(tensor_0, tensor_1),
            donate_argnames='out'
        )

        self.jit_dot = jax.jit(
            fun=lambda tensor_0, tensor_1, out: jnp.dot(tensor_0, tensor_1),
            donate_argnames='out'
        )

        self.jit_concatenate = jax.jit(
            fun=lambda tensors, axis, out: jnp.concatenate(tensors, axis),
            static_argnames='axis',
            donate_argnames='out'
        )

        self.jit_stack = jax.jit(
            fun=lambda tensors, axis, out: jnp.stack(tensors, axis),
            static_argnames='axis',
            donate_argnames='out'
        )

    def convert_to_tensor(self,
        array,
        dtype:str=None
    ):
        if type(array) == self.tensor_type:
            return array
        return jnp.array(array, dtype=self.dtype_from_str(
            mode='tensor',
            dtype=dtype
        ) if dtype is not None else None)

    def convert_to_array(self,
        tensor,
        dtype:str=None
    ) -> np.ndarray:
        return np.array(tensor, dtype=self.dtype_from_str(
            mode='array',
            dtype=dtype
        ) if dtype is not None else None)
    
    def normal(self,
        shape:tuple,
        mean:float=0.0,
        std:float=1.0,
        seed:int=None,
        dtype:str=None
    ):
        if seed is None:
            seed = np.random.randint(1000)
        return mean + std * jax.random.normal(jax.random.key(seed), shape, dtype=self.dtype_from_str(
            mode='tensor',
            dtype=dtype
        ))

    def transpose(self,
        tensor,
        axis_0:int=None,
        axis_1:int=None
    ):
        return jax.block_until_ready(self.jit_transpose(
            tensor=tensor,
            axis_0=axis_0,
            axis_1=axis_1
        ))

    def repeat(self,
        tensor,
        repeats,
        axis
    ):
        return jax.block_until_ready(self.jit_repeat(
            tensor=tensor,
            repeats=repeats,
            axis=axis
        ))

    def add(self,
        tensor_0,
        tensor_1,
        out
    ):
        return jax.block_until_ready(self.jit_add(
            tensor_0=tensor_0,
            tensor_1=tensor_1,
            out=out
        ))

    def matmul(self,
        tensor_0,
        tensor_1,
        out
    ):
        return jax.block_until_ready(self.jit_matmul(
            tensor_0=tensor_0,
            tensor_1=tensor_1,
            out=out
        ))

    def dot(self,
        tensor_0,
        tensor_1,
        out
    ):
        return jax.block_until_ready(self.jit_dot(
            tensor_0=tensor_0,
            tensor_1=tensor_1,
            out=out
        ))

    def concatenate(self,
        tensors:tuple,
        axis,
        out
    ):
        return jax.block_until_ready(self.jit_concatenate(
            tensors=tensors,
            axis=axis,
            out=out
        ))

    def stack(self,
        tensors:tuple,
        axis,
        out
    ):
        return jax.block_until_ready(self.jit_stack(
            tensors=tensors,
            axis=axis,
            out=out
        ))
    
    def update(self,
        tensor,
        indices,
        values
    ):
        return jax.block_until_ready(tensor.at[indices].set(values))