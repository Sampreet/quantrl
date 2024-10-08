#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface JAX backend."""

__name__    = 'quantrl.backends.jax'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-04-25"

# dependencies
from inspect import getfullargspec
import numpy as np
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jxe

# quantrl modules
from .base import BaseBackend

# TODO: Implement buffers

class JaxBackend(BaseBackend):
    def __init__(self,
        precision:str='double'
    ):  
        # initialize BaseBackend
        super().__init__(
            library=jnp,
            tensor_type=jxe.ArrayImpl,
            precision=precision
        )

        # enable 64-bit mode
        if self.precision == 'double':
            jax.config.update('jax_enable_x64', True)

        # set key
        self.key = None

        def numpy_transpose(
            tensor,
            axis_0:int=None,
            axis_1:int=None
        ):
            # get swapped axes
            _shape = self.shape(
                tensor=tensor
            )
            _axes = np.arange(len(_shape))
            if axis_0 is not None and axis_1 is not None:
                _axes[axis_1] = axis_0 % len(_shape)
                _axes[axis_0] = axis_1 % len(_shape)
                return np.transpose(tensor, axes=_axes)
            else:
                return self.convert_to_typed(
                    tensor=tensor
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

    def convert_to_typed(self,
        tensor,
        dtype:str=None
    ) -> jxe.ArrayImpl:
        if self.is_typed(
            tensor=tensor,
            dtype=dtype
        ):
            return tensor
        return jax.block_until_ready(jnp.array(tensor, dtype=self.dtype_from_str(
            dtype=dtype
        ) if dtype is not None else None))

    def convert_to_numpy(self,
        tensor,
        dtype:str=None
    ) -> np.ndarray:
        return jax.block_until_ready(np.array(tensor, dtype=self.dtype_from_str(
            dtype=dtype,
            numpy=True
        ) if dtype is not None else None))

    def generator(self,
        seed:int=None
    ):
        if self.key is None:
            if seed is None:
                seed = np.random.randint(1000)
            self.key = jax.block_until_ready(jax.random.key(seed))
        self.key, key = jax.random.split(self.key)
        return key

    def integers(self,
        generator,
        shape:tuple,
        low:int=0,
        high:int=1000,
        dtype:str=None
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(jax.random.randint(generator, shape, low, high, dtype=self.dtype_from_str(
            dtype=dtype
        )))

    def normal(self,
        generator,
        shape:tuple,
        mean:float=0.0,
        std:float=1.0,
        dtype:str=None
    ) -> jxe.ArrayImpl:
        return mean + std * jax.block_until_ready(jax.random.normal(generator, shape, dtype=self.dtype_from_str(
            dtype=dtype
        )))

    def uniform(self,
        generator,
        shape:tuple,
        low:float=0.0,
        high:float=1.0,
        dtype:str=None
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(jax.random.uniform(generator, shape, minval=low, maxval=high, dtype=self.dtype_from_str(
            dtype=dtype
        )))

    def transpose(self,
        tensor,
        axis_0:int=None,
        axis_1:int=None
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(self.jit_transpose(
            tensor=tensor,
            axis_0=axis_0,
            axis_1=axis_1
        ))

    def repeat(self,
        tensor,
        repeats,
        axis
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(self.jit_repeat(
            tensor=tensor,
            repeats=repeats,
            axis=axis
        ))

    def add(self,
        tensor_0,
        tensor_1,
        out
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(self.jit_add(
            tensor_0=tensor_0,
            tensor_1=tensor_1,
            out=out
        ))

    def matmul(self,
        tensor_0,
        tensor_1,
        out
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(self.jit_matmul(
            tensor_0=tensor_0,
            tensor_1=tensor_1,
            out=out
        ))

    def dot(self,
        tensor_0,
        tensor_1,
        out
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(self.jit_dot(
            tensor_0=tensor_0,
            tensor_1=tensor_1,
            out=out
        ))

    def norm(self,
        tensor,
        axis
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(jnp.linalg.norm(tensor, axis=axis))

    def concatenate(self,
        tensors:tuple,
        axis,
        out
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(self.jit_concatenate(
            tensors=tensors,
            axis=axis,
            out=out
        ))

    def stack(self,
        tensors:tuple,
        axis,
        out
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(self.jit_stack(
            tensors=tensors,
            axis=axis,
            out=out
        ))
    
    def update(self,
        tensor,
        indices,
        values
    ) -> jxe.ArrayImpl:
        return jax.block_until_ready(tensor.at[indices].set(values))
    
    def if_else(self,
        condition,
        func_true,
        func_false,
        args
    ):
        return jax.block_until_ready(jax.lax.cond(condition, func_true, func_false, (args, )))

    def iterate_i(self,
        func,
        iterations_i:int,
        Y,
        args:tuple=None
    ) -> jxe.ArrayImpl:
        # convert to comapatible function
        _func_args = getfullargspec(func).args
        if (_func_args[0] == 'self' and len(_func_args) > 3) or len(_func_args) > 2:
            body_func = lambda i, state: (func(i, *state), *state[1:])
        else:
            body_func = func

        # loop and return typed tensor
        return jax.block_until_ready(jax.lax.fori_loop(0, iterations_i, body_func, (self.convert_to_typed(
            tensor=Y
        ), args))[0])