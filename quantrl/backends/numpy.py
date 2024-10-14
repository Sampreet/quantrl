#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface NumPy backend."""

__name__    = 'quantrl.backends.numpy'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-10-13"

# dependencies
import numpy as np

# quantrl modules
from .base import BaseBackend

class NumPyBackend(BaseBackend):
    """Backend to interface the NumPy library.

    Parameters
    ----------
    precision: str, default='double'
        Precision of the numerical values in the backend. Options are ``'single'`` and ``'double'``.
    """
    def __init__(self,
        precision:str='double'
    ):
        # initialize BaseBackend
        super().__init__(
            name='numpy',
            library=np,
            tensor_type=np.ndarray,
            precision=precision
        )

    def convert_to_typed(self,
        tensor,
        dtype:str=None
    ) -> np.ndarray:
        return np.asarray(tensor, dtype=self.dtype_from_str(
            dtype=dtype,
            numpy=True
        ) if dtype is not None else None)

    def convert_to_numpy(self,
        tensor,
        dtype:str=None
    ) -> np.ndarray:
        return self.convert_to_typed(
            tensor=tensor,
            dtype=dtype
        )

    def generator(self,
        seed:int=None
    ) -> np.random.Generator:
        if self.seed_sequence is None:
            self.seed_sequence = self.get_seedsequence(seed)
        return np.random.default_rng(self.seed_sequence.spawn(1)[0])

    def integers(self,
        generator:np.random.Generator,
        shape:tuple,
        low:int=0,
        high:int=1000,
        dtype:str=None
    ):
        return generator.integers(low, high, shape, dtype=self.dtype_from_str(
            dtype=dtype
        ), endpoint=False)

    def normal(self,
        generator:np.random.Generator,
        shape:tuple,
        mean:float=0.0,
        std:float=1.0,
        dtype:str=None
    ) -> np.ndarray:
        return np.asarray(generator.normal(mean, std, shape), dtype=self.dtype_from_str(
            dtype=dtype
        ))

    def uniform(self,
        generator:np.random.Generator,
        shape:tuple,
        low:float=0.0,
        high:float=1.0,
        dtype:str=None
    ) -> np.ndarray:
        return np.asarray(generator.uniform(low, high, shape), dtype=self.dtype_from_str(
            dtype=dtype
        ))

    def transpose(self,
        tensor,
        axis_0:int=None,
        axis_1:int=None
    ) -> np.ndarray:
        if axis_0 is None or axis_1 is None:
            return self.convert_to_typed(
                tensor=tensor
            ).T

        # get swapped axes
        _shape = np.shape(tensor)
        _axes = np.arange(len(_shape))
        _axes[axis_1] = axis_0 % len(_shape)
        _axes[axis_0] = axis_1 % len(_shape)

        return np.transpose(tensor, axes=_axes)

    def repeat(self,
        tensor,
        repeats,
        axis
    ) -> np.ndarray:
        return np.repeat(tensor, repeats=repeats, axis=axis)

    def add(self,
        tensor_0,
        tensor_1,
        out
    ) -> np.ndarray:
        return np.add(tensor_0, tensor_1, out=out)

    def matmul(self,
        tensor_0,
        tensor_1,
        out
    ) -> np.ndarray:
        return np.matmul(tensor_0, tensor_1, out=out)

    def dot(self,
        tensor_0,
        tensor_1,
        out
    ) -> np.ndarray:
        return np.dot(tensor_0, tensor_1, out=out)

    def norm(self,
        tensor,
        axis
    ) -> np.ndarray:
        return np.linalg.norm(tensor, axis=axis)

    def concatenate(self,
        tensors:tuple,
        axis,
        out
    ) -> np.ndarray:
        return np.concatenate(tensors, axis=axis, out=out)

    def stack(self,
        tensors:tuple,
        axis,
        out
    ) -> np.ndarray:
        return np.stack(tensors, axis=axis, out=out)

    def update(self,
        tensor,
        indices,
        values
    ) -> np.ndarray:
        tensor[indices] = values
        return tensor

    def if_else(self,
        condition,
        func_true,
        func_false,
        args
    ):
        if condition:
            return func_true(args)
        return func_false(args)

    def iterate_i(self,
        func,
        iterations_i:int,
        Y,
        args:tuple=None
    ):
        for i in range(iterations_i):
            Y = func(i, Y, args)
        return Y
