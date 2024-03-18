#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface NumPy backend."""

__name__    = 'quantrl.backends.numpy'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-03-18"

# dependencies
import numpy as np

# quantrl modules
from .base import BaseBackend

class NumPyBackend(BaseBackend):
    def __init__(self,
        precision:str='double'
    ):
        # initialize BaseBackend
        super().__init__(
            lib=np,
            tensor_type=np.ndarray,
            precision=precision
        )

    def convert_to_tensor(self,
        array,
        dtype:str=None
    ):
        if type(array) == self.tensor_type:
            return array
        return np.array(array, dtype=self.dtype_from_str(
            mode='tensor',
            dtype=dtype
        ) if dtype is not None else None)

    def convert_to_array(self,
        tensor,
        dtype:str=None
    ):
        return self.convert_to_tensor(
            array=tensor,
            dtype=dtype
        )
    
    def normal(self,
        shape:tuple,
        mean:float=0.0,
        std:float=1.0,
        seed:int=None,
        dtype:str=None
    ):
        if seed is None:
            seed = np.random.randint(1000)
        return np.asarray(np.random.default_rng(seed).normal(mean, std, shape), dtype=self.dtype_from_str(
            mode='tensor',
            dtype=dtype
        ))

    def transpose(self,
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

    def repeat(self,
        tensor,
        repeats,
        axis
    ):
        return np.repeat(tensor, repeats=repeats, axis=axis)

    def add(self,
        tensor_0,
        tensor_1,
        out
    ):
        return np.add(tensor_0, tensor_1, out=out)

    def matmul(self,
        tensor_0,
        tensor_1,
        out
    ):
        return np.matmul(tensor_0, tensor_1, out=out)

    def dot(self,
        tensor_0,
        tensor_1,
        out
    ):
        return np.dot(tensor_0, tensor_1, out=out)

    def concatenate(self,
        tensors:tuple,
        axis,
        out
    ):
        return np.concatenate(tensors, axis=axis, out=out)

    def stack(self,
        tensors:tuple,
        axis,
        out
    ):
        return np.stack(tensors, axis=axis, out=out)
    
    def update(self,
        tensor,
        indices,
        values
    ):
        tensor[indices] = values
        return tensor