#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to interface NumPy backend."""

__name__    = 'quantrl.backends.numpy'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-07-22"

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
            library=np,
            tensor_type=np.ndarray,
            precision=precision
        )
        # set seeder
        self.seeder = None

    def convert_to_typed(self,
        tensor,
        dtype:str=None
    ) -> np.ndarray:
        if self.is_typed(
            tensor=tensor,
            dtype=dtype
        ):
            return tensor
        return np.array(tensor, dtype=self.dtype_from_str(
            dtype=dtype
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
        if self.seeder is None:
            if seed is None:
                entropy = np.random.randint(1234567890)
            else:
                entropy = np.random.default_rng(seed).integers(0, 1234567890, (1, ))[0]
            self.seeder = np.random.SeedSequence(entropy)
        return np.random.default_rng(self.seeder.spawn(1)[0])

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