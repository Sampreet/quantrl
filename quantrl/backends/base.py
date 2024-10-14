#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with base classes to interface different backends."""

__name__    = 'quantrl.backends.base'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-10-13"

# dependencies
from abc import ABC, abstractmethod

import numpy as np

class BaseBackend(ABC):
    """Backend to interface different NumPy-like libraries.

    Parameters
    ----------
    name: str
        Name of the backend.
    library: Any
        Numerical library used by the backend.
    tensor_type: Any
        Tensor type for the backend.
    precision: str, default='double'
        Precision of the numerical values in the backend. Options are ``'single'`` and ``'double'``.
    """

    def __init__(
        self,
        name:str='numpy',
        library:np=np,
        tensor_type:np.ndarray=np.ndarray,
        precision:str='double'
    ):
        # validate parameters
        assert precision in ['single', 'double'], "parameter ``precision`` can be either ``'single'`` or ``'double'``."

        # set attributes
        self.name = name
        self.library = library
        self.tensor_type = tensor_type
        self.precision = precision
        self.dtypes = {
            'typed': {
                'single': {
                    'integer': self.library.int32,
                    'real': self.library.float32,
                    'complex': self.library.complex64
                },
                'double': {
                    'integer': self.library.int64,
                    'real': self.library.float64,
                    'complex': self.library.complex128
                }
            },
            'numpy': {
                'single': {
                    'integer': np.int32,
                    'real': np.float32,
                    'complex': np.complex64
                },
                'double': {
                    'integer': np.int64,
                    'real': np.float64,
                    'complex': np.complex128
                }
            }
        }
        self.seed_sequence = None

    def is_typed(self,
        tensor,
        dtype:str=None
    ) -> bool:
        """Method to check if a tensor is a typed tensor of given dtype.

        Parameters
        ----------
        tensor: Any
            Given tensor.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is not checked.

        Returns
        -------
        is_typed: bool
            Whether the tensor is a typed tensor.
        """

        _dtype = self.dtype_from_str(
            dtype=dtype
        )
        if isinstance(tensor, self.tensor_type):
            if dtype is None or (dtype is not None and tensor.dtype == _dtype):
                return True
        return False

    def get_seedsequence(self,
        seed:int=None
    ) -> np.random.SeedSequence:
        """Method to obtain a SeedSequence object.
        
        Parameters
        ----------
        seed: int
            Initial seed to obtain the entropy.
        
        Returns
        -------
        seed_sequence: :class:`numpy.random.SeedSequence`
            The SeedSequence object.
        """
        if seed is None:
            entropy = np.random.randint(1234567890)
        else:
            entropy = np.random.default_rng(seed).integers(0, 1234567890, (1, ))[0]
        return np.random.SeedSequence(entropy)

    @abstractmethod
    def convert_to_typed(self,
        tensor,
        dtype:str=None
    ):
        """Method to obtain a typed tensor with given data-type from a numpy array or another typed tensor.

        Parameters
        ----------
        tensor: Any
            Given tensor.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type of the original array is returned.

        Returns
        -------
        tensor: Any
            Typed tensor.
        """

        raise NotImplementedError

    @abstractmethod
    def convert_to_numpy(self,
        tensor,
        dtype:str=None
    ) -> np.ndarray:
        """Method to obtain a NumPy array from a typed tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type of the original tensor is returned.

        Returns
        -------
        array: :class:`numpy.ndarray`
            NumPy array.
        """

        raise NotImplementedError

    @abstractmethod
    def generator(self,
        seed:int=None
    ):
        """Method to obtain a pseudo random number generator.

        Parameters
        ----------
        seed: Any, default=None
            Seed for the PRNG. If ``None``, a random seed is selected in ``[0, 1000)``.

        Returns
        -------
        generator: Any
            Pseudo random number generator.
        """

        raise NotImplementedError

    @abstractmethod
    def integers(self,
        generator,
        shape:tuple,
        low:int=0,
        high:int=1000,
        dtype:str=None
    ):
        """Method to obtain a typed tensor containing samples from a uniform distribution in the interval ``[low, high)``.

        Parameters
        ----------
        generator: Any
            Pseudo random number generator.
        shape: tuple
            Shape of the typed tensor.
        low: int, default=0
            Lowest value (inclusive).
        high: int, default=1000
            Highest value (exclusive).
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. the data-type is casted to real.

        Returns
        -------
        tensor: Any
            Typed tensor containing the samples.
        """

        raise NotImplementedError

    @abstractmethod
    def normal(self,
        generator,
        shape:tuple,
        mean:float=0.0,
        std:float=1.0,
        dtype:str=None
    ):
        """Method to obtain a typed tensor containing samples from a normal distribution.

        Parameters
        ----------
        generator: Any
            Pseudo random number generator.
        shape: tuple
            Shape of the typed tensor.
        mean: float, default=0.0
            Mean of the distribution.
        std: float, default=1.0
            Standard deviation of the distribution.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. the data-type is casted to real.

        Returns
        -------
        tensor: Any
            Typed tensor containing the samples.
        """

        raise NotImplementedError

    @abstractmethod
    def uniform(self,
        generator,
        shape:tuple,
        low:float=0.0,
        high:float=1.0,
        dtype:str=None
    ):
        """Method to obtain a typed tensor containing samples from a uniform distribution in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        generator: Any
            Pseudo random number generator.
        shape: tuple
            Shape of the typed tensor.
        low: float, default=0.0
            Lowest value (inclusive).
        high: float, default=1.0
            Highest value (exclusive).
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. the data-type is casted to real.

        Returns
        -------
        tensor: Any
            Typed tensor containing the samples.
        """

        raise NotImplementedError

    @abstractmethod
    def transpose(self,
        tensor,
        axis_0:int=None,
        axis_1:int=None
    ):
        """Method to transpose a typed tensor about two axes.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.
        axis_0: int, default=None
            First axis.
        axis_1: int, default=None
            Second axis.

        Returns
        -------
        tensor: Any
            Transposed typed tensor.
        """

        raise NotImplementedError

    @abstractmethod
    def repeat(self,
        tensor,
        repeats:int,
        axis:int
    ):
        """Method to repeat a typed tensor about a given axis.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.
        repeats: int
            Number of repetitions.
        axis: int
            Axis of repetition.

        Returns
        -------
        tensor: Any
            Repeated typed tensor.
        """

        raise NotImplementedError

    @abstractmethod
    def add(self,
        tensor_0,
        tensor_1,
        out
    ):
        """Method to add two typed tensors.

        Parameters
        ----------
        tensor_0: Any
            First typed tensor.
        tensor_1: Any
            Second typed tensor.
        out: Any
            Buffer to store the sum.

        Returns
        -------
        tensor: Any
            Addition of the two typed tensors.
        """

        raise NotImplementedError

    @abstractmethod
    def matmul(self,
        tensor_0,
        tensor_1,
        out
    ):
        """Method to obtain the matrix multiplication two typed tensors along the last two axes.

        Parameters
        ----------
        tensor_0: Any
            First typed tensor.
        tensor_1: Any
            Second typed tensor.
        out: Any
            Buffer to store the multiplication.

        Returns
        -------
        tensor: Any
            Multiplication of the two typed tensors.
        """

        raise NotImplementedError

    @abstractmethod
    def dot(self,
        tensor_0,
        tensor_1,
        out
    ):
        """Method to obtain the dot product of two typed tensors.

        Parameters
        ----------
        tensor_0: Any
            First typed tensor.
        tensor_1: Any
            Second typed tensor.
        out: Any
            Buffer to store the dot product.

        Returns
        -------
        tensor: Any
            Dot product of the two typed tensors.
        """

        raise NotImplementedError

    @abstractmethod
    def norm(self,
        tensor,
        axis
    ):
        """Method to obtain the norm of a typed tensor along a given axis.

        Parameters
        ----------
        tensor: Any
            Typed tensor.
        axis: int
            Axis for the norm.

        Returns
        -------
        tensor: Any
            Norm of the typed tensor.
        """

        raise NotImplementedError

    @abstractmethod
    def concatenate(self,
        tensors:tuple,
        axis,
        out
    ):
        """Method to concatenate multiple typed tensors along a given axis.

        Parameters
        ----------
        tensors: tuple
            Sequence of the tensors to concatenate.
        axis: int
            Axis of concatenation.
        out: Any
            Buffer to store the concatenation.

        Returns
        -------
        tensor: Any
            Concatenation of the typed tensors.
        """

        raise NotImplementedError

    @abstractmethod
    def stack(self,
        tensors:tuple,
        axis:int,
        out
    ):
        """Method to stack multiple typed tensors along a given axis.

        Parameters
        ----------
        tensors: tuple
            Typed tensors to stack.
        axis: int
            Axis for stacking.

        Returns
        -------
        tensor: Any
            Stacked tensor.
        """

        return NotImplementedError

    @abstractmethod
    def update(self,
        tensor,
        indices,
        values
    ):
        """Method to update selected indices of a typed tensor with given values.

        Parameters
        ----------
        tensor: Any
            Typed tensor.
        indices: Any
            Indices to update.
        values: Any
            Updated values.

        Returns
        -------
        tensor: Any
            Updated typed tensor.
        """

        raise NotImplementedError

    @abstractmethod
    def if_else(self,
        condition,
        func_true,
        func_false,
        args
    ):
        """Method to execute conditional statements.

        Parameters
        ----------
        condition: bool
            Condition to check.
        func_true: callable
            Function to call when the condition is true.
        func_false: callable
            Function to call when the condition is False.
        args: tuple
            Arguments for the functions.

        Returns
        -------
        tensor: Any
            Output of the condition.
        """

        raise NotImplementedError

    @abstractmethod
    def iterate_i(self,
        func,
        iterations_i:int,
        Y,
        args:tuple=None
    ):
        """Method to iterate over a single variable.

        Parameters
        ----------
        func: callable
            Function to iterate formatted as ``func(i, *args, *kwargs)``, where i is the index of the iteration.
        iterations_i: int
            Number of iterations in the first variable. This results in interation indices in the open interval ``[0, iterations_i)``.
        Y: Any
            The tensor which is updated at each iteration, with ``Y[0]`` containing the initial values.
        args: tuple
            Arguments for the iteration.

        Returns
        -------
        Y: Any
            Updated tensor.
        """

        raise NotImplementedError

    def dtype_from_str(self,
        dtype:str=None,
        numpy:bool=False
    ):
        """Method to obtain the data-type from a string.

        Parameters
        ----------
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to real.
        numpy: bool, default=False
            Option to use NumPy data-types.

        Returns
        -------
        dtype: type
            Selected data-type.
        """
        # default dtype is the real data-type
        if dtype is None or dtype not in ['integer', 'real', 'complex']:
            dtype = 'real'
        return self.dtypes['numpy' if numpy else 'typed'][self.precision][dtype]

    def jit_transpose(self, tensor, axis_0, axis_1):
        """Method to JIT-compile transposition."""
        return self.transpose(tensor, axis_0, axis_1)

    def jit_repeat(self, tensor, repeats, axis):
        """Method to JIT-compile repitition."""
        return self.repeat(tensor, repeats, axis)

    def jit_add(self, tensor_0, tensor_1, out):
        """Method to JIT-compile addition."""
        return self.add(tensor_0, tensor_1, out=out)

    def jit_matmul(self, tensor_0, tensor_1, out):
        """Method to JIT-compile matrix multiplication."""
        return self.matmul(tensor_0, tensor_1, out)

    def jit_dot(self, tensor_0, tensor_1, out):
        """Method to JIT-compile dot product."""
        return self.dot(tensor_0, tensor_1, out)

    def jit_concatenate(self, tensors, axis, out):
        """Method to JIT-compile concatenation."""
        return self.concatenate(tensors, axis, out)

    def jit_stack(self, tensors, axis, out):
        """Method to JIT-compile stacking."""
        return self.stack(tensors, axis, out)

    def jit_update(self, tensor, indices, values):
        """Method to JIT-compile updation."""
        return self.update(tensor, indices, values)

    def empty(self,
        shape:tuple,
        dtype:str=None
    ):
        """Method to create an empty typed tensor.

        Parameters
        ----------
        shape: tuple
            Shape of the tensor.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to real.

        Returns
        -------
        tensor: Any
            Empty typed tensor.
        """

        return self.library.empty(shape, dtype=self.dtype_from_str(
            dtype=dtype
        ))

    def zeros(self,
        shape:tuple,
        dtype:str=None
    ):
        """Method to create a typed tensor of zeros.

        Parameters
        ----------
        shape: tuple
            Shape of the tensor.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to real.

        Returns
        -------
        tensor: Any
            Typed tensor of zeros.
        """

        return self.library.zeros(shape, dtype=self.dtype_from_str(
            dtype=dtype
        ))

    def ones(self,
        shape:tuple,
        dtype:str=None
    ):
        """Method to create a typed tensor of ones.

        Parameters
        ----------
        shape: tuple
            Shape of the tensor.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to real.

        Returns
        -------
        tensor: Any
            Typed tensor of ones.
        """

        return self.library.ones(shape, dtype=self.dtype_from_str(
            dtype=dtype
        ))

    def eye(self,
        N:int,
        M:int=None,
        dtype:str=None
    ):
        """Method to create an typed identity matrix.

        Parameters
        ----------
        N: tuple
            Number of rows.
        M: int, defualt=None
            Number of columns. if ``None``, this value is set equal to the number of rows.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to real.

        Returns
        -------
        tensor: Any
            Typed identity matrix.
        """

        return self.library.eye(N, (M if M is not None else N), dtype=self.dtype_from_str(
            dtype=dtype
        ))

    def diag(self,
        tensor,
        dtype:str=None
    ):
        """Method to create an typed diagonal matrix.

        Parameters
        ----------
        tensor: tuple
            Elements of the diagonal.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type of the original tensor is returned.

        Returns
        -------
        tensor: Any
            Typed diagonal matrix.
        """

        return self.library.diag(self.convert_to_typed(
            tensor=tensor,
            dtype=dtype
        ))

    def arange(self,
        start:float,
        stop:float,
        ssz:float,
        dtype:str=None
    ):
        """Method to create a typed tensor of evenly-stepped values from ``start`` (included) to ``stop`` (excluded).

        Parameters
        ----------
        start: float
            starting point of the interval.
        stop: float
            Stopping point of the interval.
        ssz: float
            Size of the steps.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to integer.

        Returns
        -------
        tensor: Any
            Typed tensor of evenly-stepped values.
        """

        return self.library.arange(start, stop, ssz, dtype=self.dtype_from_str(
            dtype=dtype if dtype is not None else 'integer'
        ))

    def linspace(self,
        start:float,
        stop:float,
        dim:int,
        dtype:str=None
    ):
        """Method to create a typed tensor of linearly-spaced values from ``start`` to ``stop``, both inclusive.

        Parameters
        ----------
        start: float
            Starting point of the interval.
        stop: float
            Stopping point of the interval.
        dim: int
            Dimension of the tensor.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to real.

        Returns
        -------
        tensor: Any
            Typed tensor of linearly-spaced values.
        """

        return self.library.linspace(start, stop, dim, dtype=self.dtype_from_str(
            dtype=dtype
        ))

    def shape(self,
        tensor
    ) -> tuple:
        """Method to obtain the shape of a tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        shape: tuple
            Shape of the typed tensor.
        """

        return tuple(self.convert_to_typed(
            tensor=tensor
        ).shape)

    def reshape(self,
        tensor,
        shape:tuple
    ):
        """Method to reshape a typed tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.
        shape: tuple
            Shape of the new tensor.

        Returns
        -------
        tensor: Any
            Typed tensor with given shape.
        """

        return self.convert_to_typed(
            tensor=tensor
        ).reshape(shape)

    def flatten(self,
        tensor
    ):
        """Method to flatten typed tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        tensor: Any
            Flattened typed tensor.
        """

        return self.convert_to_typed(
            tensor=tensor
        ).flatten()

    def real(self,
        tensor
    ):
        """Method to obtain the real components of a complex typed tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        tensor: Any
            Real components of the complex typed tensor.
        """

        return self.library.real(self.convert_to_typed(
            tensor=tensor
        ))

    def imag(self,
        tensor
    ):
        """Method to obtain the imaginary components of a complex typed tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        tensor: Any
            Imaginary components of the complex typed tensor.
        """

        return self.library.imag(self.convert_to_typed(
            tensor=tensor
        ))

    def sqrt(self,
        tensor
    ):
        """Method to obtain the square root of a typed tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        tensor: Any
            Square root of the typed tensor.
        """

        return self.library.sqrt(self.convert_to_typed(
            tensor=tensor
        ))

    def sum(self,
        tensor,
        axis
    ):
        """Method to obtain the sum of a typed tensor along a given axis.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.
        axis: Any
            Axis along which the sum is to be calculated.

        Returns
        -------
        tensor: Any
            Sum of the typed tensor along the axis.
        """

        return self.library.sum(self.convert_to_typed(
            tensor=tensor
        ), axis)

    def cumsum(self,
        tensor,
        axis
    ):
        """Method to obtain the cumulative sum of a typed tensor along a given axis.

        Parameters
        ----------
        tensors: Any
            Given typed tensor.
        axis: Any
            Axis along which the sum is to be calculated.

        Returns
        -------
        tensor: Any
            Cumulative um of the typed tensor along the axis.
        """

        return self.library.cumsum(self.convert_to_typed(
            tensor=tensor
        ), axis)

    def conj(self,
        tensor
    ):
        """Method to obtain the complex conjugate of a typed tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        tensor: Any
            Complex conjugate of the typed tensor.
        """

        return self.library.conj(self.convert_to_typed(
            tensor=tensor
        ))

    def min(self,
        tensor
    ):
        """Method to obtain the minimum value(s) of a typed tensor along an axis.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        min: Any
            Minimum value(s) of the typed tensor along the given axis.
        """

        return self.library.min(self.convert_to_typed(
            tensor=tensor
        ))

    def max(self,
        tensor
    ):
        """Method to obtain the maximum value(s) of a typed tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        max: Any
            Maximum value(s) of the typed tensor.
        """

        return self.library.max(self.convert_to_typed(
            tensor=tensor
        ))

    def argmax(self,
        tensor
    ):
        """Method to obtain the argument of the maximum value(s) of a typed tensor.

        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        argmax: Any
            Argument of the maximum value(s) of the typed tensor.
        """

        return self.library.argmax(self.convert_to_typed(
            tensor=tensor
        ))
