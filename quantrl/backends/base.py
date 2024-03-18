#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with base classes to interface different backends."""

__name__    = 'quantrl.backends.base'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-03-18"

# dependencies
from abc import ABC, abstractmethod
import numpy as np

class BaseBackend(ABC):
    """Backend to interface different NumPy-like libraries.
    
    Parameters
    ----------
    lib: Any
        Library used by the backend.
    tensor_type: Any
        Tensor type for the backend.
    precision: str, default='double'
        Precision of the numerical values in the backend. Options are ``'single'`` and ``'double'``.
    """

    def __init__(self,
        lib,
        tensor_type,
        precision:str='double'
    ):
        # validate parameters
        assert precision in ['single', 'double'], "parameter ``precision`` can be either ``'single'`` or ``'double'``."

        # set attributes
        self.lib = lib
        self.tensor_type = tensor_type
        self.precision = precision
        self.dtypes = {
            'tensor': {
                'single': {
                    'integer': self.lib.int32,
                    'real': self.lib.float32,
                    'complex': self.lib.complex64
                },
                'double': {
                    'integer': self.lib.int64,
                    'real': self.lib.float64,
                    'complex': self.lib.complex128
                }
            },
            'array': {
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
            },
        }

    @abstractmethod
    def convert_to_tensor(self,
        array,
        dtype:str=None
    ):
        """Method to obtain a typed tensor from a given array.
        
        Parameters
        ----------
        array: Any
            Given array.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type of the original array is returned.

        Returns
        -------
        tensor: Any
            Typed tensor.
        """

        raise NotImplementedError

    @abstractmethod
    def convert_to_array(self,
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
    def normal(self,
        shape:tuple,
        mean:float=0.0,
        std:float=1.0,
        seed:int=None,
        dtype:str=None
    ):
        """Method to obtain a typed tensor containing samples from a normal distribution.

        Parameters
        ----------
        shape: tuple
            Shape of the typed tensor.
        mean: float, default=0.0
            Mean of the distribution.
        std: float, default=1.0
            Standard deviation of the distribution.
        seed: Any, default=None
            Seed for the PRNG. If ``None``, a random seed is selected in ``[0, 1000)``.
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
        tensors: Any
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

    def dtype_from_str(self,
        mode:str,
        dtype:str=None
    ):
        """Method to obtain the data-type from a given string.
        
        Parameters
        ----------
        mode: str
            Mode of backend. Options are ``'tensor'`` for the selected backend or ``'array'`` for NumPy backend.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to real.

        Returns
        -------
        dtype: type
            Selected data-type.
        """

        # validate params
        assert dtype is None or dtype in ['integer', 'real', 'complex'], "parameter ``dtype`` can be either ``'integer'``, ``'real'`` or ``'complex'``."
        assert mode in ['tensor', 'array'], "parameter ``mode`` can be either ``'tensor'`` or ``'array'``"
        
        # default dtype is the backend's real data-type
        if dtype is None:
            return self.dtypes[mode][self.precision]['real']

        return self.dtypes[mode][self.precision][dtype]

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

        return self.lib.empty(shape, dtype=self.dtype_from_str(
            mode='tensor',
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

        return self.lib.zeros(shape, dtype=self.dtype_from_str(
            mode='tensor',
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

        return self.lib.ones(shape, dtype=self.dtype_from_str(
            mode='tensor',
            dtype=dtype
        ))

    def eye(self,
        rows:int,
        cols:int=None,
        dtype:str=None
    ):
        """Method to create an typed identity matrix.
        
        Parameters
        ----------
        rows: tuple
            Number of rows.
        cols: int, defualt=None
            Number of columns. if ``None``, this value is set equal to the number of rows.
        dtype: str, default=None
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to real.

        Returns
        -------
        tensor: Any
            Typed identity matrix.
        """

        return self.lib.eye(rows, (cols if cols is not None else rows), dtype=self.dtype_from_str(
            mode='tensor',
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

        return self.lib.diag(self.convert_to_tensor(
            array=tensor,
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
            Broad data-type. Options are ``'integer'``, ``'real'`` and ``'complex'``. If ``None``, the data-type is casted to integer (int).

        Returns
        -------
        tensor: Any
            Typed tensor of evenly-stepped values.
        """

        return self.lib.arange(start, stop, ssz, dtype=self.dtype_from_str(
            mode='tensor',
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

        return self.lib.linspace(start, stop, dim, dtype=self.dtype_from_str(
            mode='tensor',
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

        return tuple(self.convert_to_tensor(
            array=tensor
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

        return self.convert_to_tensor(
            array=tensor
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

        return self.convert_to_tensor(
            array=tensor
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

        return self.lib.real(self.convert_to_tensor(
            array=tensor
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

        return self.lib.imag(self.convert_to_tensor(
            array=tensor
        ))

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

        return self.lib.conj(self.convert_to_tensor(
            array=tensor
        ))

    def min(self,
        tensor,
        axis:int=None
    ):
        """Method to obtain the minimum value(s) of a typed tensor along an axis.
        
        Parameters
        ----------
        tensor: Any
            Given typed tensor.
        axis: int, default=None
            Axis along which the minimum value(s) is(are) obtained. If ``None``, the minimum value of the complete tensor is returned.

        Returns
        -------
        min: Any
            Minimum value(s) of the typed tensor along the given axis.
        """

        return self.lib.min(self.convert_to_tensor(
            array=tensor
        ), axis)

    def max(self,
        tensor,
        axis:int=None
    ):
        """Method to obtain the maximum value(s) of a typed tensor along an axis.
        
        Parameters
        ----------
        tensor: Any
            Given typed tensor.
        axis: int, default=None
            Axis along which the maximum value(s) is(are) obtained. If ``None``, the maximum value of the complete tensor is returned.

        Returns
        -------
        max: Any
            Maximum value(s) of the typed tensor along the given axis.
        """

        return self.lib.max(self.convert_to_tensor(
            array=tensor
        ), axis)