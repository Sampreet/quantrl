#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module with base classes to interface different backends."""

__name__    = 'quantrl.backends.base'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-03-22"

# dependencies
from abc import ABC, abstractmethod
import numpy as np

class BaseBackend(ABC):
    """Backend to interface different NumPy-like libraries.
    
    Parameters
    ----------
    library: Any
        Library used by the backend.
    tensor_type: Any
        Tensor type for the backend.
    precision: str, default='double'
        Precision of the numerical values in the backend. Options are ``'single'`` and ``'double'``.
    """

    def __init__(self,
        library,
        tensor_type,
        precision:str='double'
    ):
        # validate parameters
        assert precision in ['single', 'double'], "parameter ``precision`` can be either ``'single'`` or ``'double'``."

        # set attributes
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
        if type(tensor) == self.tensor_type:
            if dtype is None or (dtype is not None and tensor.dtype == _dtype):
                return True
        return False

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

        # validate params
        assert dtype is None or dtype in ['integer', 'real', 'complex'], "parameter ``dtype`` can be either ``'integer'``, ``'real'`` or ``'complex'``."
        
        # default dtype is the real data-type
        if dtype is None:
            dtype = 'real'
        return self.dtypes['numpy' if numpy else 'typed'][self.precision][dtype]

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

        return self.library.eye(rows, (cols if cols is not None else rows), dtype=self.dtype_from_str(
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
        """Method to obtain the maximum value(s) of a typed tensor along an axis.
        
        Parameters
        ----------
        tensor: Any
            Given typed tensor.

        Returns
        -------
        max: Any
            Maximum value(s) of the typed tensor along the given axis.
        """

        return self.library.max(self.convert_to_typed(
            tensor=tensor
        ))