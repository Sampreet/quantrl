#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to implement PyTorch backend."""

__name__    = 'quantrl.backends.torch'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-03-18"

# dependencies
import numpy as np
import torch

# quantrl modules
from .base import BaseBackend

class TorchBackend(BaseBackend):
    def __init__(self,
        precision:str='double',
        device:str='cuda'
    ):
        # initialize BaseBackend
        super().__init__(
            lib=torch,
            tensor_type=torch.Tensor,
            precision=precision
        )

        # set default device
        assert 'cpu' in device or 'cuda' in device, "Invalid precision opted, options are ``'cpu'`` and ``'cuda'``."
        if 'cuda' in device and not torch.cuda.is_available():
            print("CUDA not available, defaulting to ``'cpu'``")
            device = 'cpu'
        torch.set_default_device(device)
        self.device = device

    def convert_to_tensor(self,
        array,
        dtype:str=None
    ):
        if type(array) == self.tensor_type:
            return array
        return torch.tensor(array, dtype=self.dtype_from_str(
            mode='tensor',
            dtype=dtype
        ) if dtype is not None else None)

    def convert_to_array(self,
        tensor,
        dtype:str=None
    ) -> np.ndarray:
        return np.asarray(tensor.detach().cpu().numpy() if self.device == 'cuda' else tensor.numpy(), dtype=self.dtype_from_str(
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
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        return self.empty(
            shape=shape,
            dtype=dtype
        ).normal_(mean, std, generator=generator)

    def transpose(self,
        tensor,
        axis_0:int=None,
        axis_1:int=None
    ):
        if axis_0 is not None and axis_1 is not None:
            return torch.transpose(tensor, dim0=axis_0, dim1=axis_1)
        else:
            return self.convert_to_tensor(
                array=tensor
            ).T

    def repeat(self,
        tensor,
        repeats,
        axis
    ):
        return torch.repeat_interleave(tensor, repeats=repeats, dim=axis)

    def add(self,
        tensor_0,
        tensor_1,
        out
    ):
        return torch.add(tensor_0, tensor_1, out=out)

    def matmul(self,
        tensor_0,
        tensor_1,
        out
    ):
        return torch.matmul(tensor_0, tensor_1, out=out)

    def dot(self,
        tensor_0,
        tensor_1,
        out
    ):
        return torch.dot(tensor_0, tensor_1, out=out)

    def concatenate(self,
        tensors:tuple,
        axis,
        out
    ):
        return torch.concatenate(tensors, dim=axis, out=out)

    def stack(self,
        tensors:tuple,
        axis,
        out
    ):
        return torch.stack(tensors, dim=axis, out=out)
    
    def update(self,
        tensor,
        indices,
        values
    ):
        tensor[indices] = values
        return tensor