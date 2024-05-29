#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Module to implement PyTorch backend."""

__name__    = 'quantrl.backends.torch'
__authors__ = ["Sampreet Kalita"]
__created__ = "2024-03-10"
__updated__ = "2024-04-25"

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
            library=torch,
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

        # set seed
        self.seed = None

    def convert_to_typed(self,
        tensor,
        dtype:str=None
    ) -> torch.Tensor:
        if self.is_typed(
            tensor=tensor,
            dtype=dtype
        ):
            return tensor
        return torch.tensor(tensor, dtype=self.dtype_from_str(
            dtype=dtype
        ) if dtype is not None else None)

    def convert_to_numpy(self,
        tensor,
        dtype:str=None
    ) -> np.ndarray:
        return np.asarray(tensor.detach().cpu().numpy() if self.device == 'cuda' else tensor.numpy(), dtype=self.dtype_from_str(
            dtype=dtype,
            numpy=True
        ) if dtype is not None else None)

    def generator(self,
        seed:int=None
    ) -> torch.Generator:
        if self.seed is not None:
            self.seed += 1
        elif seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(1000)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        return generator

    def integers(self,
        generator:torch.Generator,
        shape:tuple,
        low:int=0,
        high:int=1000,
        dtype:str=None
    ) -> torch.Tensor:
        return torch.randint(low, high, shape, generator=generator, dtype=self.dtype_from_str(
            dtype=dtype
        ))

    def normal(self,
        generator:torch.Generator,
        shape:tuple,
        mean:float=0.0,
        std:float=1.0,
        dtype:str=None
    ) -> torch.Tensor:
        return self.empty(
            shape=shape,
            dtype=dtype
        ).normal_(mean, std, generator=generator)

    def uniform(self,
        generator:torch.Generator,
        shape:tuple,
        low:float=0.0,
        high:float=1.0,
        dtype:str=None
    ) -> torch.Tensor:
        return low + (high - low) * torch.rand(shape, generator=generator, dtype=self.dtype_from_str(
            dtype=dtype
        ))

    def transpose(self,
        tensor,
        axis_0:int=None,
        axis_1:int=None
    ) -> torch.Tensor:
        if axis_0 is not None and axis_1 is not None:
            return torch.transpose(tensor, dim0=axis_0, dim1=axis_1)
        return self.convert_to_typed(
            tensor=tensor
        ).T

    def repeat(self,
        tensor,
        repeats,
        axis
    ) -> torch.Tensor:
        return torch.repeat_interleave(tensor, repeats=repeats, dim=axis)

    def add(self,
        tensor_0,
        tensor_1,
        out
    ) -> torch.Tensor:
        return torch.add(tensor_0, tensor_1, out=out)

    def matmul(self,
        tensor_0,
        tensor_1,
        out
    ) -> torch.Tensor:
        return torch.matmul(tensor_0, tensor_1, out=out)

    def dot(self,
        tensor_0,
        tensor_1,
        out
    ) -> torch.Tensor:
        return torch.dot(tensor_0, tensor_1, out=out)

    def norm(self,
        tensor,
        axis
    ) -> torch.Tensor:
        return torch.norm(tensor, dim=axis)

    def concatenate(self,
        tensors:tuple,
        axis,
        out
    ) -> torch.Tensor:
        return torch.concatenate(tensors, dim=axis, out=out)

    def stack(self,
        tensors:tuple,
        axis,
        out
    ) -> torch.Tensor:
        return torch.stack(tensors, dim=axis, out=out)
    
    def update(self,
        tensor,
        indices,
        values
    ) -> torch.Tensor:
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