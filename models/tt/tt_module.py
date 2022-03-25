from typing import Union
import abc

import torch
import torch.nn as nn
import numpy as np

from sympy.utilities.iterables import multiset_partitions
from sympy.ntheory import factorint

__all__ = ['_TTBase']

class _TTBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, in_features:Union[int, list, np.ndarray], out_features:Union[int, list, np.ndarray], ranks:Union[list, np.ndarray], dims:int, bias=False):
        super(_TTBase, self).__init__()

        if isinstance(in_features, int):
            self.in_shape = self.set_dims(in_features, dims)
            self.in_features = in_features
        elif isinstance(in_features, list):
            self.in_shape = np.array(in_features)
            self.in_features = np.prod(self.in_shape)
        else:
            self.in_shape = in_features
            self.in_features = np.prod(self.in_shape)

        if isinstance(out_features, int):
            self.out_shape = self.set_dims(out_features, dims)
            self.out_features = out_features
        elif isinstance(in_features, list):
            self.out_shape = np.array(out_features)
            self.out_features = np.prod(self.out_shape)
        else:
            self.out_shape = out_features
            self.out_features = np.prod(self.out_shape)

        if isinstance(ranks, np.ndarray):
            self.ranks = ranks
        else:
            self.ranks = np.array(ranks)

        self.dims = dims
        self.tt_info = dict()

        self.validate_config()

        self.set_tt_cores()
        self.set_params_info()

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

    def validate_config(self):
        '''validate configuration'''
        assert self.ranks[0] == self.ranks[-1] == 1, 'first rank & last rank should be 1!'

        if self.dims == 4:
            assert self.in_features > 16, 'in_features is too small for given dimsension!'
        elif self.dims == 6:
            assert self.in_features > 64, 'in_features is too small for given dimsension!'
        elif self.dims == 8:
            assert self.in_features > 256, 'in_features is too small for given dimsension!'

    def set_dims(self, n:int, d:int) -> list:
        p = self.fac(n)

        if len(p) < d:
            p = p +[1, ] * (d -len(p))

        def prepr(x):
            return sorted([np.prod(_) for _ in x])

        raw_fac = multiset_partitions(p, d)
        factors = [prepr(f) for f in raw_fac]

        return np.array(factors[-1])

    def fac(self, n:int):
        res = []
        f = factorint(n)
        for k, v in f.items():
            res += [k, ] *v
        return res

    @abc.abstractmethod
    def set_tt_cores(self):
        pass

    @abc.abstractmethod
    def set_params_info(self):
        pass

    @abc.abstractmethod
    def tt_op(self, inputs: torch.Tensor) -> torch.Tensor:
        pass
