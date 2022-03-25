import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import tensor_train

from sympy.utilities.iterables import multiset_partitions
from sympy.ntheory import factorint
import numpy as np

class TTDecomposition(object):
    def __init__(self, params, name_layer, num_layers, ranks=None, dims=None):
        tl.set_backend('pytorch')

        self.params = params
        self.name_layer = name_layer
        self.num_layers = num_layers

        if ranks is None:
            self.ranks = [1, 4, 4, 4, 1]
        else:
            self.ranks = ranks

        if dims is None:
            self.dims = 4
        else:
            self.dims = dims

        self.d = self.reshape_dims()

        self.encoder = dict()
        self.decoder = dict()

        self.decomposition()

    def reshape_dims(self):
        reshape_dims = []
        for i in range(self.dims):
            reshape_dims.append(i)
            reshape_dims.append(self.dims +i)

        return reshape_dims

    def set_dims(self, n:int, d:int):
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

    def decomposition(self):
        for i in range(self.num_layers):
            for keys in self.params.keys():
                if 'Encoder.{}.cell{}.input_weights.weight'.format(self.name_layer, (i+1)) in keys:

                    tmp_weight = self.params.get(keys).cpu()
                    in_shape = self.set_dims(tmp_weight.size(1), self.dims)
                    out_shape = self.set_dims(tmp_weight.size(0), self.dims)
                    tmp_weight = tmp_weight.view(*out_shape, *in_shape)
                    tmp_weight = torch.permute(tmp_weight, self.d).contiguous()
                    core_shape = out_shape * in_shape
                    tmp_weight = tmp_weight.view(*core_shape)
                    tt = tensor_train(tmp_weight, self.ranks)
                    self.set_tt_cores(tt, 'Encoder', '{}.cell{}.input_weights'.format(self.name_layer, (i+1)), in_shape, out_shape)
                elif 'Encoder.{}.cell{}.hidden_weights.weight'.format(self.name_layer, (i+1)) in keys:

                    tmp_weight = self.params.get(keys).cpu()
                    in_shape = self.set_dims(tmp_weight.size(1), self.dims)
                    out_shape = self.set_dims(tmp_weight.size(0), self.dims)
                    tmp_weight = tmp_weight.view(*out_shape, *in_shape)
                    tmp_weight = torch.permute(tmp_weight, self.d).contiguous()
                    core_shape = out_shape * in_shape
                    tmp_weight = tmp_weight.view(*core_shape)
                    tt = tensor_train(tmp_weight, self.ranks)
                    self.set_tt_cores(tt, 'Encoder', '{}.cell{}.hidden_weights'.format(self.name_layer, (i+1)), in_shape, out_shape)
                elif 'Decoder.{}.cell{}.input_weights.weight'.format(self.name_layer, (i+1)) in keys:

                    tmp_weight = self.params.get(keys).cpu()
                    in_shape = self.set_dims(tmp_weight.size(1), self.dims)
                    out_shape = self.set_dims(tmp_weight.size(0), self.dims)
                    tmp_weight = tmp_weight.view(*out_shape, *in_shape)
                    tmp_weight = torch.permute(tmp_weight, self.d).contiguous()
                    core_shape = out_shape * in_shape
                    tmp_weight = tmp_weight.view(*core_shape)
                    tt = tensor_train(tmp_weight, self.ranks)
                    self.set_tt_cores(tt, 'Decoder', '{}.cell{}.input_weights'.format(self.name_layer, (i+1)), in_shape, out_shape)
                elif 'Decoder.{}.cell{}.hidden_weights.weight'.format(self.name_layer, (i+1)) in keys:

                    tmp_weight = self.params.get(keys).cpu()
                    in_shape = self.set_dims(tmp_weight.size(1), self.dims)
                    out_shape = self.set_dims(tmp_weight.size(0), self.dims)
                    tmp_weight = tmp_weight.view(*out_shape, *in_shape)
                    tmp_weight = torch.permute(tmp_weight, self.d).contiguous()
                    core_shape = out_shape * in_shape
                    tmp_weight = tmp_weight.view(*core_shape)
                    tt = tensor_train(tmp_weight, self.ranks)
                    self.set_tt_cores(tt, 'Decoder', '{}.cell{}.hidden_weights'.format(self.name_layer, (i+1)), in_shape, out_shape)


    def set_tt_cores(self, tt, module_name, weight_info, in_shape, out_shape):
        if module_name == 'Encoder':
            for i, weight in enumerate(tt):
                weight = torch.Tensor(weight)
                param_shape = [weight.size(0), out_shape[i], in_shape[i], weight.size(-1)]
                weight = weight.view(*param_shape)
                self.encoder[weight_info + '.tt_core{}'.format(i+1)] = weight
        elif module_name == 'Decoder':
            for i, weight in enumerate(tt):
                param_shape = [weight.size(0), out_shape[i], in_shape[i], weight.size(-1)]
                weight = weight.view(*param_shape)
                weight = torch.Tensor(weight)
                self.decoder[weight_info + '.tt_core{}'.format(i+1)] = weight

