from typing import Union

import torch
import torch.nn as nn

import numpy as np

from .tt_module import _TTBase

class TTLinear(_TTBase):
    def __init__(self, in_features:int, out_features:int, ranks:Union[list, np.ndarray], dims:int, bias=False):
        super(TTLinear, self).__init__(in_features=in_features, out_features=out_features, ranks=ranks, dims=dims, bias=bias)

    def set_tt_cores(self):
        tt_cores_info = []
        for i in range(self.dims):
            tt_core_info = dict(
                    name="tt_core%d" % (i+1),
                    shape=(self.ranks[i], self.out_shape[i], self.in_shape[i], self.ranks[i+1]))
                    
            tmp = nn.Parameter(torch.Tensor(*tt_core_info["shape"]))
            self.register_parameter(tt_core_info["name"], tmp)

            tt_cores_info.append(tt_core_info)

        self.tt_info["tt_cores"] = tt_cores_info

    def set_params_info(self):
        
        original = self.in_features * self.out_features

        tt_format = np.sum(self.ranks[:self.dims] * self.in_shape * self.out_shape * self.ranks[1:self.dims +1])
        cr = original / tt_format

        self.tt_info["tt_format_params"] = tt_format
        self.tt_info["original_params"] = original
        self.tt_info["compression_ration"] = cr

        print("compression_ration is: ", cr)

    #def init_params(self): 

    def tt_op(self, inputs):
        batch_size = inputs.shape[0]
        res = inputs.view(-1, *self.in_shape)
        # res_shape = res.size()

        # weight x input
        weight = getattr(self, "tt_core%d" %(self.dims))
        # res = permute(res, (res_shape[-1], 
        


        res = torch.tensordot(weight, res, dims=([2], [-1]))

        # input x weight 
        # weight = getattr(self, "tt_core1")
        # res = torch.tensordot(res, weight, dims=([1], [-2]))

        # einsum
        #res = torch.einsum('bk..., ijkl-> b...ijl', res, weight)

        #for i in range(2, self.dims +1):
        for i in range(self.dims -1, 0, -1):
            weight = getattr(self, "tt_core%d" %i)
            res = torch.tensordot(weight, res, dims=([-1, 2], [0, -1]))
            #res = torch.tensordot(res, weight, dims=([1, -1], [-2, 0]))
            #res = torch.einsum('bi...r, rkij-> b...kj', res, weight)

        res = res.view(-1, batch_size).T.contiguous()
        return res

    def forward(self, inputs):
        res = self.tt_op(inputs)
        if self.bias is not None:
            res = torch.add(self.bias, res)

        return res

