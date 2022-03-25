from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tt_module import _TTBase

class TTConv2d(_TTBase):
    def __init__(self, in_features:int, out_features:int, ranks:list, dims:int, \
                 kernel_size: Union[int, tuple], stride=1, padding=0, bias=True):
        super(TTConv2d, self).__init__(in_features=in_features, out_features=out_features, ranks=ranks, dims=dims, bias=bias)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def set_tt_core(self):
        assert len(self.ranks) == self.dims +2, "number of ranks should be dimension + 2!! "

        tt_cores_info = []

        for i in range(self.dims +1):
            if i == 0:
                tt_core_info = dict(
                        name="tt_core0",
                        shape=(self.ranks[i+1], self.ranks[i], self.kernel_size[0], self.kernel_size[1])
                        )
            else:
                tt_core_info = dict(
                        name="tt_core%d" % i,
                        shape=(self.ranks[i], self.out_shape[i-1], self.in_shape[i-1], self.ranks[i+1])
                        )
            tmp = nn.Parameter(torch.Tensor(*tt_core_info["shape"]))
            self.register_parameter(tt_core_info["name"], tmp)

            tt_cores_info.append(tt_core_info)

        self.tt_info["tt_cores"] = tt_cores_info

    def set_params_info(self):
        
        original = self.in_features * self.out_features

        tt_format = np.sum(self.ranks[1:self.dims] * self.in_shape * self.out_shape * self.ranks[2:(self.dims +1)])
        params_tt_core0 = self.ranks[0] * np.prod(self.kernel_size) * self.ranks[1]
        tt_format += params_tt_core0

        cr = original / tt_format

        self.tt_info["tt_format_params"] = tt_format
        self.tt_info["original_params"] = original
        self.tt_info["compression_ration"] = cr

        print("compression_ration is: ", cr)

    def tt_op(self, inputs):
        batch_size = inputs.shape[0]
        image_size = inputs.shape[-2:]
        res = inputs.view(-1, 1, *image_size)

        weight = getattr(self, "tt_core0")
        res = F.conv2d(res, weight, bias=None, stride=self.stride, padding=self.padding)
        new_image_size = res.shape[-2:]

        res = res.contiguous().view(batch_size, *self.in_shape, self.ranks[1], -1)

        for i in range(1, self.dims +1):
            weight = getattr(self, "tt_core%d" %i)
            res = torch.tensordot(res, weight, dims

    def forward(self, inputs: torch.Tensor):
        res = self.tt_op(inputs)
        if self.bias is not None:
            res = torch.add(self.bias, res)

        res = res.permute(0, 3, 1, 2).contiguous()
        res = res.contiguous()

        return res





