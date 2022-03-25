import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from typing import Union
from .tt.tt_linear import TTLinear
import torch.autograd.profiler as profiler

import numpy as np
from sympy.utilities.iterables import multiset_partitions
from sympy.ntheory import factorint

class TTLSTMCell(nn.Module):
    def __init__(self, in_shape, hid_shape, ranks, dims, bias=False):
        super(TTLSTMCell, self).__init__()
        self.input_size = np.prod(in_shape)
        self.hidden_size = np.prod(hid_shape)
        self.in_shape = in_shape
        self.hid_shape = hid_shape

        self._set_hidden_shape(hid_shape)

        self.ranks = ranks
        self.dims = dims
        self.bias = bias

        self.input_weights = TTLinear(self.in_shape, self.all_gate_hidden, self.ranks, self.dims, False)
        self.hidden_weights = nn.Linear(self.hidden_size, 4* self.hidden_size, bias)

        self.reset_params()

    def _set_hidden_shape(self, hid_shape):
        hid_shape[0] *= 4
        self.all_gate_hidden = hid_shape

    def reset_params(self):
        k = 1 / self.hidden_size

        for weight in self.hidden_weights.parameters():
            nn.init.uniform_(weight.data, -(k**0.5), k**0.5)

        if self.hidden_weights.bias is not None:
            nn.init.zeros_(self.hidden_weights.bias)

    def forward(self, inputs, hidden):
        hx, cx = hidden
        
        with profiler.record_function('input to hidden TT layer'):
            in2hid = self.input_weights(inputs)

        with profiler.record_function('hidden to hidden FC layer'):
            hid2hid = self.hidden_weights(hx)
        gates = in2hid + hid2hid

        in_gate = torch.sigmoid(gates[:, : self.hidden_size])
        forget_gate = torch.sigmoid(gates[:, self.hidden_size: 2*self.hidden_size])
        cell_gate = torch.tanh(gates[:, 2*self.hidden_size: 3*self.hidden_size])
        out_gate = torch.sigmoid(gates[:, 3*self.hidden_size: ])

        cy = (forget_gate *cx) + (in_gate *cell_gate)
        hy = out_gate *torch.tanh(cy)

        return (hy, cy)

class TTLSTM(nn.Module):
    def __init__(self, input_size: Union[int, list, np.ndarray], 
                       hidden_size: Union[int, list, np.ndarray], 
                       ranks: Union[list, np.ndarray],
                       dims: int, num_layers=1, dropout=0, bias=False):

        super(TTLSTM, self).__init__()

        if isinstance(input_size, int):
            self.in_shape = self.set_dims(input_size, dims)
            self.input_size = input_size
        elif isinstance(input_size, list):
            self.in_shape = np.array(input_size)
            self.input_size = np.prod(self.in_shape)
        else:
            self.in_shape = input_size
            self.input_size = np.prod(self.in_shape)

        if isinstance(hidden_size, int):
            self.hid_shape = self.set_dims(hidden_size, dims)
            self.hidden_size = hidden_size
        elif isinstance(input_size, list):
            self.hid_shape = np.array(hidden_size)
            self.hidden_size = np.prod(self.hid_shape)
        else:
            self.hid_shape = hidden_size
            self.hidden_size = np.prod(self.hid_shape)

        if isinstance(ranks, np.ndarray):
            self.ranks = ranks
        else:
            self.ranks = np.array(ranks)

        self.dims = dims

        self.num_layers = num_layers
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.bias = bias

        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i+1)
            if i ==0:
                cell = self._set_first_layer_cell()
            else:
                cell = self._set_other_layer_cell()
            setattr(self, name, cell)
            self._all_layers.append(cell)

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

    def _set_first_layer_cell(self):
        return TTLSTMCell(self.in_shape, self.hid_shape, self.ranks, self.dims, self.bias)

    def _set_other_layer_cell(self):
        return TTLSTMCell(self.hid_shape, self.hid_shape, self.ranks, self.dims, self.bias)

    def set_hidden(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = torch.zeros(batch_size, self.hidden_size).to(device)
        return (h, c)

    def set_drop(self, inputs, p):
        return PackedSequence(self.dropout(inputs.data, p), inputs.batch_sizes)

    def hidden_slice(self, hidden, start, end):
        if isinstance(hidden, torch.Tensor):
            return hidden.narrow(0, start, end -start)
        elif isinstance(hidden, tuple):
            return (hidden[0].narrow(0, start, end - start), hidden[1].narrow(0, start, end - start))
        else:
            raise TypeError

    def hidden_as_output(self, hidden):
        if isinstance(hidden, torch.Tensor):
            return hidden
        elif isinstance(hidden, tuple):
            return hidden[0]
        else:
            raise TypeError
    
    def padded_layer(self, inputs, hidden, num_cell):
        step_inputs = inputs.unbind(0)
        step_outputs = []
        cell = self._all_layers[num_cell]

        for step_input in step_inputs:
            hidden = cell(step_input, hidden)
            step_outputs.append(self.hidden_as_output(hidden))

        return torch.stack(step_outputs, 0), hidden
    
    def packed_layer(self, inputs, hidden, num_cell):
        input_offset = 0
        num_steps = inputs.batch_sizes.size(0)
        batch_sizes = inputs.batch_sizes.data
        last_batch_size = batch_sizes[0]
        input_data = inputs.data
        h_hiddens = []
        c_hiddens = []
        step_outputs = []
        cell = self._all_layers[num_cell]

        for i in range(num_steps):
            batch_size = batch_sizes[i]
            step_input = input_data.narrow(0, input_offset, batch_size)
            input_offset += batch_size
            dec = last_batch_size - batch_size
            
            if dec > 0:
                h_hiddens.append(self.hidden_slice(hidden[0], last_batch_size - dec, last_batch_size))
                c_hiddens.append(self.hidden_slice(hidden[1], last_batch_size - dec, last_batch_size))

                hidden = self.hidden_slice(hidden, 0, last_batch_size - dec)

            last_batch_size = batch_size
            hidden = cell(step_input, hidden)
            step_outputs.append(self.hidden_as_output(hidden))

        h_hiddens.append(hidden[0])
        h_hiddens.reverse()
        c_hiddens.append(hidden[1])
        c_hiddens.reverse()

        return PackedSequence(torch.cat(step_outputs, 0), inputs.batch_sizes), (torch.cat(h_hiddens, 0), torch.cat(c_hiddens, 0))

    def forward(self, inputs, init_hidden=None):

        if isinstance(inputs, PackedSequence):
            hidden = self.set_hidden(inputs.batch_sizes.data[0], inputs.data.device) if init_hidden is None else init_hidden
            _hiddens = [hidden] *self.num_layers
            layer_input = inputs
            final_hy = []
            final_cy = []

            for i in range(self.num_layers):
                step_hidden = _hiddens[i]
                layer_output, final_hidden = self.packed_layer(layer_input, step_hidden, i)
                hy, cy = final_hidden
                final_hy.append(hy)
                final_cy.append(cy)
                layer_input = layer_output

                if (self.train and self.dropout_p != 0 and i < self.num_layers -1):
                    layer_input = set_drop(layer_input, self.dropout_p)

            return layer_input, (torch.stack(final_hy, 0), torch.stack(final_cy, 0))

        else:
            hidden = self.set_hidden(inputs.size(1), inputs.device) if init_hidden is None else init_hidden
            _hiddens = [hidden] *self.num_layers
            step_hidden = _hiddens[0]
            layer_input = inputs
            final_hy = []
            final_cy = []

            for i in range(self.num_layers):
                layer_output, final_hidden = self.padded_layer(layer_input, step_hidden, i)
                hy, cy = final_hidden
                final_hy.append(hy)
                final_cy.append(cy)
                layer_input = layer_output

                if (layer_input.requires_grad and self.dropout_p != 0 and i < self.num_layers -1):
                    layer_input = self.dropout(layer_input, self.dropout_p)

            return layer_input, (torch.stack(final_hy, 0), torch.stack(final_cy, 0))

