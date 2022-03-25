import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=False):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.input_weights = nn.Linear(input_size, 4* hidden_size, False)
        self.hidden_weights = nn.Linear(hidden_size, 4* hidden_size, bias)

        self.reset_params()

    def reset_params(self):
        k = 1 / self.hidden_size

        for weight in self.input_weights.parameters():
            nn.init.uniform_(weight.data, -(k**0.5), k**0.5)

        for weight in self.hidden_weights.parameters():
            nn.init.uniform_(weight.data, -(k**0.5), k**0.5)

        if self.hidden_weights.bias is not None:
            nn.init.zeros_(self.hidden_weights.bias)

    def forward(self, inputs, hidden):
        hx, cx = hidden
        gates = self.input_weights(inputs) + self.hidden_weights(hx)

        in_gate = torch.sigmoid(gates[:, : self.hidden_size])
        forget_gate = torch.sigmoid(gates[:, self.hidden_size: 2*self.hidden_size])
        cell_gate = torch.tanh(gates[:, 2*self.hidden_size: 3*self.hidden_size])
        out_gate = torch.sigmoid(gates[:, 3*self.hidden_size: ])

        cy = (forget_gate *cx) + (in_gate *cell_gate)
        hy = out_gate *torch.tanh(cy)

        return (hy, cy)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=False, dropout=0.2):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
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

    def _set_first_layer_cell(self):
        return LSTMCell(self.input_size, self.hidden_size, self.bias)

    def _set_other_layer_cell(self):
        return LSTMCell(self.hidden_size, self.hidden_size, self.bias)

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
            step_input = input_data.narrow(0, input_offset, batch_size).type(torch.float)
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
