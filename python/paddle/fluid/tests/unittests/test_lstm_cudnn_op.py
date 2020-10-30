#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import math

import paddle.fluid.core as core
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import random
random.seed(2)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()

SIGMOID_THRESHOLD_MIN = -40.0
SIGMOID_THRESHOLD_MAX = 13.0
EXP_MAX_INPUT = 40.0


def create_parameter(std, size, dtype):
    return np.random.uniform(low=-std, high=std, size=size).astype(dtype)


def create_parameter_for_rnn(input_size,
                             hidden_size,
                             dtype,
                             num_layers,
                             is_bidirec,
                             gate_num=4):
    flat_w = []
    std = 1.0 / math.sqrt(hidden_size)
    frame_size = gate_num * hidden_size
    direction_num = 2 if is_bidirec else 1
    for i in range(0, num_layers):
        for j in range(0, 2 * direction_num):
            if i == 0:
                if j % 2 == 0:
                    flat_w.append(("{}.weigth_{}".format(i, j),
                                   create_parameter(std, (frame_size,
                                                          input_size), dtype)))
                else:
                    flat_w.append(("{}.weigth_{}".format(i, j),
                                   create_parameter(std, (frame_size,
                                                          hidden_size), dtype)))
            else:
                if j % 2 == 0:
                    flat_w.append(
                        ("{}.weigth_{}".format(i, j), create_parameter(std, (
                            frame_size, hidden_size * direction_num), dtype)))
                else:
                    flat_w.append(("{}.weigth_{}".format(i, j),
                                   create_parameter(std, (frame_size,
                                                          hidden_size), dtype)))
    for i in range(0, num_layers):
        for j in range(0, 2 * direction_num):
            flat_w.append(("{}.bias_{}".format(i, j),
                           create_parameter(std, (frame_size), dtype)))
    return flat_w


class LayerMixin(object):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class LayerListMixin(LayerMixin):
    def __init__(self, layers=None):
        self._layers = list(layers) if layers else []

    def append(self, layer):
        self._layers.append(layer)

    def __iter__(self):
        return iter(self._layers)


class LSTMCell(LayerMixin):
    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_ih,
                 weight_hh,
                 bias_ih,
                 bias_hh,
                 bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dtype = np.float64
        self.parameters = dict()
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.parameters['weight_ih'] = self.weight_ih
        self.parameters['weight_hh'] = self.weight_hh
        if bias:
            self.bias_ih = bias_ih
            self.bias_hh = bias_hh
            self.parameters['bias_ih'] = self.bias_ih
            self.parameters['bias_hh'] = self.bias_hh
        else:
            self.bias_ih = None
            self.bias_hh = None

    def init_state(self, inputs):
        batch_size = inputs.shape[0]
        init_h = np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)
        init_c = np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)
        return init_h, init_c

    def forward(self, inputs, hx=None):
        if hx is None:
            hx = self.init_state(inputs)
        pre_hidden, pre_cell = hx
        gates = np.matmul(inputs, self.weight_ih.T)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        gates += np.matmul(pre_hidden, self.weight_hh.T)
        if self.bias_hh is not None:
            gates = gates + self.bias_hh

        chunked_gates = np.split(gates, 4, -1)

        i = 1.0 / (1.0 + np.exp(-chunked_gates[0]))
        f = 1.0 / (1.0 + np.exp(-chunked_gates[1]))
        o = 1.0 / (1.0 + np.exp(-chunked_gates[3]))
        c = f * pre_cell + i * np.tanh(chunked_gates[2])
        h = o * np.tanh(c)

        return h, (h, c)


def sequence_mask(lengths, max_len=None):
    if max_len is None:
        max_len = np.max(lengths)
    else:
        assert max_len >= np.max(lengths)
    return np.arange(max_len) < np.expand_dims(lengths, -1)


def update_state(mask, new, old):
    if not isinstance(old, (tuple, list)):
        return np.where(mask, new, old)
    else:
        return tuple(map(lambda x, y: np.where(mask, x, y), new, old))


def rnn(cell,
        inputs,
        initial_states,
        sequence_length=None,
        time_major=False,
        is_reverse=False):
    if not time_major:
        inputs = np.transpose(inputs, [1, 0, 2])
    if is_reverse:
        inputs = np.flip(inputs, 0)

    if sequence_length is None:
        mask = None
    else:
        mask = np.transpose(sequence_mask(sequence_length), [1, 0])
        mask = np.expand_dims(mask, -1)
        if is_reverse:
            mask = np.flip(mask, 0)

    time_steps = inputs.shape[0]
    state = initial_states
    outputs = []
    for t in range(time_steps):
        x_t = inputs[t]
        if mask is not None:
            m_t = mask[t]
            y, new_state = cell(x_t, state)
            y = np.where(m_t, y, 0.)
            outputs.append(y)
            state = update_state(m_t, new_state, state)
        else:
            y, new_state = cell(x_t, state)
            outputs.append(y)
            state = new_state

    outputs = np.stack(outputs)
    final_state = state

    if is_reverse:
        outputs = np.flip(outputs, 0)
    if not time_major:
        outputs = np.transpose(outputs, [1, 0, 2])
    return outputs, final_state


def birnn(cell_fw,
          cell_bw,
          inputs,
          initial_states,
          sequence_length=None,
          time_major=False):
    states_fw, states_bw = initial_states
    outputs_fw, states_fw = rnn(cell_fw,
                                inputs,
                                states_fw,
                                sequence_length,
                                time_major=time_major)

    outputs_bw, states_bw = rnn(cell_bw,
                                inputs,
                                states_bw,
                                sequence_length,
                                time_major=time_major,
                                is_reverse=True)

    outputs = np.concatenate((outputs_fw, outputs_bw), -1)
    final_states = (states_fw, states_bw)
    return outputs, final_states


def flatten(nested):
    return list(_flatten(nested))


def _flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            for subitem in _flatten(item):
                yield subitem
        else:
            yield item


def unstack(array, axis=0):
    num = array.shape[axis]
    sub_arrays = np.split(array, num, axis)
    return [np.squeeze(sub_array, axis) for sub_array in sub_arrays]


def dropout(array, p=0.0):
    if p == 0.0:
        return array

    mask = (np.random.uniform(size=array.shape) < (1 - p)).astype(array.dtype)
    return array * (mask / (1 - p))


def split_states(states, bidirectional=False, state_components=1):
    if state_components == 1:
        states = unstack(states)
        if not bidirectional:
            return states
        else:
            return list(zip(states[::2], states[1::2]))
    else:
        assert len(states) == state_components
        states = tuple([unstack(item) for item in states])
        if not bidirectional:
            return list(zip(*states))
        else:
            states = list(zip(*states))
            return list(zip(states[::2], states[1::2]))


def concat_states(states, bidirectional=False, state_components=1):
    if state_components == 1:
        return np.stack(flatten(states))
    else:
        states = flatten(states)
        componnets = []
        for i in range(state_components):
            componnets.append(states[i::state_components])
        return [np.stack(item) for item in componnets]


class RNN(LayerMixin):
    def __init__(self, cell, is_reverse=False, time_major=False):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            # for non-dygraph mode, `rnn` api uses cell.call
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major

    def forward(self, inputs, initial_states=None, sequence_length=None):
        final_outputs, final_states = rnn(self.cell,
                                          inputs,
                                          initial_states=initial_states,
                                          sequence_length=sequence_length,
                                          time_major=self.time_major,
                                          is_reverse=self.is_reverse)
        return final_outputs, final_states


class BiRNN(LayerMixin):
    def __init__(self, cell_fw, cell_bw, time_major=False):
        super(BiRNN, self).__init__()
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw
        self.time_major = time_major

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        if isinstance(initial_states, (list, tuple)):
            assert len(initial_states) == 2, \
                "length of initial_states should be 2 when it is a list/tuple"
        else:
            initial_states = [initial_states, initial_states]

        outputs, final_states = birnn(self.cell_fw, self.cell_bw, inputs,
                                      initial_states, sequence_length,
                                      self.time_major)
        return outputs, final_states


class RNNMixin(LayerListMixin):
    def forward(self, inputs, initial_states=None, sequence_length=None):
        batch_index = 1 if self.time_major else 0
        batch_size = inputs.shape[batch_index]
        dtype = inputs.dtype
        if initial_states is None:
            state_shape = (self.num_layers * self.num_directions, batch_size,
                           self.hidden_size)
            if self.state_components == 1:
                initial_states = np.zeros(state_shape, dtype)
            else:
                initial_states = tuple([
                    np.zeros(state_shape, dtype)
                    for _ in range(self.state_components)
                ])

        states = split_states(initial_states, self.num_directions == 2,
                              self.state_components)
        final_states = []

        for i, rnn_layer in enumerate(self):
            if i > 0:
                inputs = dropout(inputs, self.dropout)
            outputs, final_state = rnn_layer(inputs, states[i], sequence_length)
            final_states.append(final_state)
            inputs = outputs

        final_states = concat_states(final_states, self.num_directions == 2,
                                     self.state_components)
        return outputs, final_states


class LSTM(RNNMixin):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.,
                 time_major=False,
                 flat_w=None):
        super(LSTM, self).__init__()
        weight_len = len(flat_w)
        if direction in ["forward", "backward"]:
            is_reverse = direction == "backward"
            cell = LSTMCell(input_size, hidden_size, flat_w[0][1], flat_w[1][1],
                            flat_w[weight_len //
                                   2][1], flat_w[weight_len // 2 + 1][1])
            self.append(RNN(cell, is_reverse, time_major))
            for i in range(1, num_layers):
                cell = LSTMCell(hidden_size, hidden_size, flat_w[i * 2][1],
                                flat_w[i * 2 + 1][1],
                                flat_w[weight_len // 2 + i * 2][1],
                                flat_w[weight_len // 2 + i * 2 + 1][1])
                self.append(RNN(cell, is_reverse, time_major))
        elif direction == "bidirectional":
            cell_fw = LSTMCell(input_size, hidden_size, flat_w[0][1],
                               flat_w[1][1], flat_w[weight_len // 2][1],
                               flat_w[weight_len // 2 + 1][1])
            cell_bw = LSTMCell(input_size, hidden_size, flat_w[2][1],
                               flat_w[3][1], flat_w[weight_len // 2 + 2][1],
                               flat_w[weight_len // 2 + 3][1])
            self.append(BiRNN(cell_fw, cell_bw, time_major))
            for i in range(1, num_layers):
                cell_fw = LSTMCell(2 * hidden_size, hidden_size,
                                   flat_w[i * 4][1], flat_w[i * 4 + 1][1],
                                   flat_w[weight_len // 2 + i * 4][1],
                                   flat_w[weight_len // 2 + i * 4 + 1][1])
                cell_bw = LSTMCell(2 * hidden_size, hidden_size,
                                   flat_w[i * 4 + 2][1], flat_w[i * 4 + 3][1],
                                   flat_w[weight_len // 2 + i * 4 + 2][1],
                                   flat_w[weight_len // 2 + i * 4 + 3][1])
                self.append(BiRNN(cell_fw, cell_bw, time_major))
        else:
            raise ValueError(
                "direction should be forward, backward or bidirectional, "
                "received direction = {}".format(direction))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = 2 if direction == "bidirectional" else 1
        self.time_major = time_major
        self.num_layers = num_layers
        self.state_components = 2


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNLstmOp(OpTest):
    def get_weight_names(self, direction_num):
        weight_names = []
        for i in range(self.num_layers):
            for j in range(0, 2 * direction_num):
                weight_names.append("{}.weigth_{}".format(i, j))
        for i in range(self.num_layers):
            for j in range(0, 2 * direction_num):
                weight_names.append("{}.bias_{}".format(i, j))
        return weight_names

    def setUp(self):
        self.op_type = "cudnn_lstm"
        self.dtype = np.float64
        self.sequence_length = np.array([12, 11, 10, 9, 8], dtype=np.int32)
        self.num_layers = 2
        self.is_bidirec = False
        self.is_test = False
        self.set_attrs()

        direction_num = 2 if self.is_bidirec else 1
        direction = "bidirectional" if self.is_bidirec else "forward"
        seq_length = 2
        batch_size = 4
        input_size = 5
        hidden_size = 4

        input = np.random.uniform(
            low=-0.1, high=0.1,
            size=(seq_length, batch_size, input_size)).astype(self.dtype)
        if self.sequence_length is not None:
            input[11][1:][:] = 0
            input[10][2:][:] = 0
            input[9][3:][:] = 0
            input[8][4:][:] = 0

        flat_w = create_parameter_for_rnn(input_size, hidden_size, self.dtype,
                                          self.num_layers, self.is_bidirec)
        rnn1 = LSTM(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            time_major=True,
            direction=direction,
            flat_w=flat_w)

        output, (last_hidden, last_cell) = rnn1(
            input, sequence_length=self.sequence_length)

        init_h = np.zeros((self.num_layers * direction_num, batch_size,
                           hidden_size)).astype(self.dtype)
        init_c = np.zeros((self.num_layers * direction_num, batch_size,
                           hidden_size)).astype(self.dtype)
        state_out = np.ndarray((300)).astype("uint8")
        print(flat_w)

        self.inputs = {
            'Input': input,
            'WeightList': flat_w,
            'InitH': init_h,
            'InitC': init_c,
            'SequenceLength': self.sequence_length
        }
        if self.sequence_length is None:
            self.inputs = {
                'Input': input,
                'WeightList': flat_w,
                'InitH': init_h,
                'InitC': init_c,
            }
        self.attrs = {
            'dropout_prob': 0.0,
            'is_bidirec': self.is_bidirec,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': self.num_layers,
            'is_test': self.is_test,
        }
        self.outputs = {
            'Out': output,
            "LastH": last_hidden,
            'LastC': last_cell,
            'Reserve': np.ndarray((400)).astype("uint8"),
            'StateOut': state_out
        }

    #def test_output_with_place(self):
    #    place = core.CUDAPlace(0)
    #    self.check_output_with_place(
    #        place, no_check_set=['Reserve', 'StateOut'])

    def set_attrs(self):
        self.sequence_length = None

    def test_grad_with_place(self):
        place = core.CPUPlace()
        direction_num = 2 if self.is_bidirec else 1
        var_name_list = self.get_weight_names(direction_num)
        grad_check_list = ['Input', 'InitH', 'InitC']
        grad_check_list.extend(var_name_list)
        for var_name in var_name_list:
            self.check_grad_with_place(place,
                                       set(grad_check_list),
                                       ['Out', 'LastH', 'LastC'])


#class TestCUDNNLstmCpu(TestCUDNNLstmOp):
#    def test_output_with_place(self):
#        place = core.CPUPlace()
#        self.check_output_with_place(
#            place, no_check_set=['Reserve', 'StateOut'])
#
#
#class TestCUDNNLstmCpu1(TestCUDNNLstmCpu):
#    def set_attrs(self):
#        self.sequence_length = None
#
#
#class TestCUDNNLstmCpu2(TestCUDNNLstmCpu):
#    def set_attrs(self):
#        self.sequence_length = None
#        self.is_bidirec = True
#
#
#class TestCUDNNLstmCpu3(TestCUDNNLstmCpu):
#    def set_attrs(self):
#        self.num_layers = 2
#
#
#class TestCUDNNLstmCpu4(TestCUDNNLstmCpu):
#    def set_attrs(self):
#        self.is_bidirec = True
#        self.num_layers = 2
#        self.sequence_length = None
#
#
#class TestCUDNNLstmCpu5(TestCUDNNLstmCpu):
#    def set_attrs(self):
#        self.is_bidirec = True
#        self.num_layers = 2
#
#
#class TestCUDNNLstmCpu6(TestCUDNNLstmCpu):
#    def set_attrs(self):
#        self.is_test = True
#        self.is_bidirec = True
#        self.num_layers = 2
#
#
#class TestCUDNNLstmCpu7(TestCUDNNLstmCpu):
#    def set_attrs(self):
#        self.is_test = True
#        self.num_layers = 2
#
#
#@unittest.skipIf(not core.is_compiled_with_cuda(),
#                 "core is not compiled with CUDA")
#class TestCUDNNlstmAPI(unittest.TestCase):
#    def test_lstm(self):
#        seq_len = 20
#        batch_size = 5
#        hidden_size = 20
#        dropout_prob = 0.0
#        num_layers = 1
#        input = fluid.data(
#            name='input',
#            shape=[seq_len, batch_size, hidden_size],
#            dtype='float64')
#        init_h = layers.fill_constant([num_layers, batch_size, hidden_size],
#                                      'float64', 0.0)
#        init_c = layers.fill_constant([num_layers, batch_size, hidden_size],
#                                      'float64', 0.0)
#        rnn_out, last_h, last_c = layers.lstm(input, init_h, init_c, seq_len,
#                                              hidden_size, num_layers,
#                                              dropout_prob, False)
#        exe = fluid.Executor(fluid.CUDAPlace(0))
#        exe.run(fluid.default_startup_program())
#        input_i = np.random.uniform(
#            low=-0.1, high=0.1, size=(seq_len, batch_size,
#                                      hidden_size)).astype("float64")
#        out = exe.run(fluid.default_main_program(),
#                      feed={'input': input_i},
#                      fetch_list=[rnn_out, last_h, last_c, 'cudnn_lstm_0.w_0'])
#
#
#@unittest.skipIf(not core.is_compiled_with_cuda(),
#                 "core is not compiled with CUDA")
#class TestCUDNNlstmAPI(unittest.TestCase):
#    def test_lstm(self):
#        seq_len = 20
#        batch_size = 5
#        hidden_size = 20
#        dropout_prob = 0.0
#        num_layers = 2
#        input = fluid.data(
#            name='input',
#            shape=[seq_len, batch_size, hidden_size],
#            dtype='float64')
#        init_h = layers.fill_constant([num_layers, batch_size, hidden_size],
#                                      'float64', 0.0)
#        init_c = layers.fill_constant([num_layers, batch_size, hidden_size],
#                                      'float64', 0.0)
#        rnn_out, last_h, last_c = layers.lstm(input, init_h, init_c, seq_len,
#                                              hidden_size, num_layers,
#                                              dropout_prob, False, True)
#        exe = fluid.Executor(fluid.CUDAPlace(0))
#        exe.run(fluid.default_startup_program())
#        input_i = np.random.uniform(
#            low=-0.1, high=0.1, size=(seq_len, batch_size,
#                                      hidden_size)).astype("float64")
#        out = exe.run(fluid.default_main_program(),
#                      feed={'input': input_i},
#                      fetch_list=[rnn_out, last_h, last_c, 'cudnn_lstm_0.w_0'])
#

if __name__ == '__main__':
    unittest.main()
