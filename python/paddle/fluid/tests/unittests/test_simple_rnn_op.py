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

from test_lstm_cudnn_op import LSTMCell, RNNMixin, create_parameter_for_rnn, RNN

import unittest
import numpy as np
import math

from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import random
random.seed(2)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()


class SimpleRNNCell(LSTMCell):
    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_ih,
                 weight_hh,
                 bias_ih,
                 bias_hh,
                 bias=True,
                 act="rnn_relu"):
        super(SimpleRNNCell, self).__init__(input_size, hidden_size, weight_ih,
                                            weight_hh, bias_ih, bias_hh, bias)
        self.act = act

    def init_state(self, inputs):
        batch_size = inputs.shape[0]
        init_h = np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)
        return init_h

    def forward(self, inputs, pre_hidden=None):
        if pre_hidden is None:
            pre_hidden = self.init_state(inputs)
        z = np.matmul(inputs, self.weight_ih.T)
        if self.bias_ih is not None:
            z = z + self.bias_ih
        if self.bias_hh is not None:
            z = z + self.bias_hh
        z += np.matmul(pre_hidden, self.weight_hh.T)
        h = None
        if self.act == "rnn_relu":
            h = np.maximum(z, 0)
        elif self.act == "rnn_tanh":
            h = np.tanh(z)
        return h, h


class SimpleRNN(RNNMixin):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.,
                 time_major=False,
                 flat_w=None,
                 act="rnn_relu"):
        super(SimpleRNN, self).__init__()
        weight_len = len(flat_w)
        if direction in ["forward", "backward"]:
            is_reverse = direction == "backward"
            for i in range(0, num_layers):
                x_size = hidden_size
                if i == 0:
                    x_size = input_size
                cell = SimpleRNNCell(
                    x_size,
                    hidden_size,
                    flat_w[i * 2][1],
                    flat_w[i * 2 + 1][1],
                    flat_w[weight_len // 2 + i * 2][1],
                    flat_w[weight_len // 2 + i * 2 + 1][1],
                    act=act)
                self.append(RNN(cell, is_reverse, time_major))
        elif direction == "bidirectional":
            for i in range(0, num_layers):
                x_size = 2 * hidden_size
                if i == 0:
                    x_size = input_size
                cell_fw = SimpleRNNCell(
                    x_size,
                    hidden_size,
                    flat_w[i * 4][1],
                    flat_w[i * 4 + 1][1],
                    flat_w[weight_len // 2 + i * 4][1],
                    flat_w[weight_len // 2 + i * 4 + 1][1],
                    act=act)
                cell_bw = SimpleRNNCell(
                    x_size,
                    hidden_size,
                    flat_w[i * 4 + 2][1],
                    flat_w[i * 4 + 3][1],
                    flat_w[weight_len // 2 + i * 4 + 2][1],
                    flat_w[weight_len // 2 + i * 4 + 3][1],
                    act=act)
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
        self.state_components = 1


class TestSimpleRNNOpCpu(OpTest):
    def setUp(self):
        self.op_type = "cudnn_lstm"
        self.dtype = np.float64
        self.sequence_length = np.array([12, 11, 10, 9, 8], dtype=np.int32)
        self.num_layers = 1
        self.is_bidirec = False
        self.is_test = False
        self.act = "rnn_relu"
        self.dropout = 0.
        self.set_attrs()

        direction_num = 2 if self.is_bidirec else 1
        direction = "bidirectional" if self.is_bidirec else "forward"
        seq_length = 12
        batch_size = 5
        input_size = 21
        hidden_size = 21

        input = np.random.uniform(
            low=-0.1, high=0.1,
            size=(seq_length, batch_size, input_size)).astype(self.dtype)
        if self.sequence_length is not None:
            input[11][1:][:] = 0
            input[10][2:][:] = 0
            input[9][3:][:] = 0
            input[8][4:][:] = 0

        flat_w = create_parameter_for_rnn(
            input_size,
            hidden_size,
            self.dtype,
            self.num_layers,
            self.is_bidirec,
            gate_num=1)

        rnn1 = SimpleRNN(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            time_major=True,
            direction=direction,
            dropout=self.dropout,
            flat_w=flat_w,
            act=self.act)

        output, last_hidden = rnn1(input, sequence_length=self.sequence_length)

        init_h = np.zeros((self.num_layers * direction_num, batch_size,
                           hidden_size)).astype(self.dtype)

        state_out = np.ndarray((300)).astype("uint8")

        self.inputs = {
            'Input': input,
            'WeightList': flat_w,
            'InitH': init_h,
            'SequenceLength': self.sequence_length
        }
        if self.sequence_length is None:
            self.inputs = {
                'Input': input,
                'WeightList': flat_w,
                'InitH': init_h,
            }
        self.attrs = {
            'dropout_prob': self.dropout,
            'is_bidirec': self.is_bidirec,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': self.num_layers,
            'is_test': self.is_test,
            'cell_type': self.act
        }
        self.outputs = {
            'Out': output,
            'LastH': last_hidden,
            'LastC': np.ndarray((400)).astype("uint8"),
            'Reserve': np.ndarray((400)).astype("uint8"),
            'StateOut': state_out
        }

    def set_attrs(self):
        pass

    def test_output_with_place(self):
        place = core.CPUPlace()
        self.check_output_with_place(
            place, no_check_set=['Reserve', 'StateOut', 'LastC'])


class TestSimpleRNNOpCpu1(TestSimpleRNNOpCpu):
    def set_attrs(self):
        self.is_test = True


class TestSimpleRNNOpCpuTanh1(TestSimpleRNNOpCpu):
    def set_attrs(self):
        self.act = "rnn_tanh"


if __name__ == '__main__':
    unittest.main()
