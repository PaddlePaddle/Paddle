#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import random
import sys
import unittest

import numpy as np
from eager_op_test import OpTest

import paddle
import paddle.fluid.core as core
from python.paddle.fluid.tests.unittests.rnn.rnn_numpy import dropout, rnn

sys.path.append("./rnn")
from convert import get_params_for_net
from rnn_numpy import SimpleRNN

random.seed(2)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()


def rnn_wrapper(inputs, initial_states=None, sequence_length=None, dropout_prob=0.0, is_bidirec='forward',
                input_size=None, hidden_size=None, num_layers=1, is_test=False, mode=None):
    rnn = paddle.nn.Layer.SimpleRNN(input_size, hidden_size, num_layers, direction=is_bidirec, dropout=dropout_prob)
    return rnn(inputs, initial_states, sequence_length)

class TestSimpleRNNOp(OpTest):
    def get_weight_names(self):
        weight_names = []
        for i in range(self.num_layers):
            for j in range(0, 2 * self.direction_num):
                weight_names.append("{}.weight_{}".format(i, j))
        for i in range(self.num_layers):
            for j in range(0, 2 * self.direction_num):
                weight_names.append("{}.bias_{}".format(i, j))
        return weight_names

    def setUp(self):
        self.op_type = "rnn"
        self.python_api = rnn_wrapper
        self.dtype = "float32" if core.is_compiled_with_rocm() else "float64"
        self.sequence_length = (
            None
            if core.is_compiled_with_rocm()
            else np.array([12, 11, 10, 9, 8], dtype=np.int32)
        )
        self.num_layers = 1
        self.is_bidirec = False
        self.is_test = False
        self.mode = "RNN_TANH"
        self.dropout = 0.0
        self.set_attrs()

        self.direction_num = 2 if self.is_bidirec else 1
        direction = "bidirectional" if self.is_bidirec else "forward"
        seq_length = 12
        batch_size = 5
        input_size = 3
        hidden_size = 2

        input = np.random.uniform(
            low=-0.1, high=0.1, size=(seq_length, batch_size, input_size)
        ).astype(self.dtype)
        if self.sequence_length is not None:
            input[11][1:][:] = 0
            input[10][2:][:] = 0
            input[9][3:][:] = 0
            input[8][4:][:] = 0

        rnn1 = SimpleRNN(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            time_major=True,
            direction=direction,
            dropout=self.dropout,
            nonlinearity=self.mode,
            dtype=self.dtype,
        )

        flat_w = get_params_for_net(rnn1)

        output, last_hidden = rnn1(input, sequence_length=self.sequence_length)

        init_h = np.zeros(
            (self.num_layers * self.direction_num, batch_size, hidden_size)
        ).astype(self.dtype)

        state_out = np.ndarray((300)).astype("uint8")

        self.inputs = {
            'Input': input,
            'WeightList': flat_w,
            'PreState': [('init_h', init_h)],
            'SequenceLength': self.sequence_length,
        }
        if self.sequence_length is None:
            self.inputs = {
                'Input': input,
                'WeightList': flat_w,
                'PreState': [('init_h', init_h)],
            }
        self.attrs = {
            'dropout_prob': self.dropout,
            'is_bidirec': self.is_bidirec,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': self.num_layers,
            'is_test': self.is_test,
            'mode': self.mode,
        }
        self.outputs = {
            'Out': output,
            'State': [('last_hidden', last_hidden)],
            'Reserve': np.ndarray((400)).astype("uint8"),
            'DropoutState': state_out,
        }

    def set_attrs(self):
        pass

    def test_output(self):
        self.check_output(no_check_set=['Reserve', 'DropoutState'])

    def test_grad(self):
        if not self.is_test:
            var_name_list = self.get_weight_names()
            grad_check_list = ['Input', 'init_h']
            grad_check_list.extend(var_name_list)
            self.check_grad(set(grad_check_list), ['Out', 'last_hidden'])


class TestSimpleRNNOp1(TestSimpleRNNOp):
    def set_attrs(self):
        self.sequence_length = None


class TestSimpleRNNOp2(TestSimpleRNNOp):
    def set_attrs(self):
        self.sequence_length = None
        self.is_bidirec = True


class TestSimpleRNNOp3(TestSimpleRNNOp):
    def set_attrs(self):
        self.sequence_length = None
        self.is_test = True


class TestSimpleRNNOp4(TestSimpleRNNOp):
    def set_attrs(self):
        self.sequence_length = None
        self.is_bidirec = True
        self.is_test = True


class TestSimpleRNNOp5(TestSimpleRNNOp):
    def set_attrs(self):
        self.mode = "RNN_RELU"


if __name__ == '__main__':
    unittest.main()
