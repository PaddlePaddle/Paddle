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
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import random
import sys

from rnn_numpy import SimpleRNN, LSTM, GRU
from convert import get_params_for_net
sys.path.append("../")
from op_test import OpTest

random.seed(2)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()

SIGMOID_THRESHOLD_MIN = -40.0
SIGMOID_THRESHOLD_MAX = 13.0
EXP_MAX_INPUT = 40.0


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestRNNOp(OpTest):
    def get_weight_names(self):
        weight_names = []
        for i in range(2 * self.num_layers):
            weight_names.append('weight{}'.format(i))
        for i in range(2 * self.num_layers):
            weight_names.append('bias{}'.format(i))
        return weight_names

    def setUp(self):
        self.op_type = "rnn"
        self.dtype = np.float64
        self.sequence_length = np.array([12, 11, 10, 9, 8], dtype=np.int32)
        self.num_layers = 1
        self.is_bidirec = False
        self.mode = "LSTM"
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

        rnn1 = LSTM(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            time_major=True,
            direction=direction)

        flat_w = get_params_for_net(rnn1)

        output, (last_hidden, last_cell) = rnn1(
            input, sequence_length=self.sequence_length)

        init_h = np.zeros((self.num_layers * direction_num, batch_size,
                           hidden_size)).astype(self.dtype)
        init_c = np.zeros((self.num_layers * direction_num, batch_size,
                           hidden_size)).astype(self.dtype)
        state_out = np.ndarray((300)).astype("uint8")

        self.inputs = {
            'Input': input,
            'WeightList': flat_w,
            'PreState': [('init_h', init_h), ('init_c', init_c)],
            'SequenceLength': self.sequence_length
        }
        if self.sequence_length is None:
            self.inputs = {
                'Input': input,
                'WeightList': flat_w,
                'PreState': [('init_h', init_h), ('init_c', init_c)],
            }
        self.attrs = {
            'dropout_prob': 0.0,
            'is_bidirec': self.is_bidirec,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': self.num_layers,
            'mode': self.mode,
            'is_test': False
        }
        self.outputs = {
            'Out': output,
            "State": [('last_hidden', last_hidden), ('last_cell', last_cell)],
            'Reserve': np.ndarray((400)).astype("uint8"),
            'DropoutState': state_out
        }

    def test_output_with_place(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, no_check_set=['Reserve', 'DropoutState'])

    def set_attrs(self):
        pass

    #def test_grad_with_place(self):
    #    place = core.CPUPlace()
    #    var_name_list = self.get_weight_names()
    #    for var_name in var_name_list:
    #        self.check_grad_with_place(
    #            place,
    #            set(['Input', var_name, 'InitH', 'InitC']),
    #            ['Out', 'LastH', 'LastC'])


class TestRNNCpu(TestRNNOp):
    def test_output_with_place(self):
        place = core.CPUPlace()
        self.check_output_with_place(
            place, no_check_set=['Reserve', 'DropoutState'])


class TestRNNCpu1(TestRNNCpu):
    def set_attrs(self):
        self.sequence_length = None


class TestRNNCpu2(TestRNNCpu):
    def set_attrs(self):
        self.sequence_length = None
        self.is_bidirec = True


class TestRNNCpu3(TestRNNCpu):
    def set_attrs(self):
        self.num_layers = 2


class TestRNNCpu4(TestRNNCpu):
    def set_attrs(self):
        self.is_bidirec = True
        self.num_layers = 2
        self.sequence_length = None


class TestRNNCpu5(TestRNNCpu):
    def set_attrs(self):
        self.is_bidirec = True
        self.num_layers = 2


if __name__ == '__main__':
    unittest.main()
