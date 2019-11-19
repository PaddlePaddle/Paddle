# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import numpy

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core

from paddle.fluid.executor import Executor
from paddle.fluid import framework

from paddle.fluid.layers.rnn import LSTMCell, GRUCell, RNNCell
from paddle.fluid.layers import rnn as dynamic_rnn
from paddle.fluid import contrib
from paddle.fluid.contrib.layers import basic_lstm
import paddle.fluid.layers.utils as utils

import numpy as np


class TestLSTMCell(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16

    def test_run(self):
        inputs = fluid.data(
            name='inputs', shape=[None, self.input_size], dtype='float32')
        pre_hidden = fluid.data(
            name='pre_hidden', shape=[None, self.hidden_size], dtype='float32')
        pre_cell = fluid.data(
            name='pre_cell', shape=[None, self.hidden_size], dtype='float32')

        cell = LSTMCell(self.hidden_size)
        lstm_hidden_new, lstm_states_new = cell(inputs, [pre_hidden, pre_cell])

        lstm_unit = contrib.layers.rnn_impl.BasicLSTMUnit(
            "basicLSTM", self.hidden_size, None, None, None, None, 1.0,
            "float32")
        lstm_hidden, lstm_cell = lstm_unit(inputs, pre_hidden, pre_cell)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        inputs_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.input_size)).astype('float32')
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')
        pre_cell_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')

        param_names = [[
            "LSTMCell/BasicLSTMUnit_0.w_0", "basicLSTM/BasicLSTMUnit_0.w_0"
        ], ["LSTMCell/BasicLSTMUnit_0.b_0", "basicLSTM/BasicLSTMUnit_0.b_0"]]

        for names in param_names:
            param = np.array(fluid.global_scope().find_var(names[0]).get_tensor(
            ))
            param = np.random.uniform(
                -0.1, 0.1, size=param.shape).astype('float32')
            fluid.global_scope().find_var(names[0]).get_tensor().set(param,
                                                                     place)
            fluid.global_scope().find_var(names[1]).get_tensor().set(param,
                                                                     place)

        out = exe.run(feed={
            'inputs': inputs_np,
            'pre_hidden': pre_hidden_np,
            'pre_cell': pre_cell_np
        },
                      fetch_list=[lstm_hidden_new, lstm_hidden])

        self.assertTrue(np.allclose(out[0], out[1], rtol=1e-4, atol=0))


class TestGRUCell(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16

    def test_run(self):
        inputs = fluid.data(
            name='inputs', shape=[None, self.input_size], dtype='float32')
        pre_hidden = layers.data(
            name='pre_hidden',
            shape=[None, self.hidden_size],
            append_batch_size=False,
            dtype='float32')

        cell = GRUCell(self.hidden_size)
        gru_hidden_new, _ = cell(inputs, pre_hidden)

        gru_unit = contrib.layers.rnn_impl.BasicGRUUnit(
            "basicGRU", self.hidden_size, None, None, None, None, "float32")
        gru_hidden = gru_unit(inputs, pre_hidden)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        inputs_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.input_size)).astype('float32')
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')

        param_names = [
            ["GRUCell/BasicGRUUnit_0.w_0", "basicGRU/BasicGRUUnit_0.w_0"],
            ["GRUCell/BasicGRUUnit_0.w_1", "basicGRU/BasicGRUUnit_0.w_1"],
            ["GRUCell/BasicGRUUnit_0.b_0", "basicGRU/BasicGRUUnit_0.b_0"],
            ["GRUCell/BasicGRUUnit_0.b_1", "basicGRU/BasicGRUUnit_0.b_1"]
        ]

        for names in param_names:
            param = np.array(fluid.global_scope().find_var(names[0]).get_tensor(
            ))
            param = np.random.uniform(
                -0.1, 0.1, size=param.shape).astype('float32')
            fluid.global_scope().find_var(names[0]).get_tensor().set(param,
                                                                     place)
            fluid.global_scope().find_var(names[1]).get_tensor().set(param,
                                                                     place)

        out = exe.run(feed={'inputs': inputs_np,
                            'pre_hidden': pre_hidden_np},
                      fetch_list=[gru_hidden_new, gru_hidden])

        self.assertTrue(np.allclose(out[0], out[1], rtol=1e-4, atol=0))


class TestRnn(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_size = 16
        self.hidden_size = 16
        self.seq_len = 4

    def test_run(self):
        inputs_basic_lstm = fluid.data(
            name='inputs_basic_lstm',
            shape=[None, None, self.input_size],
            dtype='float32')
        sequence_length = fluid.data(
            name="sequence_length", shape=[None], dtype='int64')

        inputs_dynamic_rnn = layers.transpose(inputs_basic_lstm, perm=[1, 0, 2])
        cell = LSTMCell(self.hidden_size, name="LSTMCell_for_rnn")
        output, final_state = dynamic_rnn(
            cell=cell,
            inputs=inputs_dynamic_rnn,
            sequence_length=sequence_length,
            is_reverse=False)
        output_new = layers.transpose(output, perm=[1, 0, 2])

        rnn_out, last_hidden, last_cell = basic_lstm(inputs_basic_lstm, None, None, self.hidden_size, num_layers=1, \
                batch_first = False, bidirectional=False, sequence_length=sequence_length, forget_bias = 1.0)

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        inputs_basic_lstm_np = np.random.uniform(
            -0.1, 0.1,
            (self.seq_len, self.batch_size, self.input_size)).astype('float32')
        sequence_length_np = np.ones(
            self.batch_size, dtype='int64') * self.seq_len

        inputs_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.input_size)).astype('float32')
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')
        pre_cell_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')

        param_names = [[
            "LSTMCell_for_rnn/BasicLSTMUnit_0.w_0",
            "basic_lstm_layers_0/BasicLSTMUnit_0.w_0"
        ], [
            "LSTMCell_for_rnn/BasicLSTMUnit_0.b_0",
            "basic_lstm_layers_0/BasicLSTMUnit_0.b_0"
        ]]

        for names in param_names:
            param = np.array(fluid.global_scope().find_var(names[0]).get_tensor(
            ))
            param = np.random.uniform(
                -0.1, 0.1, size=param.shape).astype('float32')
            fluid.global_scope().find_var(names[0]).get_tensor().set(param,
                                                                     place)
            fluid.global_scope().find_var(names[1]).get_tensor().set(param,
                                                                     place)

        out = exe.run(feed={
            'inputs_basic_lstm': inputs_basic_lstm_np,
            'sequence_length': sequence_length_np,
            'inputs': inputs_np,
            'pre_hidden': pre_hidden_np,
            'pre_cell': pre_cell_np
        },
                      fetch_list=[output_new, rnn_out])

        self.assertTrue(np.allclose(out[0], out[1], rtol=1e-4))


class TestRnnUtil(unittest.TestCase):
    """
    Test cases for rnn apis' utility methods for coverage.
    """

    def test_case(self):
        inputs = {"key1": 1, "key2": 2}
        func = lambda x: x + 1
        outputs = utils.map_structure(func, inputs)
        utils.assert_same_structure(inputs, outputs)
        try:
            inputs["key3"] = 3
            utils.assert_same_structure(inputs, outputs)
        except ValueError as identifier:
            pass


if __name__ == '__main__':
    unittest.main()
