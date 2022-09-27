# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid.contrib.layers import basic_gru, basic_lstm
from paddle.fluid.executor import Executor
from paddle.fluid import framework
from test_imperative_base import new_program_scope
import numpy as np


class TestBasicGRUApiName(unittest.TestCase):

    def setUp(self):
        self.name_set = set([
            "test1_fw_w_0_gate", "test1_fw_w_0_candidate", "test1_fw_b_0_gate",
            "test1_fw_b_0_candidate", "test1_bw_w_0_gate",
            "test1_bw_w_0_candidate", "test1_bw_b_0_gate",
            "test1_bw_b_0_candidate"
        ])

    def test_name(self):
        batch_size = 20
        input_size = 128
        hidden_size = 256
        num_layers = 1
        dropout = 0.5
        bidirectional = True
        batch_first = False

        with new_program_scope():
            input = layers.data(name="input",
                                shape=[-1, batch_size, input_size],
                                dtype='float32')
            pre_hidden = layers.data(name="pre_hidden",
                                     shape=[-1, hidden_size],
                                     dtype='float32')
            sequence_length = layers.data(name="sequence_length",
                                          shape=[-1],
                                          dtype='int32')


            rnn_out, last_hidden = basic_gru( input, pre_hidden, hidden_size, num_layers = num_layers, \
                sequence_length = sequence_length, dropout_prob=dropout, bidirectional = bidirectional, \
                batch_first = batch_first, param_attr=fluid.ParamAttr( name ="test1"), bias_attr=fluid.ParamAttr( name="test1"), name="basic_gru")

            var_list = fluid.io.get_program_parameter(
                fluid.default_main_program())

            for var in var_list:
                self.assertTrue(var.name in self.name_set)


class TestBasicLSTMApiName(unittest.TestCase):

    def setUp(self):
        self.name_set = set([
            "test1_fw_w_0", "test1_fw_b_0", "test1_fw_w_1", "test1_fw_b_1",
            "test1_bw_w_0", "test1_bw_b_0", "test1_bw_w_1", "test1_bw_b_1"
        ])

    def test_name(self):
        batch_size = 20
        input_size = 128
        hidden_size = 256
        num_layers = 2
        dropout = 0.5
        bidirectional = True
        batch_first = False

        with new_program_scope():
            input = layers.data(name="input",
                                shape=[-1, batch_size, input_size],
                                dtype='float32')
            pre_hidden = layers.data(name="pre_hidden",
                                     shape=[-1, hidden_size],
                                     dtype='float32')
            pre_cell = layers.data(name="pre_cell",
                                   shape=[-1, hidden_size],
                                   dtype='float32')
            sequence_length = layers.data(name="sequence_length",
                                          shape=[-1],
                                          dtype='int32')

            rnn_out, last_hidden, last_cell = basic_lstm( input, pre_hidden, pre_cell, \
                hidden_size, num_layers = num_layers, \
                sequence_length = sequence_length, dropout_prob=dropout, bidirectional = bidirectional, \
                param_attr=fluid.ParamAttr( name ="test1"), bias_attr=fluid.ParamAttr( name = "test1"),  \
                batch_first = batch_first)

            var_list = fluid.io.get_program_parameter(
                fluid.default_main_program())

            for var in var_list:
                self.assertTrue(var.name in self.name_set)


if __name__ == '__main__':
    unittest.main()
