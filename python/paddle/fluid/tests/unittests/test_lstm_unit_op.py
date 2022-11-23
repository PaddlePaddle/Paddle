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

import unittest
import numpy as np
from op_test import OpTest
from paddle import fluid
from paddle.fluid.layers import lstm_unit
from paddle.fluid.framework import program_guard, Program


def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))


def tanh_np(x):
    return 2 * sigmoid_np(2. * x) - 1.


class LstmUnitTestError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            batch_size, dict_dim, emb_dim, hidden_dim = 32, 128, 64, 512
            data = fluid.data(name='step_data',
                              shape=[batch_size],
                              dtype='int64')
            inputs = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            pre_hidden = fluid.data(name='pre_hidden',
                                    shape=[batch_size, hidden_dim],
                                    dtype='float32')
            pre_cell = fluid.data(name='pre_cell',
                                  shape=[batch_size, hidden_dim],
                                  dtype='float32')

            np_input = np.random.uniform(
                -0.1, 0.1, (batch_size, emb_dim)).astype('float64')
            np_pre_hidden = np.random.uniform(
                -0.1, 0.1, (batch_size, hidden_dim)).astype('float64')
            np_pre_cell = np.random.uniform(
                -0.1, 0.1, (batch_size, hidden_dim)).astype('float64')

            def test_input_Variable():
                lstm_unit(np_input, pre_hidden, pre_cell)

            self.assertRaises(TypeError, test_input_Variable)

            def test_pre_hidden_Variable():
                lstm_unit(inputs, np_pre_hidden, pre_cell)

            self.assertRaises(TypeError, test_pre_hidden_Variable)

            def test_pre_cell_Variable():
                lstm_unit(inputs, pre_hidden, np_pre_cell)

            self.assertRaises(TypeError, test_pre_cell_Variable)

            def test_input_type():
                error_input = fluid.data(name='error_input',
                                         shape=[batch_size, emb_dim],
                                         dtype='int32')
                lstm_unit(error_input, pre_hidden, pre_cell)

            self.assertRaises(TypeError, test_input_type)

            def test_pre_hidden_type():
                error_pre_hidden = fluid.data(name='error_pre_hidden',
                                              shape=[batch_size, hidden_dim],
                                              dtype='int32')
                lstm_unit(inputs, error_pre_hidden, pre_cell)

            self.assertRaises(TypeError, test_pre_hidden_type)

            def test_pre_cell_type():
                error_pre_cell = fluid.data(name='error_pre_cell',
                                            shape=[batch_size, hidden_dim],
                                            dtype='int32')
                lstm_unit(inputs, pre_hidden, error_pre_cell)

            self.assertRaises(TypeError, test_pre_cell_type)


class LstmUnitTest(OpTest):

    def setUp(self):
        self.op_type = "lstm_unit"
        x_np = np.random.normal(size=(15, 160)).astype("float64")
        c_np = np.random.normal(size=(15, 40)).astype("float64")
        i_np, f_np, o_np, j_np = np.split(x_np, 4, axis=1)
        forget_bias_np = 0.
        self.attrs = {'forget_bias': 0.}

        new_c = c_np * sigmoid_np(f_np + forget_bias_np) + sigmoid_np(
            i_np) * tanh_np(j_np)
        new_h = tanh_np(new_c) * sigmoid_np(o_np)

        self.inputs = {'X': x_np, 'C_prev': c_np}
        self.outputs = {'C': new_c, 'H': new_h}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'C_prev'], ['C', 'H'])


if __name__ == "__main__":
    unittest.main()
