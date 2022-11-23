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

import unittest
import numpy
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from paddle.fluid.contrib.layers import BasicLSTMUnit
from paddle.fluid.executor import Executor
from paddle.fluid import framework

import numpy as np

np.set_seed(123)

SIGMOID_THRESHOLD_MIN = -40.0
SIGMOID_THRESHOLD_MAX = 13.0
EXP_MAX_INPUT = 40.0


def sigmoid(x):
    y = np.copy(x)
    y[x < SIGMOID_THRESHOLD_MIN] = SIGMOID_THRESHOLD_MIN
    y[x > SIGMOID_THRESHOLD_MAX] = SIGMOID_THRESHOLD_MAX
    return 1. / (1. + np.exp(-y))


def tanh(x):
    y = -2. * x
    y[y > EXP_MAX_INPUT] = EXP_MAX_INPUT
    return (2. / (1. + np.exp(y))) - 1.


def step(step_in, pre_hidden, pre_cell, gate_w, gate_b, forget_bias=1.0):
    concat_1 = np.concatenate([step_in, pre_hidden], 1)

    gate_input = np.matmul(concat_1, gate_w)
    gate_input += gate_b
    i, j, f, o = np.split(gate_input, indices_or_sections=4, axis=1)

    new_cell = pre_cell * sigmoid(f + forget_bias) + sigmoid(i) * tanh(j)
    new_hidden = tanh(new_cell) * sigmoid(o)

    return new_hidden, new_cell


class TestBasicGRUUnit(unittest.TestCase):

    def setUp(self):
        self.hidden_size = 5
        self.batch_size = 5

    def test_run(self):
        x = layers.data(name='x', shape=[-1, self.hidden_size], dtype='float32')
        pre_hidden = layers.data(name="pre_hidden",
                                 shape=[-1, self.hidden_size],
                                 dtype='float32')
        pre_cell = layers.data(name="pre_cell",
                               shape=[-1, self.hidden_size],
                               dtype='float32')

        lstm_unit = BasicLSTMUnit("lstm_unit", self.hidden_size)

        new_hidden, new_cell = lstm_unit(x, pre_hidden, pre_cell)

        new_hidden.persisbale = True
        new_cell.persisbale = True

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        exe = Executor(place)
        exe.run(framework.default_startup_program())

        param_list = fluid.default_main_program().block(0).all_parameters()

        # process weight and bias

        gate_w_name = "lstm_unit/BasicLSTMUnit_0.w_0"
        gate_b_name = "lstm_unit/BasicLSTMUnit_0.b_0"

        gate_w = np.array(
            fluid.global_scope().find_var(gate_w_name).get_tensor())
        gate_w = np.random.uniform(-0.1, 0.1,
                                   size=gate_w.shape).astype('float32')
        fluid.global_scope().find_var(gate_w_name).get_tensor().set(
            gate_w, place)

        gate_b = np.array(
            fluid.global_scope().find_var(gate_b_name).get_tensor())
        gate_b = np.random.uniform(-0.1, 0.1,
                                   size=gate_b.shape).astype('float32')
        fluid.global_scope().find_var(gate_b_name).get_tensor().set(
            gate_b, place)

        step_input_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')
        pre_cell_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')

        out = exe.run( feed={ 'x' : step_input_np, 'pre_hidden' : pre_hidden_np, \
                              'pre_cell' : pre_cell_np },
                fetch_list=[ new_hidden, new_cell])

        api_hidden_out = out[0]
        api_cell_out = out[1]

        np_hidden_out, np_cell_out = step(step_input_np, pre_hidden_np,
                                          pre_cell_np, gate_w, gate_b)

        np.testing.assert_allclose(api_hidden_out,
                                   np_hidden_out,
                                   rtol=0.0001,
                                   atol=0)
        np.testing.assert_allclose(api_cell_out,
                                   np_cell_out,
                                   rtol=0.0001,
                                   atol=0)


if __name__ == '__main__':
    unittest.main()
