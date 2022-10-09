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
from paddle.fluid.contrib.layers import BasicGRUUnit
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


def step(step_in, pre_hidden, gate_w, gate_b, candidate_w, candidate_b):
    concat_1 = np.concatenate([step_in, pre_hidden], 1)

    gate_input = np.matmul(concat_1, gate_w)
    gate_input += gate_b
    gate_input = sigmoid(gate_input)
    r, u = np.split(gate_input, indices_or_sections=2, axis=1)

    r_hidden = r * pre_hidden

    candidate = np.matmul(np.concatenate([step_in, r_hidden], 1), candidate_w)

    candidate += candidate_b
    c = tanh(candidate)

    new_hidden = u * pre_hidden + (1 - u) * c

    return new_hidden


class TestBasicGRUUnit(unittest.TestCase):

    def setUp(self):
        self.hidden_size = 5
        self.batch_size = 5

    def test_run(self):
        x = layers.data(name='x', shape=[-1, self.hidden_size], dtype='float32')
        pre_hidden = layers.data(name="pre_hidden",
                                 shape=[-1, self.hidden_size],
                                 dtype='float32')
        gru_unit = BasicGRUUnit("gru_unit", self.hidden_size)

        new_hidden = gru_unit(x, pre_hidden)

        new_hidden.persisbale = True

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        exe = Executor(place)
        exe.run(framework.default_startup_program())

        param_list = fluid.default_main_program().block(0).all_parameters()

        # process weight and bias

        gate_w_name = "gru_unit/BasicGRUUnit_0.w_0"
        gate_b_name = "gru_unit/BasicGRUUnit_0.b_0"
        candidate_w_name = "gru_unit/BasicGRUUnit_0.w_1"
        candidate_b_name = "gru_unit/BasicGRUUnit_0.b_1"

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

        candidate_w = np.array(
            fluid.global_scope().find_var(candidate_w_name).get_tensor())
        candidate_w = np.random.uniform(
            -0.1, 0.1, size=candidate_w.shape).astype('float32')
        fluid.global_scope().find_var(candidate_w_name).get_tensor().set(
            candidate_w, place)

        candidate_b = np.array(
            fluid.global_scope().find_var(candidate_b_name).get_tensor())
        candidate_b = np.random.uniform(
            -0.1, 0.1, size=candidate_b.shape).astype('float32')
        fluid.global_scope().find_var(candidate_b_name).get_tensor().set(
            candidate_b, place)

        step_input_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')
        pre_hidden_np = np.random.uniform(
            -0.1, 0.1, (self.batch_size, self.hidden_size)).astype('float32')

        out = exe.run(feed={
            'x': step_input_np,
            'pre_hidden': pre_hidden_np
        },
                      fetch_list=[new_hidden])

        api_out = out[0]

        np_out = step(step_input_np, pre_hidden_np, gate_w, gate_b, candidate_w,
                      candidate_b)

        np.testing.assert_allclose(api_out, np_out, rtol=0.0001, atol=0)


if __name__ == '__main__':
    unittest.main()
