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

from __future__ import print_function

import unittest
import numpy
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph.rnn import CudnnGRU

import numpy as np

np.random.seed = 123

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def tanh(x):
    return 2. * sigmoid(2. * x) - 1.

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


class TestCudnnGRU(unittest.TestCase):
    def setUp(self):
        self.input_size = 100
        self.hidden_size = 200
        self.batch_size = 64

    def test_run(self):

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with fluid.dygraph.guard(place):
            
            cudnn_gru = CudnnGRU(self.hidden_size, self.input_size)

            param_list = cudnn_gru.state_dict()

            # process weight and bias

            gate_w_name = "_gate_weight"
            gate_b_name = "_gate_bias"
            candidate_w_name = "_candidate_weight"
            candidate_b_name = "_candidate_bias"

            gate_w = param_list[gate_w_name].numpy()
            gate_w = np.random.uniform(
                -0.1, 0.1, size=gate_w.shape).astype('float64')
            param_list[gate_w_name].set_value(gate_w)

            gate_b = param_list[gate_b_name].numpy()
            gate_b = np.random.uniform(
                -0.1, 0.1, size=gate_b.shape).astype('float64')
            param_list[gate_b_name].set_value(gate_b)

            candidate_w = param_list[candidate_w_name].numpy()
            candidate_w = np.random.uniform(
                -0.1, 0.1, size=candidate_w.shape).astype('float64')
            param_list[candidate_w_name].set_value(candidate_w)

            candidate_b = param_list[candidate_b_name].numpy()
            candidate_b = np.random.uniform(
                -0.1, 0.1, size=candidate_b.shape).astype('float64')
            param_list[candidate_b_name].set_value(candidate_b)

            step_input_np = np.random.uniform(-0.1, 0.1, (
                self.batch_size, self.input_size)).astype('float64')
            pre_hidden_np = np.random.uniform(-0.1, 0.1, (
                self.batch_size, self.hidden_size)).astype('float64')
            
            step_input_var = fluid.dygraph.to_variable(step_input_np)
            pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)
            api_out = cudnn_gru(step_input_var, pre_hidden_var)

        np_out = step(step_input_np, pre_hidden_np, gate_w, gate_b, candidate_w,
                      candidate_b)

        self.assertTrue(np.allclose(api_out.numpy(), np_out, rtol=1e-5, atol=0))
        

if __name__ == '__main__':
    unittest.main()
