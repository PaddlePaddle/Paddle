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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph import LSTMCell

import numpy as np

np.random.seed = 123


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def tanh(x):
    return 2. * sigmoid(2. * x) - 1.


def non_cudnn_step(step_in,
                   pre_hidden,
                   pre_cell,
                   gate_w,
                   gate_b,
                   forget_bias=1.0):
    concat_1 = np.concatenate([step_in, pre_hidden], 1)

    gate_input = np.matmul(concat_1, gate_w)
    gate_input += gate_b
    i, j, f, o = np.split(gate_input, indices_or_sections=4, axis=1)

    new_cell = pre_cell * sigmoid(f + forget_bias) + sigmoid(i) * tanh(j)
    new_hidden = tanh(new_cell) * sigmoid(o)

    return new_hidden, new_cell


def cudnn_step(step_input_np, pre_hidden_np, pre_cell_np, weight_ih, bias_ih,
               weight_hh, bias_hh):

    igates = np.matmul(step_input_np, weight_ih.transpose(1, 0))
    igates = igates + bias_ih
    hgates = np.matmul(pre_hidden_np, weight_hh.transpose(1, 0))
    hgates = hgates + bias_hh

    chunked_igates = np.split(igates, indices_or_sections=4, axis=1)
    chunked_hgates = np.split(hgates, indices_or_sections=4, axis=1)

    ingate = chunked_igates[0] + chunked_hgates[0]
    ingate = sigmoid(ingate)

    forgetgate = chunked_igates[1] + chunked_hgates[1]
    forgetgate = sigmoid(forgetgate)

    cellgate = chunked_igates[2] + chunked_hgates[2]
    cellgate = tanh(cellgate)

    outgate = chunked_igates[3] + chunked_hgates[3]
    outgate = sigmoid(outgate)

    new_cell = (forgetgate * pre_cell_np) + (ingate * cellgate)
    new_hidden = outgate * tanh(new_cell)

    return new_hidden, new_cell


class TestCudnnLSTM(unittest.TestCase):

    def setUp(self):
        self.input_size = 100
        self.hidden_size = 200
        self.batch_size = 128

    def test_run(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with fluid.dygraph.guard(place):
            param_attr = fluid.ParamAttr(name="param_attr")
            bias_attr = fluid.ParamAttr(name="bias_attr")
            named_cudnn_lstm = LSTMCell(self.hidden_size, self.input_size,
                                        param_attr, bias_attr)
            cudnn_lstm = LSTMCell(self.hidden_size, self.input_size)

            param_list = cudnn_lstm.state_dict()
            named_param_list = named_cudnn_lstm.state_dict()

            # process weight and bias

            weight_ih_name = "_weight_ih"
            bias_ih_name = "_bias_ih"
            weight_hh_name = "_weight_hh"
            bias_hh_name = "_bias_hh"
            weight_ih = param_list[weight_ih_name].numpy()
            weight_ih = np.random.uniform(
                -0.1, 0.1, size=weight_ih.shape).astype('float64')
            param_list[weight_ih_name].set_value(weight_ih)
            named_param_list[weight_ih_name].set_value(weight_ih)

            bias_ih = param_list[bias_ih_name].numpy()
            bias_ih = np.random.uniform(-0.1, 0.1,
                                        size=bias_ih.shape).astype('float64')
            param_list[bias_ih_name].set_value(bias_ih)
            named_param_list[bias_ih_name].set_value(bias_ih)

            weight_hh = param_list[weight_hh_name].numpy()
            weight_hh = np.random.uniform(
                -0.1, 0.1, size=weight_hh.shape).astype('float64')
            param_list[weight_hh_name].set_value(weight_hh)
            named_param_list[weight_hh_name].set_value(weight_hh)

            bias_hh = param_list[bias_hh_name].numpy()
            bias_hh = np.random.uniform(-0.1, 0.1,
                                        size=bias_hh.shape).astype('float64')
            param_list[bias_hh_name].set_value(bias_hh)
            named_param_list[bias_hh_name].set_value(bias_hh)

            step_input_np = np.random.uniform(
                -0.1, 0.1, (self.batch_size, self.input_size)).astype('float64')
            pre_hidden_np = np.random.uniform(
                -0.1, 0.1,
                (self.batch_size, self.hidden_size)).astype('float64')
            pre_cell_np = np.random.uniform(
                -0.1, 0.1,
                (self.batch_size, self.hidden_size)).astype('float64')

            step_input_var = fluid.dygraph.to_variable(step_input_np)
            pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)
            pre_cell_var = fluid.dygraph.to_variable(pre_cell_np)
            api_out = cudnn_lstm(step_input_var, pre_hidden_var, pre_cell_var)
            named_api_out = named_cudnn_lstm(step_input_var, pre_hidden_var,
                                             pre_cell_var)

            api_hidden_out = api_out[0]
            api_cell_out = api_out[1]
            named_api_hidden_out = named_api_out[0]
            named_api_cell_out = named_api_out[1]

            np_hidden_out, np_cell_out = cudnn_step(step_input_np,
                                                    pre_hidden_np, pre_cell_np,
                                                    weight_ih, bias_ih,
                                                    weight_hh, bias_hh)
            np.testing.assert_allclose(api_hidden_out.numpy(),
                                       np_hidden_out,
                                       rtol=1e-05,
                                       atol=0)
            np.testing.assert_allclose(api_cell_out.numpy(),
                                       np_cell_out,
                                       rtol=1e-05,
                                       atol=0)
            np.testing.assert_allclose(named_api_hidden_out.numpy(),
                                       np_hidden_out,
                                       rtol=1e-05,
                                       atol=0)
            np.testing.assert_allclose(named_api_cell_out.numpy(),
                                       np_cell_out,
                                       rtol=1e-05,
                                       atol=0)


class TestNonCudnnLSTM(unittest.TestCase):

    def setUp(self):
        self.input_size = 100
        self.hidden_size = 200
        self.batch_size = 128

    def test_run(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()

        with fluid.dygraph.guard(place):
            param_attr = fluid.ParamAttr(name="param_attr")
            bias_attr = fluid.ParamAttr(name="bias_attr")
            named_cudnn_lstm = LSTMCell(self.hidden_size,
                                        self.input_size,
                                        param_attr,
                                        bias_attr,
                                        use_cudnn_impl=False)
            cudnn_lstm = LSTMCell(self.hidden_size,
                                  self.input_size,
                                  use_cudnn_impl=False)

            param_list = cudnn_lstm.state_dict()
            named_param_list = named_cudnn_lstm.state_dict()

            # process weight and bias

            gate_w_name = "_weight"
            gate_b_name = "_bias"

            gate_w = param_list[gate_w_name].numpy()
            gate_w = np.random.uniform(-0.1, 0.1,
                                       size=gate_w.shape).astype('float64')
            param_list[gate_w_name].set_value(gate_w)
            named_param_list[gate_w_name].set_value(gate_w)

            gate_b = param_list[gate_b_name].numpy()
            gate_b = np.random.uniform(-0.1, 0.1,
                                       size=gate_b.shape).astype('float64')
            param_list[gate_b_name].set_value(gate_b)
            named_param_list[gate_b_name].set_value(gate_b)

            step_input_np = np.random.uniform(
                -0.1, 0.1, (self.batch_size, self.input_size)).astype('float64')
            pre_hidden_np = np.random.uniform(
                -0.1, 0.1,
                (self.batch_size, self.hidden_size)).astype('float64')
            pre_cell_np = np.random.uniform(
                -0.1, 0.1,
                (self.batch_size, self.hidden_size)).astype('float64')

            step_input_var = fluid.dygraph.to_variable(step_input_np)
            pre_hidden_var = fluid.dygraph.to_variable(pre_hidden_np)
            pre_cell_var = fluid.dygraph.to_variable(pre_cell_np)
            api_out = cudnn_lstm(step_input_var, pre_hidden_var, pre_cell_var)
            named_api_out = named_cudnn_lstm(step_input_var, pre_hidden_var,
                                             pre_cell_var)

            api_hidden_out = api_out[0]
            api_cell_out = api_out[1]
            named_api_hidden_out = named_api_out[0]
            named_api_cell_out = named_api_out[1]

            np_hidden_out, np_cell_out = non_cudnn_step(step_input_np,
                                                        pre_hidden_np,
                                                        pre_cell_np, gate_w,
                                                        gate_b)

            np.testing.assert_allclose(api_hidden_out.numpy(),
                                       np_hidden_out,
                                       rtol=1e-05,
                                       atol=0)
            np.testing.assert_allclose(api_cell_out.numpy(),
                                       np_cell_out,
                                       rtol=1e-05,
                                       atol=0)
            np.testing.assert_allclose(named_api_hidden_out.numpy(),
                                       np_hidden_out,
                                       rtol=1e-05,
                                       atol=0)
            np.testing.assert_allclose(named_api_cell_out.numpy(),
                                       np_cell_out,
                                       rtol=1e-05,
                                       atol=0)


if __name__ == '__main__':
    unittest.main()
