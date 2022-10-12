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
from paddle.fluid.contrib.layers import basic_lstm
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


def lstm_np(input,
            init_h,
            init_c,
            hidden_size,
            gate_weight,
            gate_bias,
            num_layers=1,
            batch_first=False,
            is_bidirect=False,
            sequence_length=None,
            forget_bias=1.0):

    def step(step_in, pre_hidden, pre_cell, gate_w, gate_b):
        concat_1 = np.concatenate([step_in, pre_hidden], 1)

        gate_input = np.matmul(concat_1, gate_w)
        gate_input += gate_b
        i, j, f, o = np.split(gate_input, indices_or_sections=4, axis=1)

        new_cell = pre_cell * sigmoid(f + forget_bias) + sigmoid(i) * tanh(j)
        new_hidden = tanh(new_cell) * sigmoid(o)

        return new_hidden, new_cell

    mask = None

    if batch_first:
        input = np.tranpose(input, [1, 0, 2])
        if mask is not None:
            mask = np.transpose(mask, [1, 0])

    batch_size = input.shape[1]
    if sequence_length is not None:
        max_seq_len = input.shape[0]

        mask = np.zeros([batch_size, max_seq_len])

        for i, len in enumerate(sequence_length):
            mask[i, :len] = 1.0

        mask = np.transpose(mask, [1, 0])

    direc_num = 1
    if is_bidirect:
        direc_num = 2
    if init_h:
        init_h = np.reshape(init_h, [num_layers, direc_num, -1, hidden_size])
        init_c = np.reshape(init_c, [num_layers, direc_num, -1, hidden_size])
    else:
        init_h = np.zeros([num_layers, direc_num, batch_size, hidden_size])
        init_c = np.zeros([num_layers, direc_num, batch_size, hidden_size])

    def get_single_direction_output(rnn_input, mask=None, direc_index=0):
        seq_len = rnn_input.shape[0]

        output = []
        # init pre hidden
        pre_hidden_array = []
        pre_cell_array = []
        for i in range(num_layers):
            pre_hidden_array.append(init_h[i, direc_index])
            pre_cell_array.append(init_c[i, direc_index])

        for i in range(seq_len):
            step_input = rnn_input[i]

            if mask is not None:
                step_mask = mask[i]
                step_mask = np.reshape(step_mask, [-1, 1])
                #print("np mask", step_mask.shape  )

            for i in range(num_layers):
                new_hidden, new_cell = step(
                    step_input, pre_hidden_array[i], pre_cell_array[i],
                    gate_weight[direc_index * num_layers + i],
                    gate_bias[direc_index * num_layers + i])

                if mask is not None:

                    new_hidden = np.multiply(
                        new_hidden, step_mask) - np.multiply(
                            pre_hidden_array[i], (step_mask - 1.0))
                    #new_hidden = new_hidden * step_mask - pre_hidden_array[i] * ( step_mask -1 )
                    #new_cell = new_cell * step_mask - pre_cell_array[i] * (step_mask -1)
                    new_cell = np.multiply(new_cell, step_mask) - np.multiply(
                        pre_cell_array[i], (step_mask - 1.0))

                pre_hidden_array[i] = new_hidden
                pre_cell_array[i] = new_cell

                step_input = new_hidden
            output.append(step_input)
        rnn_out = np.concatenate(output, 0)
        rnn_out = np.reshape(rnn_out, [seq_len, -1, hidden_size])

        last_hidden_out = np.concatenate(pre_hidden_array, 0)
        last_hidden_out = np.reshape(last_hidden_out,
                                     [num_layers, -1, hidden_size])

        last_cell_out = np.concatenate(pre_cell_array, 0)
        last_cell_out = np.reshape(last_cell_out, [num_layers, -1, hidden_size])

        return rnn_out, last_hidden_out, last_cell_out

    fw_rnn_out, fw_last_hidden, fw_last_cell = get_single_direction_output(
        input, mask, direc_index=0)

    if is_bidirect:
        bw_input = input[::-1]
        bw_mask = None
        if mask is not None:
            bw_mask = mask[::-1]

        bw_rnn_out, bw_last_hidden, bw_last_cell = get_single_direction_output(
            bw_input, bw_mask, direc_index=1)

        bw_rnn_out = bw_rnn_out[::-1]

        rnn_out = np.concatenate([fw_rnn_out, bw_rnn_out], 2)
        last_hidden = np.concatenate([fw_last_hidden, bw_last_hidden], 1)
        last_hidden = np.reshape(last_hidden,
                                 [num_layers * direc_num, -1, hidden_size])

        last_cell = np.concatenate([fw_last_cell, bw_last_cell], 1)
        last_cell = np.reshape(last_cell,
                               [num_layers * direc_num, -1, hidden_size])

        if batch_first:
            rnn_out = np.transpose(rnn_out, [1, 0, 2])

        return rnn_out, last_hidden, last_cell
    else:
        rnn_out = fw_rnn_out
        last_hidden = fw_last_hidden
        last_cell = fw_last_cell

        if batch_first:
            rnn_out = np.transpose(rnn_out, [1, 0, 2])

        return rnn_out, last_hidden, last_cell


class TestBasicLSTMApi(unittest.TestCase):

    def setUp(self):
        self.hidden_size = 10
        self.batch_size = 5
        self.seq_len = 6
        self.num_layers = 2
        self.is_bidirect = True
        self.batch_first = False
        self.forget_bias = 1.0

    def test_run(self):
        x = layers.data(name='x',
                        shape=[-1, self.batch_size, self.hidden_size],
                        dtype='float32')
        sequence_length = layers.data(name="sequence_length",
                                      shape=[-1],
                                      dtype='float32')

        rnn_out, last_hidden, last_cell = basic_lstm( x, None, None, self.hidden_size, num_layers=self.num_layers, \
                batch_first = self.batch_first, bidirectional=self.is_bidirect, sequence_length=sequence_length, forget_bias = self.forget_bias )

        last_hidden.persisbale = True
        rnn_out.persisbale = True

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = Executor(place)
        exe.run(framework.default_startup_program())

        param_list = fluid.default_main_program().block(0).all_parameters()

        # process weight and bias
        gate_weight = []
        gate_bias = []

        for i in range(self.num_layers):
            gate_w_name = "basic_lstm_layers_" + str(i) + "/BasicLSTMUnit_0.w_0"
            gate_b_name = "basic_lstm_layers_" + str(i) + "/BasicLSTMUnit_0.b_0"

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

            gate_weight.append(gate_w)
            gate_bias.append(gate_b)

        if self.is_bidirect:
            for i in range(self.num_layers):
                gate_w_name = "basic_lstm_reverse_layers_" + str(
                    i) + "/BasicLSTMUnit_0.w_0"
                gate_b_name = "basic_lstm_reverse_layers_" + str(
                    i) + "/BasicLSTMUnit_0.b_0"

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

                gate_weight.append(gate_w)
                gate_bias.append(gate_b)

        step_input_np = np.random.uniform(
            -0.1, 0.1,
            (self.seq_len, self.batch_size, self.hidden_size)).astype('float32')
        sequence_length_np = np.random.randint(
            self.seq_len // 2, self.seq_len,
            size=(self.batch_size)).astype('int64')

        out = exe.run(feed={
            'x': step_input_np,
            'sequence_length': sequence_length_np
        },
                      fetch_list=[rnn_out, last_hidden, last_cell])

        api_rnn_out = out[0]
        api_last_hidden = out[1]
        api_last_cell = out[2]

        np_out = lstm_np(step_input_np,
                         None,
                         None,
                         self.hidden_size,
                         gate_weight,
                         gate_bias,
                         num_layers=self.num_layers,
                         batch_first=self.batch_first,
                         is_bidirect=self.is_bidirect,
                         sequence_length=sequence_length_np)

        np.testing.assert_allclose(api_rnn_out, np_out[0], rtol=0.0001, atol=0)
        np.testing.assert_allclose(api_last_hidden,
                                   np_out[1],
                                   rtol=0.0001,
                                   atol=0)
        np.testing.assert_allclose(api_last_cell,
                                   np_out[2],
                                   rtol=0.0001,
                                   atol=0)


if __name__ == '__main__':
    unittest.main()
