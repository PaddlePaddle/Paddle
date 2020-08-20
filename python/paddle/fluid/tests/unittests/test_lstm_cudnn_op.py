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

import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.layers as layers

SIGMOID_THRESHOLD_MIN = -40.0
SIGMOID_THRESHOLD_MAX = 13.0
EXP_MAX_INPUT = 40.0


def lstm_naive(input, w):
    seq_len, batch_size, hidden_size = input.shape

    offset = 0
    wi = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    wf = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    wc = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    wo = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    ri = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    rf = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    rc = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    ro = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size

    bi_1 = w[offset:offset + hidden_size]
    offset += hidden_size
    bf_1 = w[offset:offset + hidden_size]
    offset += hidden_size
    bc_1 = w[offset:offset + hidden_size]
    offset += hidden_size
    bo_1 = w[offset:offset + hidden_size]
    offset += hidden_size

    bi_2 = w[offset:offset + hidden_size]
    offset += hidden_size
    bf_2 = w[offset:offset + hidden_size]
    offset += hidden_size
    bc_2 = w[offset:offset + hidden_size]
    offset += hidden_size
    bo_2 = w[offset:offset + hidden_size]

    def sigmoid(x):
        y = np.copy(x)
        y[x < SIGMOID_THRESHOLD_MIN] = SIGMOID_THRESHOLD_MIN
        y[x > SIGMOID_THRESHOLD_MAX] = SIGMOID_THRESHOLD_MAX
        return 1. / (1. + np.exp(-y))

    def tanh(x):
        y = -2. * x
        y[y > EXP_MAX_INPUT] = EXP_MAX_INPUT
        return (2. / (1. + np.exp(y))) - 1.

    output = []
    pre_h = np.zeros((1, batch_size, hidden_size), dtype=input.dtype)
    pre_c = np.zeros((1, batch_size, hidden_size), dtype=input.dtype)

    for i in range(seq_len):
        emb_1 = input[i]

        input_gate = sigmoid(
            np.matmul(emb_1, wi) + np.matmul(pre_h, ri) + bi_1 + bi_2)
        forget_gate = sigmoid(
            np.matmul(emb_1, wf) + np.matmul(pre_h, rf) + bf_1 + bf_2)
        output_gate = sigmoid(
            np.matmul(emb_1, wo) + np.matmul(pre_h, ro) + bo_1 + bo_2)
        c_t_temp = tanh(
            np.matmul(emb_1, wc) + np.matmul(pre_h, rc) + bc_1 + bc_2)
        new_c = input_gate * c_t_temp + forget_gate * pre_c
        new_h = output_gate * tanh(new_c)

        pre_h = new_h
        pre_c = new_c

        output.append(new_h)

    output = np.concatenate(output, -1)
    output = output.reshape((batch_size, -1, hidden_size))
    output = output.transpose((1, 0, 2))

    return output, pre_h, pre_c


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNLstmOp(OpTest):
    # TODO(GaoWei8):when input dtype is fp64, precision threshold should be removed.
    def setUp(self):
        self.op_type = "cudnn_lstm"
        self.dtype = np.float64

        seq_length = 20
        batch_size = 5
        hidden_size = 20

        input_weight_size = (hidden_size * hidden_size) * 4
        hidden_weight_size = (hidden_size * hidden_size) * 4
        weight_size = input_weight_size + hidden_weight_size
        weight_size += hidden_size * 8

        input = np.random.uniform(
            low=-0.1, high=0.1, size=(seq_length, batch_size,
                                      hidden_size)).astype(self.dtype)
        flat_w = np.random.uniform(
            low=-0.1, high=0.1, size=(weight_size)).astype(self.dtype)

        output, last_hidden, last_cell = lstm_naive(input, flat_w)

        init_h = np.zeros((1, batch_size, hidden_size), dtype=np.float64)
        init_c = np.zeros((1, batch_size, hidden_size), dtype=np.float64)
        state_out = np.ndarray((300)).astype("uint8")

        self.inputs = {
            'Input': input,
            'W': flat_w,
            'InitH': init_h,
            'InitC': init_c
        }
        self.attrs = {
            'dropout_prob': 0.0,
            'is_bidirec': False,
            'input_size': hidden_size,
            'hidden_size': hidden_size,
            'num_layers': 1,
        }
        self.outputs = {
            'Out': output,
            "LastH": last_hidden,
            'LastC': last_cell,
            'Reserve': np.ndarray((400)).astype("uint8"),
            'StateOut': state_out
        }

    def test_output_with_place(self):
        # depend on the scope structure
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, no_check_set=['Reserve', 'StateOut'])

    def test_grad_with_place(self):
        # depend on the scope structure
        place = core.CUDAPlace(0)
        self.check_grad_with_place(
            place,
            set(['Input', 'W', 'InitH', 'InitC']), ['Out', 'LastH', 'LastC'],
            max_relative_error=1e-4)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNlstmAPI(unittest.TestCase):
    def test_lstm(self):
        seq_len = 20
        batch_size = 5
        hidden_size = 20
        dropout_prob = 0.0
        num_layers = 1
        input = fluid.data(
            name='input',
            shape=[seq_len, batch_size, hidden_size],
            dtype='float64')
        init_h = layers.fill_constant([num_layers, batch_size, hidden_size],
                                      'float64', 0.0)
        init_c = layers.fill_constant([num_layers, batch_size, hidden_size],
                                      'float64', 0.0)
        rnn_out, last_h, last_c = layers.lstm(input, init_h, init_c, seq_len,
                                              hidden_size, num_layers,
                                              dropout_prob)
        exe = fluid.Executor(fluid.CUDAPlace(0))
        exe.run(fluid.default_startup_program())
        input_i = np.random.uniform(
            low=-0.1, high=0.1, size=(seq_len, batch_size,
                                      hidden_size)).astype("float64")
        out = exe.run(fluid.default_main_program(),
                      feed={'input': input_i},
                      fetch_list=[rnn_out, last_h, last_c, 'cudnn_lstm_0.w_0'])

        output, last_hidden, last_cell = lstm_naive(input_i, out[3])

        self.assertTrue(np.allclose(output, out[0], atol=1e-5))
        self.assertTrue(np.allclose(last_hidden, out[1], atol=1e-5))
        self.assertTrue(np.allclose(last_cell, out[2], atol=1e-5))


if __name__ == '__main__':
    unittest.main()
