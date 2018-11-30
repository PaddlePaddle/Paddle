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


def lstm_naive(
        input,
        w, ):
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
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    output = []
    pre_h = np.zeros((batch_size, hidden_size), dtype=input.dtype)
    pre_c = np.zeros((batch_size, hidden_size), dtype=input.dtype)

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

    return output


class TestCUDNNLstmOp(OpTest):
    def setUp(self):
        self.op_type = "cudnn_lstm"
        self.dtype = np.float32

        num_steps = 50
        batch_size = 20
        hidden_size = 200

        input_weight_size = (hidden_size * hidden_size) * 4
        hidden_weight_size = (hidden_size * hidden_size) * 4
        weight_size = input_weight_size + hidden_weight_size
        weight_size += hidden_size * 8

        input = np.random.random(
            (num_steps, batch_size, hidden_size)).astype(self.dtype)
        flat_w = np.random.random((weight_size)).astype(self.dtype)

        output = lstm_naive(input, flat_w)

        init_h = np.zeros((batch_size, hidden_size), dtype=np.float32)
        init_c = np.zeros((batch_size, hidden_size), dtype=np.float32)
        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'W': OpTest.np_dtype_to_fluid_dtype(flat_w),
            'InitH': OpTest.np_dtype_to_fluid_dtype(init_h),
            'InitC': OpTest.np_dtype_to_fluid_dtype(init_c),
        }
        self.attrs = {
            'max_len': num_steps,
            'dropout_prob': 0.0,
            'is_bidirec': False,
            'input_size': hidden_size,
            'hidden_size': hidden_size,
            'num_layers': 1,
        }
        self.outputs = {'Out': output}

    def test_grad_with_place(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, atol=1e-5)

    def test_output_with_place(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, atol=1e-5, no_check_set=['last_h', 'last_c'])
