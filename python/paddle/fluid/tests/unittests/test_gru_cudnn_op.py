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

SIGMOID_THRESHOLD_MIN = -40
SIGMOID_THRESHOLD_MAX = 13
EXP_MAX_INPUT = 40
np.set_printoptions(threshold=1e6, suppress=True)


def gru_naive(
        input,
        w, ):
    '''Compute native gru for reference.'''
    seq_len, batch_size, hidden_size = input.shape

    offset = 0
    wu = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    wr = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    wc = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size

    ru = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    rr = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size
    rc = w[offset:offset + hidden_size * hidden_size].reshape(
        (hidden_size, hidden_size)).transpose()
    offset += hidden_size * hidden_size

    bx_u = w[offset:offset + hidden_size]
    offset += hidden_size
    bx_r = w[offset:offset + hidden_size]
    offset += hidden_size
    bx_c = w[offset:offset + hidden_size]
    offset += hidden_size

    bh_u = w[offset:offset + hidden_size]
    offset += hidden_size
    bh_r = w[offset:offset + hidden_size]
    offset += hidden_size
    bh_c = w[offset:offset + hidden_size]
    offset += hidden_size

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
    pre_h = np.zeros((batch_size, hidden_size), dtype=input.dtype)

    for i in range(seq_len):
        emb_1 = input[i]

        reset_gate = sigmoid(
            np.matmul(emb_1, wu) + np.matmul(pre_h, ru) + bx_u + bh_u)
        update_gate = sigmoid(
            np.matmul(emb_1, wr) + np.matmul(pre_h, rr) + bx_r + bh_r)
        tmp1 = np.matmul(emb_1, wc)
        tmp2 = np.matmul(pre_h, rc) + bh_c
        tmp3 = tmp2 * reset_gate

        h_t_temp = tanh(tmp1 + tmp3 + bx_c)
        new_h = update_gate * pre_h + (1 - update_gate) * h_t_temp

        pre_h = new_h
        output.append(new_h)
    output = np.concatenate(output, -1)
    output = output.reshape((batch_size, -1, hidden_size))
    output = output.transpose((1, 0, 2))
    return output, pre_h


class TestCUDNNGruOp(OpTest):
    def setUp(self):
        self.op_type = "cudnn_gru"
        self.dtype = np.float32

        num_steps = 20
        batch_size = 5
        hidden_size = 20

        input_weight_size = (hidden_size * hidden_size) * 3
        hidden_weight_size = (hidden_size * hidden_size) * 3
        weight_size = input_weight_size + hidden_weight_size
        weight_size += hidden_size * 6

        input = np.random.uniform(
            low=-0.1, high=0.1, size=(num_steps, batch_size,
                                      hidden_size)).astype(self.dtype)
        flat_w = np.random.uniform(
            low=-0.1, high=0.1, size=(weight_size)).astype(self.dtype)

        output, last_hidden = gru_naive(input, flat_w)

        init_h = np.zeros((batch_size, hidden_size), dtype=np.float32)
        scope = core.Scope()
        program = fluid.Program()
        block = program.global_block()

        cache_temp = block.create_var(
            name="Cache",
            persistable=True,
            type=core.VarDesc.VarType.RAW,
            stop_gradient=True)
        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'W': OpTest.np_dtype_to_fluid_dtype(flat_w),
            'InitH': OpTest.np_dtype_to_fluid_dtype(init_h),
        }
        self.cache_name_list = ['Cache']
        self.attrs = {
            'max_len': num_steps,
            'dropout_prob': 0.0,
            'is_bidirec': False,
            'input_size': hidden_size,
            'hidden_size': hidden_size,
            'num_layers': 1,
        }
        self.outputs = {'Out': output, "last_h": last_hidden}

    def test_output_with_place(self):
        if self.testcuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)

    def test_grad_with_place(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                set(['Input', 'W', 'InitH']), ['Out', 'last_h'],
                max_relative_error=0.02)

    def testcuda(self):
        return core.is_compiled_with_cuda()


if __name__ == '__main__':
    unittest.main()
