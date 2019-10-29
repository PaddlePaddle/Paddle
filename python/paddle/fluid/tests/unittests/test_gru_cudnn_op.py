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


def sigmoid(x):
    y = np.copy(x)
    y[x < SIGMOID_THRESHOLD_MIN] = SIGMOID_THRESHOLD_MIN
    y[x > SIGMOID_THRESHOLD_MAX] = SIGMOID_THRESHOLD_MAX
    return 1. / (1. + np.exp(-y))


def tanh(x):
    y = -2. * x
    y[y > EXP_MAX_INPUT] = EXP_MAX_INPUT
    return (2. / (1. + np.exp(y))) - 1.


def gru_reference(input, w, num_direction, num_layers):
    '''Compute bidirection gru for reference.'''
    seq_len, batch_size, hidden_size = input.shape

    def step(step_in, pre_h, wu, wr, wc, ru, rr, rc, bx_u, bx_r, bx_c, bh_u,
             bh_r, bh_c):
        reset_gate = sigmoid(
            np.matmul(step_in, wu) + np.matmul(pre_h, ru) + bx_u + bh_u)
        update_gate = sigmoid(
            np.matmul(step_in, wr) + np.matmul(pre_h, rr) + bx_r + bh_r)
        tmp1 = np.matmul(step_in, wc)
        tmp2 = np.matmul(pre_h, rc) + bh_c
        tmp3 = tmp2 * reset_gate

        h_t_temp = tanh(tmp1 + tmp3 + bx_c)
        new_h = update_gate * pre_h + (1 - update_gate) * h_t_temp

        return new_h

    offset = 0
    wu = []
    wr = []
    wc = []
    ru = []
    rr = []
    rc = []
    bx_u = []
    bx_r = []
    bx_c = []
    bh_u = []
    bh_r = []
    bh_c = []

    for j in range(num_layers):
        #w_shape = hidden_size if j == 0 else hidden_size * num_direction
        # for num_layers > 1, need rearrange w according to cudnn's flat w.
        # Refine it later
        w_shape = hidden_size
        for i in range(num_direction):
            wu.append(w[offset:offset + hidden_size * w_shape].reshape((
                hidden_size, w_shape)).transpose())
            offset += hidden_size * w_shape
            wr.append(w[offset:offset + hidden_size * w_shape].reshape((
                hidden_size, w_shape)).transpose())
            offset += hidden_size * w_shape
            wc.append(w[offset:offset + hidden_size * w_shape].reshape((
                hidden_size, w_shape)).transpose())
            offset += hidden_size * w_shape

            ru.append(w[offset:offset + hidden_size * hidden_size].reshape((
                hidden_size, hidden_size)).transpose())
            offset += hidden_size * hidden_size
            rr.append(w[offset:offset + hidden_size * hidden_size].reshape((
                hidden_size, hidden_size)).transpose())
            offset += hidden_size * hidden_size
            rc.append(w[offset:offset + hidden_size * hidden_size].reshape((
                hidden_size, hidden_size)).transpose())
            offset += hidden_size * hidden_size

    for j in range(num_layers):
        for i in range(num_direction):
            bx_u.append(w[offset:offset + hidden_size])
            offset += hidden_size
            bx_r.append(w[offset:offset + hidden_size])
            offset += hidden_size
            bx_c.append(w[offset:offset + hidden_size])
            offset += hidden_size

            bh_u.append(w[offset:offset + hidden_size])
            offset += hidden_size
            bh_r.append(w[offset:offset + hidden_size])
            offset += hidden_size
            bh_c.append(w[offset:offset + hidden_size])
            offset += hidden_size

    init_h = np.zeros(
        (num_layers, num_direction, batch_size, hidden_size), dtype=input.dtype)

    def get_single_direction_output(rnn_input, direc_index=0):
        seq_len = rnn_input.shape[0]

        output = []
        pre_hidden_array = []

        for i in range(num_layers):
            pre_hidden_array.append(init_h[i, direc_index])

        for i in range(seq_len):
            step_input = rnn_input[i]

            for i in range(num_layers):
                new_hidden = step(
                    step_input,
                    pre_hidden_array[i],
                    wu[i * num_direction + direc_index],
                    wr[i * num_direction + direc_index],
                    wc[i * num_direction + direc_index],
                    ru[i * num_direction + direc_index],
                    rr[i * num_direction + direc_index],
                    rc[i * num_direction + direc_index],
                    bx_u[i * num_direction + direc_index],
                    bx_r[i * num_direction + direc_index],
                    bx_c[i * num_direction + direc_index],
                    bh_u[i * num_direction + direc_index],
                    bh_r[i * num_direction + direc_index],
                    bh_c[i * num_direction + direc_index], )

                pre_hidden_array[i] = new_hidden
                step_input = new_hidden
            output.append(step_input)

        rnn_out = np.concatenate(output, 0)
        rnn_out = np.reshape(rnn_out, [seq_len, -1, hidden_size])

        last_hidden_out = np.concatenate(pre_hidden_array, 0)
        last_hidden_out = np.reshape(last_hidden_out,
                                     [num_layers, -1, hidden_size])

        return rnn_out, last_hidden_out

    fw_rnn_out, fw_last_hidden = get_single_direction_output(
        input, direc_index=0)

    if num_direction == 2:
        bw_input = input[::-1]

        bw_rnn_out, bw_last_hidden = get_single_direction_output(
            bw_input, direc_index=1)

        bw_rnn_out = bw_rnn_out[::-1]

        rnn_out = np.concatenate([fw_rnn_out, bw_rnn_out], 2)
        last_hidden = np.concatenate([fw_last_hidden, bw_last_hidden], 1)
        last_hidden = np.reshape(last_hidden,
                                 [num_layers * num_direction, -1, hidden_size])
        return rnn_out, last_hidden
    else:
        rnn_out = fw_rnn_out
        last_hidden = fw_last_hidden

        return rnn_out, last_hidden


class TestCUDNNGruOp(OpTest):
    def config(self):
        self.num_steps = 2
        self.batch_size = 3
        self.hidden_size = 4
        self.bidirec = True
        self.num_layers = 1
        self.input_size = 4

    def setUp(self):
        self.op_type = "cudnn_gru"
        self.config()
        self.dtype = np.float32

        num_directions = 2 if self.bidirec else 1
        gate_size = 3
        layer_input_size = 0
        weight_size = 0

        for i in range(self.num_layers):
            for direc in range(num_directions):
                layer_input_size = self.input_size if i == 0 else self.hidden_size * num_directions

                wi_size = gate_size * layer_input_size * self.hidden_size
                wh_size = gate_size * self.hidden_size * self.hidden_size
                bi_size = gate_size * self.hidden_size
                bh_size = gate_size * self.hidden_size
                weight_size += (wi_size + wh_size + bi_size + bh_size)

        rs = np.random.RandomState(123)
        input = rs.uniform(
            low=-0.1,
            high=0.1,
            size=(self.num_steps, self.batch_size,
                  self.hidden_size)).astype(self.dtype)

        flat_w = rs.uniform(
            low=-0.1, high=0.1, size=(weight_size)).astype(self.dtype)

        init_h = np.zeros(
            (num_directions * self.num_layers, self.batch_size,
             self.hidden_size),
            dtype=np.float32)
        output, last_hidden = gru_reference(input, flat_w, num_directions,
                                            self.num_layers)

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
            'max_len': self.num_steps,
            'dropout_prob': 0.0,
            'is_bidirec': self.bidirec,
            'input_size': self.hidden_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
        }
        self.outputs = {'Out': output, 'last_h': last_hidden}

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


class TestCUDNNGruOp2(TestCUDNNGruOp):
    def config(self):
        self.num_steps = 2
        self.batch_size = 3
        self.hidden_size = 4
        self.bidirec = False
        self.num_layers = 1
        self.input_size = 4


if __name__ == '__main__':
    unittest.main()
