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
from op_test import OpTest


def row_conv_forward(x, lod, wt):
    out = np.zeros_like(x)
    num_sequences = len(lod[0])
    seq_info = [0]
    for seq_len in lod[0]:
        seq_info.append(seq_info[-1] + seq_len)
    context_length = wt.shape[0]

    for i in range(num_sequences):  # loop over number of sequences
        start = seq_info[i]
        end = seq_info[i + 1]
        curinput = x[start:end, :]
        curoutput = out[start:end, :]

        cur_timesteps = end - start
        for j in range(cur_timesteps):  # loop over different timesteps
            for k in range(context_length):
                if j + k >= cur_timesteps:
                    continue
                curoutput[j, :] += curinput[j + k, :] * wt[k, :]

    return out


class TestRowConvOp1(OpTest):
    def setUp(self):

        self.op_type = "row_conv"
        lod = [[2, 3, 2]]
        T = sum(lod[0])
        D = 16
        context_length = 2

        x = np.random.random((T, D)).astype("float32")
        wt = np.random.random((context_length, D)).astype("float32")
        self.inputs = {'X': (x, lod), 'Filter': wt}

        out = row_conv_forward(x, lod, wt)
        self.outputs = {'Out': (out, lod)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Filter'], 'Out', max_relative_error=0.05)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Filter'], 'Out', max_relative_error=0.05, no_grad_set=set('X'))

    def test_check_grad_ignore_wt(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Filter'))


class TestRowConvOp2(OpTest):
    def setUp(self):

        self.op_type = "row_conv"
        lod = [[20, 30, 50]]
        T = sum(lod[0])
        D = 35
        context_length = 35

        x = np.random.random((T, D)).astype("float32")
        wt = np.random.random((context_length, D)).astype("float32")
        self.inputs = {'X': (x, lod), 'Filter': wt}

        out = row_conv_forward(x, lod, wt)
        self.outputs = {'Out': (out, lod)}

    def test_check_output(self):
        self.check_output()

    #max_relative_error is increased from 0.05 to 0.06 as for higher
    #dimensional input, the dX on CPU for some values has max_rel_error
    #slightly more than 0.05
    def test_check_grad_normal(self):
        self.check_grad(['X', 'Filter'], 'Out', max_relative_error=0.06)

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Filter'], 'Out', max_relative_error=0.06, no_grad_set=set('X'))

    def test_check_grad_ignore_wt(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.06, no_grad_set=set('Filter'))


def row_conv_foward_Tensor(x, wt):
    out = np.zeros_like(x)
    num_sequence = x.shape[0]
    timesteps = x.shape[1]
    context_length = wt.shape[0]
    for i in range(num_sequence):
        cur_in = x[i:i + 1, :][0]
        cur_out = out[i:i + 1, :][0]
        for j in range(timesteps):
            for k in range(context_length):
                if j + k >= timesteps:
                    continue
                cur_out[j, :] += cur_in[j + k, :] * wt[k, :]
    return out


class TestRowOpWithTensorInput(OpTest):
    def setUp(self):
        self.op_type = "row_conv"
        length = [3, 2, 4]
        B = 2
        T = sum(length)
        D = 16
        context_length = 2

        x = np.random.random((B, T, D)).astype("float32")
        wt = np.random.random((context_length, D)).astype("float32")
        self.inputs = {'X': x, 'Filter': wt}

        out = row_conv_foward_Tensor(x, wt)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Filter'], 'Out', max_relative_error=0.05, no_grad_set=set('X'))

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Filter'], 'Out', max_relative_error=0.05)

    def test_check_grad_ignore_wt(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Filter'))


if __name__ == '__main__':
    unittest.main()
