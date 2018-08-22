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
from test_fusion_lstm_op import fc, ACTIVATION


def attention_lstm(
        x,  # T x M
        lod,  # 1 x N
        h0,  # N x D
        c0,  # N x D
        fcws,  # (M+D) x 1, 1x1
        fcbs,  # 1 x 1, 1x1
        w,  # (M+D) x 4D
        b,  # 1 x 4D
        act_gate,
        act_cell,
        act_cand):
    hidden
    cell
    return hidden, cell


class TestAttentionLSTMOp(OpTest):
    def set_conf(self):
        self.lod = [[3]]

    def setUp(self):
        self.op_type = 'attention_lstm'
        self.lod = [[3]]
        self.M = 30
        self.D = 15
        self.has_initial_hidden = True
        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'
        self.set_conf()

        T = sum(self.lod[0])
        bs = len(self.lod[0])

        x = np.random.normal(size=(T, self.M)).astype('float32')
        c0 = np.random.normal(size=(bs, self.D)).astype('float32')
        if self.has_initial_hidden:
            h0 = np.random.normal(size=(bs, self.D)).astype('float32')
        else:
            h0 = np.zeros((bs, self.D)).astype('float32')

        fcw1 = np.random.normal(size=(self.M + self.D, 1)).astype('float32')
        fcb1 = np.random.normal(size=(1, 1)).astype('float32')
        fcw2 = np.random.normal(size=(1, 1)).astype('float32')
        fcb2 = np.random.normal(size=(1, 1)).astype('float32')

        # lstm weight and bias
        w = np.random.normal(size=(self.M + self.D,
                                   self.D * 4)).astype('float32')
        b = np.random.normal(size=(1, self.D * 4)).astype('float32')

        h, c = attention_lstm(x, self.lod, h0, c0, [fcw1, fcw2], [fcb1, fcb2],
                              ACTIVATION[self.act_gate],
                              ACTIVATION[self.act_cell],
                              ACTIVATION[self.act_cand])

        self.inputs = {
            'X': (x, self.lod),
            'C0': c0,
            'AttentionWeight': fcw1,
            'AttentionBias': fcb1,
            'AttentionScalar': fcw2,
            'AttentionScalarBias': fcb2,
            'LSTMWeight': w,
            'LSTMBias': b
        }

        if self.has_initial_hidden:
            self.inputs['H0'] = h0

        self.outputs = {
            'Hidden': (h, self.lod),
            'Cell': (c, self.lod),
        }
        self.attrs = {
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand
        }

    def test_check_output(self):
        self.check_output()


class TestAttentionOpNonInit(TestAttentionLSTMOp):
    def set_conf(self):
        self.has_initial_hidden = False


class TestAttentionOpMD1(TestAttentionLSTMOp):
    def set_conf(self):
        self.M = 36
        self.D = 8


class TestAttentionOpMD2(TestAttentionLSTMOp):
    def set_conf(self):
        self.M = 8
        self.D = 8


class TestAttentionOpMD3(TestAttentionLSTMOp):
    def set_conf(self):
        self.M = 15
        self.D = 30


class TestAttentionOpBS1(TestAttentionLSTMOp):
    def set_conf(self):
        self.lod = [[5]]
        self.M = 16
        self.D = 32


class TestAttentionOpBS2(TestAttentionLSTMOp):
    def set_conf(self):
        self.lod = [[3, 6]]


class TestAttentionOpBS5(TestAttentionLSTMOp):
    def set_conf(self):
        self.lod = [[3, 2, 4, 7, 5]]


if __name__ == '__main__':
    unittest.main()
