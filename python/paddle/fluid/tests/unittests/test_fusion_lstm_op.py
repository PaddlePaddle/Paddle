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

import unittest
import numpy as np
from op_test import OpTest
from test_lstm_op import lstm, ACTIVATION


def fc(x, w, b):
    return np.dot(x, w) + b


def fusion_lstm(
        x,  # T x M
        lod,  # 1 x N
        wx=None,  # M x 4D
        bx=None,  # 1 x 4D
        h0=None,  # N x D
        c0=None,  # N x D
        w_h=None,  # D x 4D
        w_b=None,  # 1 x 4D
        w_c=None,  # 1 x 3D
        is_reverse=False,
        act_gate=None,
        act_cell=None,
        act_cand=None):
    return lstm(
        fc(x, wx, bx), lod, h0, c0, w_h, w_b, w_c, is_reverse, act_gate,
        act_cell, act_cand)


class TestLstmOp(OpTest):
    def set_argument(self):
        self.lod = [[2, 3, 2]]

    def setUp(self):
        self.op_type = 'fusion_lstm'
        self.lod = [[2, 3, 2]]
        self.M = 8
        self.D = 16
        self.has_initial_state = False
        self.is_reverse = False
        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'
        self.use_peepholes = False
        self.set_argument()

        T = sum(self.lod[0])
        bs = len(self.lod[0])

        x = np.random.normal(size=(T, self.M)).astype('float32')
        if self.has_initial_state:
            h0 = np.random.normal(size=(bs, self.D)).astype('float32')
            c0 = np.random.normal(size=(bs, self.D)).astype('float32')
        else:
            h0 = np.zeros((bs, self.D)).astype('float32')
            c0 = np.zeros((bs, self.D)).astype('float32')

        wh = np.random.normal(size=(self.D, 4 * self.D)).astype('float32')

        if self.use_peepholes:
            b = np.random.normal(size=(1, 7 * self.D)).astype('float32')
        else:
            b = np.random.normal(size=(1, 4 * self.D)).astype('float32')
        w_b = np.copy(b[:, 0:4 * self.D])
        w_c = b[:, 4 * self.D:] if self.use_peepholes else None

        # this is the weight of fc
        wx = np.random.normal(size=(self.M, 4 * self.D)).astype('float32')
        # this is the bias of fc
        # and it should be manually added into the bias of this fusion LSTM
        bx = np.random.normal(size=(1, 4 * self.D)).astype('float32')
        b[0, 0:4 * self.D] += bx[0, :]
        h, c = fusion_lstm(x, self.lod, wx, bx, h0, c0, wh, w_b, w_c,
                           self.is_reverse, ACTIVATION[self.act_gate],
                           ACTIVATION[self.act_cell], ACTIVATION[self.act_cand])

        self.inputs = {
            'X': (x, self.lod),
            'WeightX': wx,
            'WeightH': wh,
            'Bias': b
        }

        if self.has_initial_state:
            self.inputs['H0'] = h0
            self.inputs['C0'] = c0

        self.outputs = {
            'Hidden': (h, self.lod),
            'Cell': (c, self.lod),
        }
        self.attrs = {
            'use_peepholes': self.use_peepholes,
            'is_reverse': self.is_reverse,
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand
        }

    def test_check_output(self):
        self.check_output(atol=1e-8)


class TestLstmOpInitReverse(TestLstmOp):
    def set_argument(self):
        self.has_initial_state = True
        self.is_reverse = True


class TestLstmOpMD1(TestLstmOp):
    def set_argument(self):
        self.M = 35
        self.D = 8


class TestLstmOpMD2(TestLstmOp):
    def set_argument(self):
        self.M = 36
        self.D = 8


class TestLstmOpBS1(TestLstmOp):
    def set_argument(self):
        self.lod = [[3]]
        self.D = 16


if __name__ == '__main__':
    unittest.main()
