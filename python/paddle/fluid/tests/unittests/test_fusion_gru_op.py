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
import math
from op_test import OpTest
from paddle.fluid.tests.unittests.test_gru_op import gru
from paddle.fluid.tests.unittests.test_fusion_lstm_op import fc, ACTIVATION


def fusion_gru(
        x,  # T x M
        lod,  # 1 x N
        h0,  # N x D
        wx,  # M x 3D
        wh,  # D x 3D
        bias,  # 1 x 3D
        is_reverse,
        origin_mode,
        act_state,
        act_gate):
    return gru(fc(x, wx, bias),
               lod,
               h0,
               wh,
               np.zeros((1, wh.shape[1]), dtype='float32'),
               is_reverse,
               act_state,
               act_gate,
               origin_mode=origin_mode)


class TestFusionGRUOp(OpTest):

    def set_confs(self):
        pass

    def setUp(self):
        self.op_type = "fusion_gru"
        self.lod = [[2, 4, 3]]
        self.M = 3
        self.D = 5
        self.is_reverse = False
        self.with_h0 = True
        self.with_bias = True
        self.act_state = 'tanh'
        self.act_gate = 'sigmoid'
        self.origin_mode = False
        self.use_mkldnn = False
        self.set_confs()

        T = sum(self.lod[0])
        N = len(self.lod[0])

        x = np.random.rand(T, self.M).astype('float32')
        wx = np.random.rand(self.M, 3 * self.D).astype('float32')
        wh = np.random.rand(self.D, 3 * self.D).astype('float32')
        bias = np.random.rand(
            1, 3 * self.D).astype('float32') if self.with_bias else np.zeros(
                (1, 3 * self.D), dtype='float32')
        h0 = np.random.rand(
            N, self.D).astype('float32') if self.with_h0 else np.zeros(
                (N, self.D), dtype='float32')

        _, _, _, hidden = fusion_gru(x, self.lod, h0, wx, wh, bias,
                                     self.is_reverse, self.origin_mode,
                                     ACTIVATION[self.act_state],
                                     ACTIVATION[self.act_gate])

        self.inputs = {'X': (x, self.lod), 'WeightX': wx, 'WeightH': wh}

        if self.with_bias:
            self.inputs['Bias'] = bias

        if self.with_h0:
            self.inputs['H0'] = h0

        self.outputs = {'Hidden': (hidden, self.lod)}

        self.attrs = {
            'activation': self.act_state,
            'gate_activation': self.act_gate,
            'is_reverse': self.is_reverse,
            'origin_mode': self.origin_mode,
            'use_mkldnn': self.use_mkldnn
        }

    def test_check_output(self):
        for use_seq in {True, False}:
            self.attrs['use_seq'] = use_seq
            self.check_output(check_dygraph=False)


class TestFusionGRUOpNoInitial(TestFusionGRUOp):

    def set_confs(self):
        self.with_h0 = False


class TestFusionGRUOpNoBias(TestFusionGRUOp):

    def set_confs(self):
        self.with_bias = False


class TestFusionGRUOpReverse(TestFusionGRUOp):

    def set_confs(self):
        self.is_reverse = True


class TestFusionGRUOpMD1(TestFusionGRUOp):

    def set_confs(self):
        self.M = 36
        self.D = 8


class TestFusionGRUOpMD2(TestFusionGRUOp):

    def set_confs(self):
        self.M = 8
        self.D = 8


class TestFusionGRUOpMD3(TestFusionGRUOp):

    def set_confs(self):
        self.M = 17
        self.D = 15


class TestFusionGRUOpBS1(TestFusionGRUOp):

    def set_confs(self):
        self.lod = [[3]]
        self.D = 16


if __name__ == "__main__":
    from paddle import enable_static
    enable_static()
    unittest.main()
