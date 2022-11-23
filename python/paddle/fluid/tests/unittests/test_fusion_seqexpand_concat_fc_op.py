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
from test_fusion_lstm_op import fc, ACTIVATION


def fusion_seqexpand_concat_fc(xs, lod, w, b, fc_act):

    T = sum(lod[0])
    N = len(lod[0])
    num_inputs = len(xs)
    D = w.shape[1]

    expanded_inputs = [xs[0]]
    for i in range(num_inputs - 1):
        x = xs[i + 1]
        assert x.shape[0] == N
        expanded = np.repeat(x, lod[0], axis=0)
        assert expanded.shape[0] == T
        assert expanded.shape[1] == x.shape[1]
        expanded_inputs.append(expanded)

    fc_input = np.concatenate(expanded_inputs, axis=1)
    assert fc_input.shape[0] == T
    assert fc_input.shape[1] == w.shape[0]
    fc_out = fc(fc_input, w, b)
    fc_out = fc_act(fc_out)
    assert fc_out.shape[0] == T
    assert fc_out.shape[1] == D
    return fc_out


class TestFusionSeqExpandConcatFCOp(OpTest):

    def set_conf(self):
        pass

    def setUp(self):
        self.op_type = 'fusion_seqexpand_concat_fc'
        self.lod = [[3, 5, 8, 2]]
        self.inputs_M = [15, 10, 10]
        self.D = 20
        self.with_bias = True
        self.fc_act = 'relu'
        self.set_conf()

        T = sum(self.lod[0])
        bs = len(self.lod[0])
        num_inputs = len(self.inputs_M)

        x0 = np.random.normal(size=(T, self.inputs_M[0])).astype('float32')
        xs = [x0]
        for i in range(num_inputs - 1):
            xi = np.random.normal(size=(bs,
                                        self.inputs_M[i + 1])).astype('float32')
            xs.append(xi)

        # fc weight and bias
        w = np.random.normal(size=(sum(self.inputs_M),
                                   self.D)).astype('float32')
        b = np.random.normal(
            size=(1, self.D)).astype('float32') if self.with_bias else np.zeros(
                (1, self.D)).astype('float32')

        out = fusion_seqexpand_concat_fc(xs, self.lod, w, b,
                                         ACTIVATION[self.fc_act])

        self.inputs = {'X': [('x0', (x0, self.lod))], 'FCWeight': w}
        normal_lod = [[1] * bs]
        for i in range(num_inputs - 1):
            self.inputs['X'].append(('x%d' % (i + 1), (xs[i + 1], normal_lod)))

        if self.with_bias:
            self.inputs['FCBias'] = b

        self.outputs = {'Out': (out, self.lod)}
        self.attrs = {'fc_activation': self.fc_act}

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestFusionSECFCOpNonBias(TestFusionSeqExpandConcatFCOp):

    def set_conf(self):
        self.with_bias = False


class TestFusionSECFCOpNonAct(TestFusionSeqExpandConcatFCOp):

    def set_conf(self):
        self.fc_act = 'identity'


class TestFusionSECFCOpMD1(TestFusionSeqExpandConcatFCOp):

    def set_conf(self):
        self.inputs_M = [3, 4, 2, 1, 5]
        self.D = 8


class TestFusionSECFCOpMD2(TestFusionSeqExpandConcatFCOp):

    def set_conf(self):
        self.lod = [[5, 6]]
        self.inputs_M = [1, 1]


class TestFusionSECFCOpBS1_1(TestFusionSeqExpandConcatFCOp):

    def set_conf(self):
        self.lod = [[1]]
        self.inputs_M = [3, 4, 2]


class TestFusionSECFCOpBS1_2(TestFusionSeqExpandConcatFCOp):

    def set_conf(self):
        self.lod = [[1]]
        self.inputs_M = [3, 4]


class TestFusionSECFCOpBS1_3(TestFusionSeqExpandConcatFCOp):

    def set_conf(self):
        self.lod = [[5]]
        self.inputs_M = [6, 3]


if __name__ == '__main__':
    unittest.main()
