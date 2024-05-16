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

import sys
import unittest

import numpy as np
from op_test import OpTest

sys.path.append("../deprecated/legacy_test")
from test_fc_op import MatrixGenerate, fc_refer


class TestFusionRepeatedFCReluOp(OpTest):
    def setUp(self):
        self.bs = 3
        self.ic = 9
        self.oc = [2, 4, 3]
        assert len(self.oc) > 1, 'Should larger than 1'
        self.set_conf()
        self.op_type = 'fusion_repeated_fc_relu'
        sz = len(self.oc)
        ics = [self.ic] + self.oc[0 : sz - 1]
        assert len(ics) == len(self.oc)
        weights = []
        biases = []
        outs = []

        i = 0
        matrix = MatrixGenerate(self.bs, ics[i], self.oc[i], 1, 1)
        inp = np.reshape(matrix.input, [self.bs, ics[i]])
        weights.append(
            (
                f'W_{i}',
                np.reshape(matrix.weights, [ics[i], self.oc[i]]),
            )
        )
        biases.append((f'B_{i}', matrix.bias))
        outs.append(
            np.reshape(
                np.maximum(fc_refer(matrix, True), 0), [self.bs, self.oc[i]]
            )
        )

        for i in range(sz - 1):
            matrix = MatrixGenerate(self.bs, ics[i + 1], self.oc[i + 1], 1, 1)
            matrix.input = np.reshape(outs[i], [self.bs, ics[i + 1], 1, 1])
            out = fc_refer(matrix, True)
            weights.append(
                (
                    f'W_{i + 1}',
                    np.reshape(matrix.weights, [ics[i + 1], self.oc[i + 1]]),
                )
            )
            biases.append((f'B_{i + 1}', matrix.bias))
            outs.append(
                np.reshape(np.maximum(out, 0), [self.bs, self.oc[i + 1]])
            )

        relu_outs = []
        for i in range(sz - 1):
            relu_outs.append((f'ReluOut_{i}', outs[i]))

        self.inputs = {
            'X': inp,
            'W': weights,
            'Bias': biases,
        }

        self.outputs = {'Out': outs[-1], 'ReluOut': relu_outs}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def set_conf(self):
        pass


class TestFusionRepeatedFCReluOpBS1(TestFusionRepeatedFCReluOp):
    def set_conf(self):
        self.bs = 1
        self.oc = [4, 2, 7, 5, 512, 1024]


if __name__ == '__main__':
    unittest.main()
