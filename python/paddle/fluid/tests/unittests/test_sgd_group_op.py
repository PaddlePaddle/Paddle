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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from op_test import OpTest


class TestSGDOp(OpTest):
    def setUp(self):
        self.op_type = "sgd_group"
        w0 = np.random.random((1, 124)).astype('float32')
        w1 = np.random.random((3, 24)).astype('float32')
        w2 = np.random.random((4, 104)).astype('float32')

        g0 = np.random.random((1, 124)).astype('float32')
        g1 = np.random.random((3, 24)).astype('float32')
        g2 = np.random.random((4, 104)).astype('float32')

        lr0 = np.array([0.1]).astype("float32")
        lr1 = np.array([0.2]).astype("float32")
        lr2 = np.array([0.3]).astype("float32")

        o0 = w0 - lr0 * g0
        o1 = w1 - lr1 * g1
        o2 = w2 - lr2 * g2

        self.inputs = {
            "Params": [("w0", w0), ("w1", w1), ("w2", w2)],
            "Grads": [("g0", g0), ("g1", g1), ("g2", g2)],
            'LearningRates': [("lr0", lr0), ("lr1", lr1), ("lr2", lr2)]
        }

        self.outputs = {'ParamOuts': [("o0", o0), ("o1", o1), ("o2", o2)]}

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
