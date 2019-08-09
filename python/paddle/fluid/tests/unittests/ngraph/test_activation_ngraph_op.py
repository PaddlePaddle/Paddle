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
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_activation_op import TestSigmoid, TestRelu, TestTanh


class TestNGRAPHReluDim4(TestRelu):
    def setUp(self):
        super(TestNGRAPHReluDim4, self).setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}


class TestNGRAPHTanhDim4(TestTanh):
    def setUp(self):
        super(TestNGRAPHTanhDim4, self).setUp()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype("float32")
        }
        self.outputs = {'Out': np.tanh(self.inputs['X'])}


if __name__ == '__main__':
    unittest.main()
