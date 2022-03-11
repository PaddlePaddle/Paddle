#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.executor import Executor
import sys
sys.path.append('..')
from op_test import OpTest
import paddle

paddle.enable_static()


class TestGaussianRandomOp(OpTest):
    def setUp(self):
        self.op_type = "gaussian_random"
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.set_attrs()
        self.inputs = {}
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
        }
        paddle.seed(10)

        self.outputs = {'Out': np.zeros((123, 92), dtype='float32')}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.

    def test_check_output(self):
        self.check_output_with_place_customized(self.verify_output, self.place)
        # self.check_output_customized(self.verify_output, self.place)

    def verify_output(self, outs):
        self.assertEqual(outs[0].shape, (123, 92))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float32")
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float32")
        hist2 /= float(outs[0].size)
        self.assertTrue(
            np.allclose(
                hist, hist2, rtol=0, atol=0.01),
            "hist: " + str(hist) + " hist2: " + str(hist2))


class TestMeanStdAreInt(TestGaussianRandomOp):
    def set_attrs(self):
        self.mean = 1
        self.std = 2


if __name__ == "__main__":
    unittest.main()
