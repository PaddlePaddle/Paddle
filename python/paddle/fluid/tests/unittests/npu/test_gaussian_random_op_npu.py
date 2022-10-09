#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

sys.path.append("..")
import paddle
import paddle.fluid as fluid
from op_test import OpTest
from test_gaussian_random_op import TestGaussianRandomOp

paddle.enable_static()


class TestNPUGaussianRandomOp(OpTest):

    def setUp(self):
        self.set_npu()
        self.op_type = "gaussian_random"
        self.init_dtype()
        self.set_attrs()
        self.inputs = {}
        self.use_mkldnn = False
        self.attrs = {
            "shape": [123, 92],
            "mean": self.mean,
            "std": self.std,
            "seed": 10,
            "use_mkldnn": self.use_mkldnn
        }
        paddle.seed(10)

        self.outputs = {'Out': np.zeros((123, 92), dtype='float32')}

    def set_attrs(self):
        self.mean = 1.0
        self.std = 2.

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_customized(self.verify_output, self.place)

    def verify_output(self, outs):
        self.assertEqual(outs[0].shape, (123, 92))
        hist, _ = np.histogram(outs[0], range=(-3, 5))
        hist = hist.astype("float32")
        hist /= float(outs[0].size)
        data = np.random.normal(size=(123, 92), loc=1, scale=2)
        hist2, _ = np.histogram(data, range=(-3, 5))
        hist2 = hist2.astype("float32")
        hist2 /= float(outs[0].size)
        np.testing.assert_allclose(hist,
                                   hist2,
                                   rtol=0,
                                   atol=0.01,
                                   err_msg="hist: " + str(hist) + " hist2: " +
                                   str(hist2))


if __name__ == "__main__":
    unittest.main()
