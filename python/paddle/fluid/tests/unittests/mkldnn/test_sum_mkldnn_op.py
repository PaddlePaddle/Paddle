# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid.tests.unittests.test_sum_op import TestSumOp
import numpy as np


class TestMKLDNN(TestSumOp):
    def setUp(self):
        self.op_type = "sum"
        self.init_kernel_type()
        self.use_mkldnn = True
        x0 = np.random.random((25, 4)).astype(self.dtype)
        x1 = np.random.random((25, 4)).astype(self.dtype)
        x2 = np.random.random((25, 4)).astype(self.dtype)
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def init_kernel_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad(['x0'], 'Out', check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
