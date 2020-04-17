# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest, os
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci


@skip_check_grad_ci(reason="Tests inference only optimization.")
class TestMatMulOpSpecial(OpTest):
    def generate_data(self):
        self.x = np.random.random([1, 128, 128]).astype("float32")
        self.y = np.random.random([1, 128, 64]).astype("float32")
        self.out = np.matmul(self.x, self.y)
        self.reshape_out = []
        self.transpose_out = []

    def set_attributes(self):
        self.attrs = {
            'reshape_Out': self.reshape_out,
            'transpose_Out': self.transpose_out
        }

    def setUp(self):
        os.environ["DNNL_MAX_CPU_ISA"] = "AVX"
        self.op_type = "matmul"
        self._cpu_only = True
        self.use_mkldnn = True
        self.generate_data()
        self.set_attributes()

        self.inputs = {'X': self.x, 'Y': self.y}
        self.attrs['use_mkldnn'] = self.use_mkldnn

        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output()


class TestMatMulOpSpecialSimplest(TestMatMulOpSpecial):
    def generate_data(self):
        bs = 8
        self.x = np.random.random([bs, 12, 128, 128]).astype("float32")
        self.y = np.random.random([bs, 12, 128, 64]).astype("float32")
        self.transpose_out = [0, 2, 1, 3]
        self.reshape_out = [0, 0, 768]
        self.out = np.matmul(self.x, self.y).transpose([0, 2, 1, 3]).reshape(
            [bs, -1, 768])


class TestMatMulOpSpecialSimplestOtherDims(TestMatMulOpSpecial):
    def generate_data(self):
        bs = 3
        self.x = np.random.random([bs, 12, 128, 128]).astype("float32")
        self.y = np.random.random([bs, 12, 128, 128]).astype("float32")
        self.transpose_out = [0, 2, 1, 3]
        self.reshape_out = [0, 0, 12 * 128]
        self.out = np.matmul(self.x, self.y).transpose([0, 2, 1, 3]).reshape(
            [bs, -1, 12 * 128])


if __name__ == '__main__':
    unittest.main()
