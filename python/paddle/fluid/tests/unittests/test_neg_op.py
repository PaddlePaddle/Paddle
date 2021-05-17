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

import unittest
import numpy as np
import paddle
from op_test import OpTest

paddle.enable_static()


class TestNegOp(OpTest):
    def setUp(self):
        self.op_type = 'neg'
        self.dtype = np.float64
        self.inputs = {'X': np.random.random((32, 64)).astype(self.dtype)}
        self.outputs = {'Out': np.negative(self.inputs['X'])}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestNegOpFp32(TestNegOp):
    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()


class TestNegOpFp16(TestNegOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output()


class TestNegOpInt64(TestNegOp):
    def init_dtype(self):
        self.dtype = np.Int64

    def test_check_output(self):
        self.check_output()


class TestNegOpInt32(TestNegOp):
    def init_dtype(self):
        self.dtype = np.int32

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
