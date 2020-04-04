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
from op_test import OpTest
import paddle.fluid.core as core


class TestPadOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.dtype = self.get_dtype()
        self.op_type = "pad"
        self.inputs = {'X': np.random.random(self.shape).astype(self.dtype), }
        self.attrs = {}
        self.attrs['paddings'] = np.array(self.paddings).flatten()
        self.attrs['pad_value'] = self.pad_value
        self.outputs = {
            'Out': np.pad(self.inputs['X'],
                          self.paddings,
                          mode='constant',
                          constant_values=self.pad_value)
        }

    def get_dtype(self):
        return np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')

    def initTestCase(self):
        self.shape = (16, 16)
        self.paddings = [(0, 1), (2, 3)]
        self.pad_value = 0.0


class TestCase1(TestPadOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5)
        self.paddings = [(0, 1), (2, 3), (2, 1), (1, 1)]
        self.pad_value = 0.5


class TestCase2(TestPadOp):
    def initTestCase(self):
        self.shape = (5, 5, 5)
        self.paddings = [(0, 0), (0, 0), (1, 2)]
        self.pad_value = 1.0


class TestCase3(TestPadOp):
    def initTestCase(self):
        self.shape = (100)
        self.paddings = [(0, 1)]
        self.pad_value = 0.9


#----------------Pad Fp16----------------


def create_test_fp16(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestPadFp16(parent):
        def get_dtype(self):
            return np.float16

        def test_check_grad_normal(self):
            self.check_grad(['X'], 'Out', max_relative_error=0.3)

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestPadFp16.__name__ = cls_name
    globals()[cls_name] = TestPadFp16


create_test_fp16(TestPadOp)
create_test_fp16(TestCase1)
create_test_fp16(TestCase2)
create_test_fp16(TestCase3)

if __name__ == '__main__':
    unittest.main()
