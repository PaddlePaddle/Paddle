# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import sys

sys.path.append("..")
from op_test import OpTest

paddle.enable_static()


class TestClipByNormOp(OpTest):

    def setUp(self):
        self.set_npu()
        self.max_relative_error = 0.006
        self.init_dtype()
        self.initTestCase()
        input = np.random.random(self.shape).astype(self.dtype)
        input[np.abs(input) < self.max_relative_error] = 0.5
        self.op_type = "clip_by_norm"
        self.inputs = {
            'X': input,
        }
        self.attrs = {}
        self.attrs['max_norm'] = self.max_norm
        norm = np.sqrt(np.sum(np.square(input)))
        if norm > self.max_norm:
            output = self.max_norm * input / norm
        else:
            output = input
        self.outputs = {'Out': output}

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def initTestCase(self):
        self.shape = (100, )
        self.max_norm = 1.0

    def init_dtype(self):
        self.dtype = np.float32


class TestCase1(TestClipByNormOp):

    def initTestCase(self):
        self.shape = (100, )
        self.max_norm = 1e20


class TestCase2(TestClipByNormOp):

    def initTestCase(self):
        self.shape = (16, 16)
        self.max_norm = 0.1


class TestCase3(TestClipByNormOp):

    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max_norm = 1.0


class TestClipByNormOpFp16(TestClipByNormOp):

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)


class TestClipByNormOpFp16Case1(TestClipByNormOpFp16):

    def initTestCase(self):
        self.shape = (100, )
        self.max_norm = 1e20


class TestClipByNormOpFp16Case2(TestClipByNormOpFp16):

    def initTestCase(self):
        self.shape = (16, 16)
        self.max_norm = 0.1


class TestClipByNormOpFp16Case3(TestClipByNormOpFp16):

    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max_norm = 1.0


if __name__ == '__main__':
    unittest.main()
