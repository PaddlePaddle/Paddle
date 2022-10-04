#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

sys.path.append("..")
import unittest
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test_xpu import OpTest, XPUOpTest
import paddle
from paddle.fluid import Program, program_guard


class TestXPUClipByNormOp(XPUOpTest):

    def setUp(self):
        self.op_type = "clip_by_norm"
        self.dtype = np.float32
        self.use_xpu = True
        self.max_relative_error = 0.006
        self.initTestCase()
        input = np.random.random(self.shape).astype("float32")
        input[np.abs(input) < self.max_relative_error] = 0.5
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

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def initTestCase(self):
        self.shape = (100, )
        self.max_norm = 1.0


class TestCase1(TestXPUClipByNormOp):

    def initTestCase(self):
        self.shape = (100, )
        self.max_norm = 1e20


class TestCase2(TestXPUClipByNormOp):

    def initTestCase(self):
        self.shape = (16, 16)
        self.max_norm = 0.1


class TestCase3(TestXPUClipByNormOp):

    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max_norm = 1.0


if __name__ == "__main__":
    unittest.main()
