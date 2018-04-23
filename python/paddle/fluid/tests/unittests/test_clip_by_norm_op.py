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
from op_test import OpTest


class TestClipByNormOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.initTestCase()
        input = np.random.random(self.shape).astype("float32")
        input[np.abs(input) < self.max_relative_error] = 0.5
        self.op_type = "clip_by_norm"
        self.inputs = {'X': input, }
        self.attrs = {}
        self.attrs['max_norm'] = self.max_norm
        norm = np.sqrt(np.sum(np.square(input)))
        if norm > self.max_norm:
            output = self.max_norm * input / norm
        else:
            output = input
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def initTestCase(self):
        self.shape = (100, )
        self.max_norm = 1.0


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


if __name__ == '__main__':
    unittest.main()
