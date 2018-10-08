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


class TestClipByNormOpWithSelectedRows(OpTest):
    def setUp(self):
        self.initTestCase()

        self.max_relative_error = 0.006

        scope = core.Scope()
        x_selected_rows = scope.var('X').get_selected_rows()
        x_selected_rows.set_rows([1, 1, 2, 0])
        x_tensor = x_selected_rows.get_tensor()
        x_tensor = np.random.random((4, 1)).astype("float32")
        x_tensor[np.abs(x_tensor) < self.max_relative_error] = 0.5

        self.op_type = "clip_by_norm"
        self.inputs = {'X': x_selected_rows, }
        self.attrs = {}
        self.attrs['max_norm'] = self.max_norm
        y_tensor = np.zeros((3, 1))
        y_tensor[0::1] = np.sum(x_tensor[0::1], x_tensor[1::1])
        y_tensor[1::1] = x_tensor[2::1]
        y_tensor[2::1] = x_tensor[3::1]
        norm = np.sqrt(np.sum(np.square(y_tensor)))
        if norm > self.max_norm:
            output = self.max_norm * y_tensor / norm
        else:
            output = y_tensor
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output()

    def initTestCase(self):
        self.shape = (100, )
        self.max_norm = 1.0


if __name__ == '__main__':
    unittest.main()
