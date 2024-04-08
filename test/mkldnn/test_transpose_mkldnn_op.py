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

import unittest

import numpy as np
from op_test import OpTest


class TestTransposeMKLDNN(OpTest):
    def setUp(self):
        self.init_op_type()
        self.initTestCase()
        self.inputs = {'X': np.random.random(self.shape).astype("float32")}
        self.attrs = {
            'axis': list(self.axis),
            'use_mkldnn': self.use_mkldnn,
        }
        self.outputs = {
            'XShape': np.random.random(self.shape).astype("float32"),
            'Out': self.inputs['X'].transpose(self.axis),
        }

    def init_op_type(self):
        self.op_type = "transpose2"
        self.use_mkldnn = True

    def test_check_output(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_output(
            no_check_set=['XShape'], check_dygraph=False, check_pir_onednn=True
        )

    def test_check_grad(self):
        # TODO(wangzhongpu): support onednn op in dygraph mode
        self.check_grad(
            ['X'], 'Out', check_dygraph=False, check_pir_onednn=True
        )

    def initTestCase(self):
        self.shape = (30, 4)
        self.axis = (1, 0)


class TestCase0MKLDNN(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = (100,)
        self.axis = (0,)


class TestCase1a(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = (3, 4, 10)
        self.axis = (0, 2, 1)


class TestCase1b(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = (3, 4, 10)
        self.axis = (2, 1, 0)


class TestCase2(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 2, 3, 1)


class TestCase3(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.axis = (4, 2, 3, 1, 0)


class TestCase4(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6, 1)
        self.axis = (4, 2, 3, 1, 0, 5)


class TestCase_ZeroDim(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = ()
        self.axis = ()


if __name__ == '__main__':
    unittest.main()
