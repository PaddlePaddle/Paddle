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

from paddle.fluid.tests.unittests.test_transpose_op import TestTransposeOp


class TestTransposeMKLDNN(TestTransposeOp):
    def init_op_type(self):
        self.op_type = "transpose2"
        self.use_mkldnn = True
        return


class TestCase0MKLDNN(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = (3, )
        self.axis = (0, )


class TestCase1a(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = (3, 4, 5)
        self.axis = (0, 2, 1)


class TestCase1b(TestTransposeMKLDNN):
    def initTestCase(self):
        self.shape = (3, 4, 5)
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


if __name__ == '__main__':
    unittest.main()
