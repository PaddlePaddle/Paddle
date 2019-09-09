# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from op_test import OpTest
import numpy as np
import unittest


def strided_slice_native_forward(input, begin, end, stride):
    dim = input.ndim
    result = {
        1: lambda input, begin, end, stride: input[begin[0]:end[0]:stride[0]],
        2: lambda input, begin, end, stride: input[begin[0]:end[0]:stride[0], \
                begin[1]:end[1]:stride[1]],
        3: lambda input, begin, end, stride: input[begin[0]:end[0]:stride[0], \
                begin[1]:end[1]:stride[1], begin[2]:end[2]:stride[2]],
        4: lambda input, begin, end, stride: input[begin[0]:end[0]:stride[0], \
                begin[1]:end[1]:stride[1], begin[2]:end[2]:stride[2], begin[3]:end[3]:stride[3]],
        5: lambda input, begin, end, stride: input[begin[0]:end[0]:stride[0], \
                begin[1]:end[1]:stride[1], begin[2]:end[2]:stride[2], begin[3]:end[3]:stride[3], begin[4]:end[4]:stride[4]],
        6: lambda input, begin, end, stride: input[begin[0]:end[0]:stride[0], \
                begin[1]:end[1]:stride[1], begin[2]:end[2]:stride[2], begin[3]:end[3]:stride[3], \
                begin[4]:end[4]:stride[4], begin[5]:end[5]:stride[5]]
    }[dim](input, begin, end, stride)

    return result


class TestStrideSliceOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'strided_slice'
        self.output = strided_slice_native_forward(self.input, self.begin,
                                                   self.end, self.stride)

        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'begin': self.begin,
            'end': self.end,
            'stride': self.stride
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(set(['Input']), 'Out')

    def initTestCase(self):
        self.input = np.random.rand(6)
        self.begin = [-4]
        self.end = [-3]
        self.stride = [1]


class TestStrideSliceOp1(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(6)
        self.begin = [3]
        self.end = [8]
        self.stride = [1]


class TestStrideSliceOp2(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(6)
        self.begin = [5]
        self.end = [0]
        self.stride = [-1]


class TestStrideSliceOp3(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(6)
        self.begin = [-1]
        self.end = [-3]
        self.stride = [-1]


class TestStrideSliceOp4(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 4, 6)
        self.begin = [0, -1, 0]
        self.end = [2, -3, 5]
        self.stride = [1, -1, 1]


class TestStrideSliceOp5(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3)
        self.begin = [1, 0, 0]
        self.end = [2, 1, 3]
        self.stride = [1, 1, 1]


class TestStrideSliceOp6(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3)
        self.begin = [1, -1, 0]
        self.end = [2, -3, 3]
        self.stride = [1, -1, 1]


class TestStrideSliceOp7(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3)
        self.begin = [1, 0, 0]
        self.end = [2, 2, 3]
        self.stride = [1, 1, 1]


class TestStrideSliceOp7(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(100, 3, 3)
        self.begin = [1, 0, 0]
        self.end = [109, 2, 3]
        self.stride = [1, 1, 1]


if __name__ == "__main__":
    unittest.main()
