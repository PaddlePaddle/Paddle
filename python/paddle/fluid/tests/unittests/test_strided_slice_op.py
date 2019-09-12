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


def strided_slice_native_forward(input, axes, starts, ends, strides):
    dim = input.ndim
    start = []
    end = []
    stride = []
    for i in range(dim):
        start.append(0)
        end.append(input.shape[i])
        stride.append(1)

    for i in range(len(axes)):
        start[axes[i]] = starts[i]
        end[axes[i]] = ends[i]
        stride[axes[i]] = strides[i]

    result = {
        1: lambda input, start, end, stride: input[start[0]:end[0]:stride[0]],
        2: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1]],
        3: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1], start[2]:end[2]:stride[2]],
        4: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1], start[2]:end[2]:stride[2], start[3]:end[3]:stride[3]],
        5: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1], start[2]:end[2]:stride[2], start[3]:end[3]:stride[3], start[4]:end[4]:stride[4]],
        6: lambda input, start, end, stride: input[start[0]:end[0]:stride[0], \
                start[1]:end[1]:stride[1], start[2]:end[2]:stride[2], start[3]:end[3]:stride[3], \
                start[4]:end[4]:stride[4], start[5]:end[5]:stride[5]]
    }[dim](input, start, end, stride)

    return result


class TestStrideSliceOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'strided_slice'
        self.output = strided_slice_native_forward(
            self.input, self.axes, self.starts, self.ends, self.strides)

        self.inputs = {'Input': self.input}
        self.outputs = {'Out': self.output}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'strides': self.strides
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(set(['Input']), 'Out')

    def initTestCase(self):
        self.input = np.random.rand(6)
        self.axes = [0]
        self.starts = [-4]
        self.ends = [-3]
        self.strides = [1]


class TestStrideSliceOp1(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(6)
        self.axes = [0]
        self.starts = [3]
        self.ends = [8]
        self.strides = [1]


class TestStrideSliceOp2(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(6)
        self.axes = [0]
        self.starts = [5]
        self.ends = [0]
        self.strides = [-1]


class TestStrideSliceOp3(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(6)
        self.axes = [0]
        self.starts = [-1]
        self.ends = [-3]
        self.strides = [-1]


class TestStrideSliceOp4(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 4, 6)
        self.axes = [0, 1, 2]
        self.starts = [0, -1, 0]
        self.ends = [2, -3, 5]
        self.strides = [1, -1, 1]


class TestStrideSliceOp5(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 1, 3]
        self.strides = [1, 1, 1]


class TestStrideSliceOp6(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3)
        self.axes = [0, 1, 2]
        self.starts = [1, -1, 0]
        self.ends = [2, -3, 3]
        self.strides = [1, -1, 1]


class TestStrideSliceOp7(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3)
        self.axes = [0, 1, 2]
        self.starts = [1, 0, 0]
        self.ends = [2, 2, 3]
        self.strides = [1, 1, 1]


class TestStrideSliceOp8(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(1, 3, 1)
        self.axes = [1]
        self.starts = [1]
        self.ends = [2]
        self.strides = [1]


class TestStrideSliceOp9(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(1, 3, 1)
        self.axes = [1]
        self.starts = [-1]
        self.ends = [-2]
        self.strides = [-1]


class TestStrideSliceOp10(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3)
        self.axes = [0, 1]
        self.starts = [1, 0]
        self.ends = [2, 2]
        self.strides = [1, 1]


class TestStrideSliceOp11(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4)
        self.axes = [0, 1, 2, 3]
        self.starts = [1, 0, 0, 0]
        self.ends = [2, 2, 3, 4]
        self.strides = [1, 1, 1, 2]


class TestStrideSliceOp12(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 4, 5)
        self.axes = [0, 1, 2, 3, 4]
        self.starts = [1, 0, 0, 0, 0]
        self.ends = [2, 2, 3, 4, 4]
        self.strides = [1, 1, 1, 1, 1]


class TestStrideSliceOp13(TestStrideSliceOp):
    def initTestCase(self):
        self.input = np.random.rand(3, 3, 3, 6, 7, 8)
        self.axes = [0, 1, 2, 3, 4, 5]
        self.starts = [1, 0, 0, 0, 1, 2]
        self.ends = [2, 2, 3, 1, 2, 8]
        self.strides = [1, 1, 1, 1, 1, 2]


if __name__ == "__main__":
    unittest.main()
