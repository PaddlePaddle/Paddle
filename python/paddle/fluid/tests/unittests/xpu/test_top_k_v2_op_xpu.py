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
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid.core as core

paddle.enable_static()


def numpy_topk(x, k=1, axis=-1, largest=True):
    if axis < 0:
        axis = len(x.shape) + axis
    if largest:
        indices = np.argsort(-x, axis=axis)
    else:
        indices = np.argsort(x, axis=axis)
    if largest:
        value = -np.sort(-x, axis=axis)
    else:
        value = np.sort(x, axis=axis)
    indices = indices.take(indices=range(0, k), axis=axis)
    value = value.take(indices=range(0, k), axis=axis)
    return value, indices


class TestTopkOp(OpTest):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 20)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad(set(['X']), 'Out')


class TestTopkOp1(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp2(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp3(TestTopkOp):
    def init_args(self):
        self.k = 5
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp4(TestTopkOp):
    def init_args(self):
        self.k = 1
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp5(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 2
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp6(TestTopkOp):
    def init_args(self):
        self.k = 5
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(8, 32, 64)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp7(TestTopkOp):
    def init_args(self):
        self.k = 10
        self.axis = 2
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(8, 5, 10, 16)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp8(TestTopkOp):
    def init_args(self):
        self.k = 1
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(8, 32, 64)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp9(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp10(TestTopkOp):
    def init_args(self):
        self.k = 3
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp11(TestTopkOp):
    def init_args(self):
        self.k = 5
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


class TestTopkOp12(TestTopkOp):
    def init_args(self):
        self.k = 1
        self.axis = 1
        self.largest = True

    def setUp(self):
        self.op_type = "top_k_v2"
        self.dtype = np.float32
        self.input_data = np.random.rand(10, 10, 5)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'k': self.k, 'axis': self.axis, 'largest': self.largest}
        output, indices = numpy_topk(
            self.input_data, axis=self.axis, k=self.k, largest=self.largest)
        self.outputs = {'Out': output, 'Indices': indices}


if __name__ == "__main__":
    unittest.main()
