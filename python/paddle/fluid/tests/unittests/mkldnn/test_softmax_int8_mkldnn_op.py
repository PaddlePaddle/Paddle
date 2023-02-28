# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.core as core
import paddle.nn.functional as F
from paddle import enable_static
from paddle.fluid.tests.unittests.op_test import OpTest


def stable_softmax(x, clip):
    '''Compute the softmax of vector x in a numerically stable way.'''
    shiftx = (x.astype(np.int32) - np.max(x)).clip(clip)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxMKLDNNOpInt8(OpTest):
    def init_data_type(self):
        self.dtype = np.int8
        self.mkldnn_data_type = 'int8'

    def get_shape(self):
        return [10, 10]

    def get_axis(self):
        return -1

    def get_input(self):
        return np.random.randint(1, 127, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -32)
        return np.round(out * 127).astype(self.dtype)

    def setUp(self):
        self.op_type = 'softmax'
        self.prim_op_type = "comp"
        self._cpu_only = True
        self.init_data_type()
        self.shape = self.get_shape()
        self.axis = self.get_axis()
        self.python_api = F.softmax
        np.random.seed(0)

        self.x = self.get_input()
        self.out = self.get_output()
        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axis': self.axis,
            'use_mkldnn': True,
            'mkldnn_data_type': self.mkldnn_data_type,
        }
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_dygraph=False)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp2Int8(TestSoftmaxMKLDNNOpInt8):
    def get_shape(self):
        return [2, 3, 4, 5]


class TestSoftmaxMKLDNNOp3Int8(TestSoftmaxMKLDNNOpInt8):
    def get_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 0


class TestSoftmaxMKLDNNOp4Int8(TestSoftmaxMKLDNNOpInt8):
    def get_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 1


class TestSoftmaxMKLDNNOp5Int8(TestSoftmaxMKLDNNOpInt8):
    def get_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 2


class TestSoftmaxMKLDNNOp6Int8(TestSoftmaxMKLDNNOpInt8):
    def get_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 3


class TestSoftmaxMKLDNNOpUint8(TestSoftmaxMKLDNNOpInt8):
    def init_data_type(self):
        self.dtype = np.uint8
        self.mkldnn_data_type = 'uint8'

    def get_input(self):
        return np.random.uniform(0, 255, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -64)
        return np.round(out * 255).astype(self.dtype)


class TestSoftmaxMKLDNNOp2Uint8(TestSoftmaxMKLDNNOpUint8):
    def get_shape(self):
        return [2, 3, 4, 5]


class TestSoftmaxMKLDNNOp3Uint8(TestSoftmaxMKLDNNOpUint8):
    def get_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 0


class TestSoftmaxMKLDNNOp4Uint8(TestSoftmaxMKLDNNOpUint8):
    def get_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 1


class TestSoftmaxMKLDNNOp5Uint8(TestSoftmaxMKLDNNOpUint8):
    def get_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 2


class TestSoftmaxMKLDNNOp6Uint8(TestSoftmaxMKLDNNOpUint8):
    def get_shape(self):
        return [2, 3, 4, 5]

    def get_axis(self):
        return 3


if __name__ == '__main__':
    enable_static()
    unittest.main()
