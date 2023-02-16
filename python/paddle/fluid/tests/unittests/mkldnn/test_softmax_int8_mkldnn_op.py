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

from paddle.fluid.tests.unittests.test_softmax_op import (
    TestSoftmaxOp,
    TestSoftmaxOp2,
    TestSoftmaxOp3,
    TestSoftmaxOp4,
    TestSoftmaxOp5,
    TestSoftmaxOp6,
)


def stable_softmax(x, clip):
    '''Compute the softmax of vector x in a numerically stable way.'''
    shiftx = (x.astype(np.int32) - np.max(x)).clip(clip)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestSoftmaxMKLDNNOpInt8(TestSoftmaxOp):
    def init_data_type(self):
        self.dtype = np.int8

    def init_kernel_type(self):
        self.use_mkldnn = True

    def get_input(self):
        return np.random.uniform(-128, 127, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -32)
        return np.round(out * 127).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOpUint8(TestSoftmaxOp):
    def init_data_type(self):
        self.dtype = np.uint8

    def init_kernel_type(self):
        self.use_mkldnn = True

    def get_input(self):
        return np.random.uniform(0, 255, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -64)
        return np.round(out * 255).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp2Int8(TestSoftmaxOp2):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.int8

    def get_input(self):
        return np.random.uniform(-128, 127, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -32)
        return np.round(out * 127).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp2Uint8(TestSoftmaxOp2):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.uint8

    def get_input(self):
        return np.random.uniform(0, 255, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -64)
        return np.round(out * 255).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp3Int8(TestSoftmaxOp3):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.int8

    def get_input(self):
        return np.random.uniform(-128, 127, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -32)
        return np.round(out * 127).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp3Uint8(TestSoftmaxOp3):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.uint8

    def get_input(self):
        return np.random.uniform(0, 255, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -64)
        return np.round(out * 255).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp4Int8(TestSoftmaxOp4):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.int8

    def get_input(self):
        return np.random.uniform(-128, 127, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -32)
        return np.round(out * 127).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp4Uint8(TestSoftmaxOp4):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.uint8

    def get_input(self):
        return np.random.uniform(0, 255, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -64)
        return np.round(out * 255).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp5Int8(TestSoftmaxOp5):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.int8

    def get_input(self):
        return np.random.uniform(-128, 127, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -32)
        return np.round(out * 127).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp5Uint8(TestSoftmaxOp5):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.uint8

    def get_input(self):
        return np.random.uniform(0, 255, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -64)
        return np.round(out * 255).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp6Int8(TestSoftmaxOp6):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.int8

    def get_input(self):
        return np.random.uniform(-128, 127, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -32)
        return np.round(out * 127).astype(self.dtype)

    def test_check_grad(self):
        pass


class TestSoftmaxMKLDNNOp6Uint8(TestSoftmaxOp6):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.uint8

    def get_input(self):
        return np.random.uniform(0, 255, self.shape).astype(self.dtype)

    def get_output(self):
        out = np.apply_along_axis(stable_softmax, self.axis, self.x, -64)
        return np.round(out * 255).astype(self.dtype)

    def test_check_grad(self):
        pass


if __name__ == '__main__':
    from paddle import enable_static

    enable_static()
    unittest.main()
