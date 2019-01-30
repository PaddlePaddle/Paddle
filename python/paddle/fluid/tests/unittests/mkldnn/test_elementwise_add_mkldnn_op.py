#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_elementwise_add_op import *
'''
Some tests differ from the tests defined in test_elementwise_add_op.py
because MKLDNN does not support tensors of number of dimensions 3.
Such dimensions cause exceptions in MKLDNN reorder primitive.
'''


class TestMKLDNNElementwiseAddOp(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_scalar(TestElementwiseAddOp_scalar):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_scalar2(TestElementwiseAddOp_scalar2):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_Vector(TestElementwiseAddOp_Vector):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TesMKLDNNtElementwiseAddOp_broadcast_0(TestElementwiseAddOp_broadcast_0):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(2).astype(self.dtype)
        self.out = self.x + self.y.reshape(2, 1, 1, 1)

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_broadcast_1(TestElementwiseAddOp_broadcast_1):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(3).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 1, 1)

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_broadcast_2(TestElementwiseAddOp_broadcast_2):
    def init_input_output(self):
        self.x = np.random.rand(2, 2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 1, 4)

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_broadcast_3(TestElementwiseAddOp_broadcast_3):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_broadcast_4(TestElementwiseAddOp_broadcast_4):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_rowwise_add_0(
        TestElementwiseAddOp_rowwise_add_0):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 4, 1)

    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_rowwise_add_1(
        TestElementwiseAddOp_rowwise_add_1):
    def init_kernel_type(self):
        self.use_mkldnn = True


class TestMKLDNNElementwiseAddOp_channelwise_add(
        TestElementwiseAddOp_channelwise_add):
    def init_input_output(self):
        self.x = np.random.rand(3, 5, 20, 20).astype(self.dtype)
        self.y = np.random.rand(3, 1, 1, 1).astype(self.dtype)
        self.out = self.x + self.y

    def init_kernel_type(self):
        self.use_mkldnn = True


if __name__ == '__main__':
    unittest.main()
