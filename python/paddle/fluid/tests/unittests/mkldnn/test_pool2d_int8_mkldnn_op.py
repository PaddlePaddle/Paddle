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
from __future__ import division

import unittest
import numpy as np

import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTestInt8
from paddle.fluid.tests.unittests.test_pool2d_op import TestPool2D_Op, avg_pool2D_forward_naive, max_pool2D_forward_naive


class TestPool2dMKLDNNInt8_Op(OpTestInt8):
    def setUp(self):
        self.op_type = "pool2d"
        self.use_cudnn = False
        self.init_kernel_type()
        self.use_mkldnn = False
        self.init_data_type()
        self.init_test_case()
        self.padding_algorithm = "EXPLICIT"
        self.init_paddings()
        self.init_global_pool()
        self.init_kernel_type()
        self.init_pool_type()
        self.init_ceil_mode()
        self.init_exclusive()
        self.init_adaptive()
        self.init_data_format()
        self.init_shape()

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'ceil_mode': self.ceil_mode,
            'data_format': self.data_format,
            'exclusive': self.exclusive,
            'adaptive': self.adaptive,
            "padding_algorithm": self.padding_algorithm,
        }

        assert self.dtype in [np.int8, np.uint8
                              ], 'Dtype should be int8 or uint8'
        input = np.random.randint(0, 100, self.shape).astype(self.dtype)
        output = (self.pool2D_forward_naive(
            input, self.ksize, self.strides, self.paddings, self.global_pool,
            self.ceil_mode, self.exclusive, self.adaptive,
            self.dtype)).astype(self.dtype)
        self.inputs = {'X': OpTestInt8.np_dtype_to_fluid_dtype(input)}
        self.outputs = {'Out': output}

    def has_cudnn(self):
        return core.is_compiled_with_cuda() and self.use_cudnn

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_shape(self):
        self.shape = [2, 3, 5, 5]

    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_paddings(self):
        self.paddings = [0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = True

    def init_ceil_mode(self):
        self.ceil_mode = False

    def init_exclusive(self):
        self.exclusive = True

    def init_adaptive(self):
        self.adaptive = False

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.int8

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output_with_place(
            core.CPUPlace(), atol=1e-5, check_dygraph=False)


class TestCase1Avg(TestPool2dMKLDNNInt8_Op):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_global_pool(self):
        self.global_pool = False

    def init_exclusive(self):
        self.exclusive = True


class TestCase2Avg(TestPool2dMKLDNNInt8_Op):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_global_pool(self):
        self.global_pool = False

    def init_exclusive(self):
        self.exclusive = False


class TestCase0Max(TestPool2dMKLDNNInt8_Op):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase1Max(TestCase1Avg):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase2Max(TestCase2Avg):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


def create_test_s8_u8_class(parent):
    class TestS8Case(parent):
        def init_data_type(self):
            self.dtype = np.int8

    class TestU8Case(parent):
        def init_data_type(self):
            self.dtype = np.uint8

    cls_name_s8 = "{0}_{1}".format(parent.__name__, "mkldnn_s8")
    cls_name_u8 = "{0}_{1}".format(parent.__name__, "mkldnn_u8")
    TestS8Case.__name__ = cls_name_s8
    TestU8Case.__name__ = cls_name_u8
    globals()[cls_name_s8] = TestS8Case
    globals()[cls_name_u8] = TestU8Case


create_test_s8_u8_class(TestPool2dMKLDNNInt8_Op)
create_test_s8_u8_class(TestCase1Avg)
create_test_s8_u8_class(TestCase2Avg)
create_test_s8_u8_class(TestCase0Max)
create_test_s8_u8_class(TestCase1Max)
create_test_s8_u8_class(TestCase2Max)

if __name__ == '__main__':
    unittest.main()
