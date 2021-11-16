# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
from paddle.fluid.tests.unittests.test_pool2d_op import TestPool2D_Op_Mixin, max_pool2D_forward_naive
from paddle.fluid.tests.unittests.npu.test_pool2d_op_npu import pool2d_backward_navie as pool2d_backward_naive
from paddle import enable_static


@OpTestTool.skip_if_not_cpu_bf16()
class TestPoolBf16MklDNNOpGrad(TestPool2D_Op_Mixin, OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_data_type(self):
        self.dtype = np.uint16

    def setUp(self):
        super(TestPoolBf16MklDNNOpGrad, self).setUp()
        self.attrs['mkldnn_data_type'] = "bfloat16"
        self.x_fp32 = np.random.random(self.shape).astype(np.float32)

        output = self.pool2D_forward_naive(
            self.x_fp32, self.ksize, self.strides, self.paddings,
            self.global_pool, self.ceil_mode, self.exclusive, self.adaptive,
            "float32").astype(np.float32)

        self.inputs = {'X': convert_float_to_uint16(self.x_fp32)}
        self.outputs = {'Out': convert_float_to_uint16(output)}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        x_grad = pool2d_backward_naive(
            self.x_fp32,
            ksize=self.ksize,
            strides=self.strides,
            paddings=self.paddings,
            global_pool=self.global_pool,
            ceil_mode=False,
            exclusive=self.exclusive,
            adaptive=self.adaptive,
            data_format=self.data_format,
            pool_type=self.pool_type,
            padding_algorithm=self.padding_algorithm)
        x_grad = x_grad / np.prod(self.outputs['Out'].shape)
        self.check_grad_with_place(
            core.CPUPlace(), set(['X']), 'Out', user_defined_grads=[x_grad])


@OpTestTool.skip_if_not_cpu_bf16()
class TestPoolBf16MklDNNOp(TestPool2D_Op_Mixin, OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def setUp(self):
        TestPool2D_Op_Mixin.setUp(self)
        self.dtype = np.uint16

        input = np.random.random(self.shape).astype(np.float32)
        output = (self.pool2D_forward_naive(
            input, self.ksize, self.strides, self.paddings, self.global_pool,
            self.ceil_mode, self.exclusive, self.adaptive,
            "float32")).astype(np.float32)

        self.inputs = {'X': convert_float_to_uint16(input)}
        self.outputs = {'Out': convert_float_to_uint16(output)}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        pass


class TestCase1Avg(TestPoolBf16MklDNNOp):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_global_pool(self):
        self.global_pool = False

    def init_exclusive(self):
        self.exclusive = True


class TestCase2Avg(TestPoolBf16MklDNNOp):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_global_pool(self):
        self.global_pool = False

    def init_exclusive(self):
        self.exclusive = False


class TestCase0Max(TestPoolBf16MklDNNOp):
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


class TestCase1PadZeroExclusiveAvgGrad(TestPoolBf16MklDNNOpGrad):
    def init_test_case(self):
        self.ksize = [3, 3]
        self.strides = [1, 1]

    def init_shape(self):
        self.shape = [2, 3, 7, 7]

    def init_paddings(self):
        self.paddings = [0, 0]

    def init_global_pool(self):
        self.global_pool = False

    def init_exclusive(self):
        self.exclusive = True


class TestCase2PadOneNonExclusiveAvgGrad(TestCase1PadZeroExclusiveAvgGrad):
    def init_exclusive(self):
        self.exclusive = False


class TestCase0InitialMaxGrad(TestPoolBf16MklDNNOpGrad):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase1PadZeroExclusiveMaxGrad(TestCase1PadZeroExclusiveAvgGrad):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase2PadOneNonExclusiveMaxGrad(TestCase2PadOneNonExclusiveAvgGrad):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


if __name__ == "__main__":
    enable_static()
    unittest.main()
