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
import os
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci, convert_float_to_uint16
from paddle.fluid.tests.unittests.test_pool2d_op import TestPool2D_Op, avg_pool2D_forward_naive, max_pool2D_forward_naive
from paddle import enable_static


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestPoolBf16MklDNNOp(TestPool2D_Op):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def setUp(self):
        TestPool2D_Op.setUp(self)
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


if __name__ == "__main__":
    enable_static()
    unittest.main()
