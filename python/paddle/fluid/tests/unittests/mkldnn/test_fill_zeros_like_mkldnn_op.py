#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool
from paddle.fluid.framework import convert_np_dtype_to_dtype_
import paddle
import paddle.fluid.core as core


@OpTestTool.skip_if_not_cpu()
class TestFillZerosLike2DOneDNNOpV2(OpTest):
    def setUp(self):
        self.initialize_op()
        self.inputs = {
            'X': np.random.random(self.input_shape).astype(self.dtype)
        }
        self.outputs = {
            'Out': np.zeros_like(self.inputs["X"]).astype(self.dtype)
        }

    def initialize_op(self):
        self.op_type = "fill_zeros_like2"
        self.attrs = {
            'dtype': convert_np_dtype_to_dtype_(self.dtype),
            'use_mkldnn': True
        }
        self.set_dtype()
        self.set_input_shape()

    def set_dtype(self):
        self.dtype = np.float32

    def set_input_shape(self):
        self.input_shape = (13, 42)

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())


class TestFillZerosLike4DOneDNNOpV2(TestFillZerosLike2DOneDNNOpV2):
    def set_input_shape(self):
        self.input_shape = (5, 6, 7, 9)


class TestFillZerosLike5DOneDNNOpV2(TestFillZerosLike2DOneDNNOpV2):
    def set_input_shape(self):
        self.input_shape = (3, 5, 7, 8, 2)


class TestFillZerosLikeBF16OneDNNOpV2(TestFillZerosLike2DOneDNNOpV2):
    def set_dtype(self):
        self.dtype = np.uint16


class TestFillZerosLikeINT8OneDNNOpV2(TestFillZerosLike2DOneDNNOpV2):
    def set_dtype(self):
        self.dtype = np.int8


class TestFillZerosLikeUINT8OneDNNOpV2(TestFillZerosLike2DOneDNNOpV2):
    def set_dtype(self):
        self.dtype = np.uint8


class TestFillZerosLike2DOneDNNOp(TestFillZerosLike2DOneDNNOpV2):
    def initialize_op(self):
        self.op_type = "fill_zeros_like"
        self.attrs = {'use_mkldnn': True}
        self.set_dtype()
        self.set_input_shape()


class TestFillZerosLikeBF16OneDNNOp(TestFillZerosLike2DOneDNNOp):
    def set_dtype(self):
        self.dtype = np.uint16


class TestFillZerosLikeINT8OneDNNOp(TestFillZerosLike2DOneDNNOp):
    def set_dtype(self):
        self.dtype = np.int8


class TestFillZerosLikeUINT8OneDNNOp(TestFillZerosLike2DOneDNNOp):
    def set_dtype(self):
        self.dtype = np.uint8


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
