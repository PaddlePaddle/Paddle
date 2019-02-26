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

import unittest
import numpy as np

import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_conv2d_op import TestConv2dOp


def conv2d_bias_forward_naive(out, bias):
    (in_n, out_c, out_h, out_w) = out.shape

    for l in range(out_c):
        for i in range(out_h):
            for j in range(out_w):
                out[:, l, i, j] = out[:, l, i, j] + bias[l]
    return out


def conv_2d_residual_naive(out, residual):
    (in_n, out_c, out_h, out_w) = out.shape
    for i in range(in_n):
        for j in range(out_c):
            for k in range(out_h):
                for l in range(out_w):
                    out[i, j, k, l] = out[i, j, k, l] + residual[i, j, k, l]
    return out


class TestConv2dMKLDNNBiasResidualOp(TestConv2dOp):
    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        self.data_format = "NCHW"
        self.use_mkldnn = True
        self._cpu_only = True

    def init_test_case(self):
        self.fuse_bias = False
        self.bias_size = [6]
        self.fuse_residual = False
        self.input_residual_size = [2, 6, 3, 3]
        self.fuse_relu_before_depthwise_conv = False
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def setUp(self):
        TestConv2dOp.setUp(self)
        output = self.outputs['Output']
        if hasattr(self, "fuse_bias"):
            if self.fuse_bias and hasattr(self, "bias_size"):
                bias = np.random.random(self.bias_size).astype(self.dtype)
                output = conv2d_bias_forward_naive(output, bias)
                output = output.astype(self.dtype)
                self.inputs['Bias'] = OpTest.np_dtype_to_fluid_dtype(bias)

        if hasattr(self, "fuse_residual"):
            if self.fuse_residual and hasattr(self, "input_residual_size"):
                input_residual = np.random.random(
                    self.input_residual_size).astype(self.dtype)
                output = conv_2d_residual_naive(output, input_residual)
                output = output.astype(self.dtype)

                self.attrs['fuse_residual_connection'] = self.fuse_residual
                self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                    input_residual)

        self.outputs['Output'] = output


class TestMKLDNNFuseReluBiasResidual(TestConv2dMKLDNNBiasResidualOp):
    def init_test_case(self):
        # TestConv2dMKLDNNBiasResidualOp.init_test_case(self)
        self.fuse_bias = True
        self.fuse_residual = True
        self.fuse_relu_before_depthwise_conv = True
        self.bias_size = [6]
        self.input_residual_size = [2, 6, 5, 5]
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def test_check_grad(self):
        pass

    def test_check_grad_no_filter(self):
        pass

    def test_check_grad_no_input(self):
        pass


class TestWithPadWithBias(TestConv2dMKLDNNBiasResidualOp):
    def init_test_case(self):
        TestConv2dMKLDNNBiasResidualOp.init_test_case(self)
        self.pad = [1, 1]
        self.fuse_bias = True
        self.bias_size = [6]
        self.fuse_residual = True
        self.input_residual_size = [2, 6, 5, 5]

    def test_check_grad(self):
        pass

    def test_check_grad_no_filter(self):
        pass

    def test_check_grad_no_input(self):
        pass


class TestWithStride(TestConv2dMKLDNNBiasResidualOp):
    def init_test_case(self):
        TestConv2dMKLDNNBiasResidualOp.init_test_case(self)
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]


class TestWithGroup(TestConv2dMKLDNNBiasResidualOp):
    def init_group(self):
        self.groups = 3


class TestWith1x1(TestConv2dMKLDNNBiasResidualOp):
    def init_test_case(self):
        TestConv2dMKLDNNBiasResidualOp.init_test_case(self)
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


class TestWithInput1x1Filter1x1(TestConv2dMKLDNNBiasResidualOp):
    def init_test_case(self):
        TestConv2dMKLDNNBiasResidualOp.init_test_case(self)
        self.input_size = [2, 3, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


if __name__ == '__main__':
    unittest.main()
