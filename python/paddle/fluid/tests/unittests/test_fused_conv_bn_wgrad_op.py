#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, skip_check_grad_ci

import paddle
import paddle.fluid.core as core
import paddle.nn as nn
from paddle.nn import Conv2D


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or paddle.device.cuda.get_device_capability()[0] < 8,
    "only support with cuda and Ampere or later devices",
)
class TestFusedConvBnWgradOp(OpTest):
    def setUp(self):
        self.op_type = "fused_conv_bn_wgrad"
        self.dtype = np.float16
        self.math_type = np.float32
        self.outputs = None
        self.padding_algorithm = "EXIPLICIT"
        self.data_format = "NHWC"
        self.groups = 1

        self.init_dilation()
        self.init_test_case()
        self.init_paddings()
        self.init_attr()

        self.c_dim = self.input_size[-1]
        self.bn_x = np.random.random(self.input_size).astype(self.dtype) - 0.5
        self.scale = np.random.random(self.c_dim).astype(self.dtype) - 0.5
        self.bias = np.random.random(self.c_dim).astype(self.dtype) - 0.5

        paddle.disable_static()
        paddle.set_default_dtype(self.dtype)
        self.relu = nn.ReLU()
        self.conv = Conv2D(
            in_channels=self.input_size[-1],
            out_channels=self.filter_size[0],
            kernel_size=self.filter_size[-1],
            stride=self.stride,
            padding=self.pad,
            groups=1,
            bias_attr=False,
            data_format=self.data_format,
        )

        # calculate reference
        dy, w, dw = self.init_outputs()

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'dilations': self.dilations,
        }

        self.inputs = {
            'BN_X': OpTest.np_dtype_to_fluid_dtype(self.bn_x),
            'W': OpTest.np_dtype_to_fluid_dtype(w),
            'dY': OpTest.np_dtype_to_fluid_dtype(dy),
            'Scale': OpTest.np_dtype_to_fluid_dtype(self.scale),
            'Bias': OpTest.np_dtype_to_fluid_dtype(self.bias),
        }

        self.outputs = {
            'dW': dw,
        }

    def has_cuda(self):
        return core.is_compiled_with_cuda()

    def test_check_output(self):
        paddle.enable_static()
        if self.has_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=2e-2)

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 5, 5, 8]  # NHWC
        assert np.mod(self.input_size[-1], self.groups) == 0
        f_c = self.input_size[-1] // self.groups
        self.filter_size = [16, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_attr(self):
        self.fuse_shortcut = False
        self.fuse_dual = False

    def init_outputs(self):
        # calculate dgrad with paddle
        scale_tensor = paddle.to_tensor(
            self.scale.reshape(1, 1, 1, self.c_dim), stop_gradient=False
        )
        bias_tensor = paddle.to_tensor(
            self.bias.reshape(1, 1, 1, self.c_dim), stop_gradient=False
        )
        bn_x_tensor = paddle.to_tensor(self.bn_x, stop_gradient=False)
        after_bias_tensor = (bn_x_tensor * scale_tensor) + bias_tensor
        y_tensor = self.conv(self.relu(after_bias_tensor))
        y_dim = y_tensor.shape
        dy = np.random.random(y_dim).astype(self.dtype)
        paddle.autograd.backward([y_tensor], [paddle.to_tensor(dy)], True)
        return (
            dy,
            self.conv.weight.numpy().astype(self.dtype),
            self.conv.weight.grad.numpy().astype(self.dtype),
        )


if __name__ == '__main__':
    unittest.main()
