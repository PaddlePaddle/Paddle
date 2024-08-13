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

import unittest
from functools import reduce
from operator import mul

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16
from op_test_xpu import XPUOpTest

import paddle
from paddle.framework import core

paddle.enable_static()


def ref_layer_norm(x, scale, bias, epsilon, begin_norm_axis=1):
    x_shape = x.shape
    left = reduce(mul, x_shape[0:begin_norm_axis], 1)
    right = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)
    x.shape = [left, right]
    mean = np.mean(x, axis=1)
    variance = np.var(x, axis=1) + epsilon
    y = np.divide(
        (x - mean.reshape([left, 1])), (np.sqrt(variance)).reshape([left, 1])
    )
    if scale is not None:
        y = scale.reshape([1, right]) * y
    if bias is not None:
        y = y + bias.reshape([1, right])
    x.shape, y.shape = x_shape, x_shape
    mean.shape = x_shape[0:begin_norm_axis]
    variance.shape = x_shape[0:begin_norm_axis]
    return y, mean, variance


class XPUTestLayerNormOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'layer_norm'
        self.use_dynamic_create_class = False

    class TestXPULayerNormOp(XPUOpTest):
        def setUp(self):
            self.op_type = "layer_norm"
            if self.in_type == np.uint16:
                self.dtype = np.float32
            else:
                self.dtype = self.in_type
            self.shape = [2, 3, 4, 5]
            self.epsilon = 1e-05
            self.begin_norm_axis = 1
            self.use_fp16_scale_bias = False
            self.use_bf16_scale_bias = False
            self.set_attrs()

            self.atol = 1e-4
            if self.dtype == np.float16 or self.in_type == np.uint16:
                self.atol = 1e-2

            right = reduce(
                mul, self.shape[self.begin_norm_axis : len(self.shape)], 1
            )
            np.random.seed(10)
            x_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
            scale_np = np.random.uniform(-1, 1, [right]).astype('float32')
            bias_np = np.random.uniform(-1, 1, [right]).astype('float32')
            if self.dtype == np.float16 and self.use_fp16_scale_bias:
                scale_np = scale_np.astype('float16')
                bias_np = bias_np.astype('float16')
            if (
                self.dtype == np.uint16 and self.use_bf16_scale_bias
            ):  # bfloat16 actually
                scale_np = convert_float_to_uint16(scale_np)
                bias_np = convert_float_to_uint16(bias_np)
            ref_y_np, ref_mean_np, ref_variance_np = ref_layer_norm(
                x_np, scale_np, bias_np, self.epsilon, self.begin_norm_axis
            )
            ref_y_np = ref_y_np.astype(self.dtype)

            self.inputs = {'X': x_np, 'Scale': scale_np, 'Bias': bias_np}
            self.outputs = {
                'Y': ref_y_np,
                'Mean': ref_mean_np,
                'Variance': ref_variance_np,
            }
            self.attrs = {
                'begin_norm_axis': self.begin_norm_axis,
                'use_xpu': True,
            }

        def set_attrs(self):
            pass

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0), atol=self.atol)

        def test_check_grad(self):
            self.check_grad_with_place(
                paddle.XPUPlace(0), ['X'], 'Y', max_relative_error=self.atol
            )

    class TestXPULayerNormOpAxis2(TestXPULayerNormOp):
        def set_attrs(self):
            self.begin_norm_axis = 2

    class TestXPULayerNormOpAxis3(TestXPULayerNormOp):
        def set_attrs(self):
            self.begin_norm_axis = 3

    class TestXPULayerNormOp2D(TestXPULayerNormOp):
        def set_attrs(self):
            self.shape = [10, 12]

    class TestXPULayerNormOp3D(TestXPULayerNormOp):
        def set_attrs(self):
            self.shape = [4, 5, 6]

    class TestXPULayerNormOpFP16(TestXPULayerNormOp):
        def set_attrs(self):
            self.use_fp16_scale_bias = False

    class TestXPULayerNormOpFP16_2D(TestXPULayerNormOp):
        def set_attrs(self):
            self.shape = [10, 12]
            self.use_fp16_scale_bias = False

    class TestXPULayerNormOpFP16_3D(TestXPULayerNormOp):
        def set_attrs(self):
            self.shape = [4, 5, 6]
            self.use_fp16_scale_bias = False

    class TestXPULayerNormOpBF16(TestXPULayerNormOp):
        def set_attrs(self):
            self.use_bf16_scale_bias = True
            if core.get_xpu_device_version(0) == core.XPUVersion.XPU3:
                self.dtype = np.uint16
            else:
                self.dtype = np.float32

    class TestXPULayerNormOpBF16_2D(TestXPULayerNormOp):
        def set_attrs(self):
            self.shape = [10, 12]
            self.use_bf16_scale_bias = True
            if core.get_xpu_device_version(0) == core.XPUVersion.XPU3:
                self.dtype = np.uint16
            else:
                self.dtype = np.float32

    class TestXPULayerNormOpBF16_3D(TestXPULayerNormOp):
        def set_attrs(self):
            self.shape = [4, 5, 6]
            self.use_bf16_scale_bias = True
            if core.get_xpu_device_version(0) == core.XPUVersion.XPU3:
                self.dtype = np.uint16
            else:
                self.dtype = np.float32

    # @check_run_big_shape_test()
    # class TestXPULayerNormOpLargeShape1(TestXPULayerNormOp):
    #     def set_attrs(self):
    #         self.shape = [1024, 5120]
    #         self.use_bf16_scale_bias = True
    #         self.use_fp16_scale_bias = True


support_types = get_xpu_op_support_types('layer_norm')
for stype in support_types:
    create_test_class(globals(), XPUTestLayerNormOp, stype)

if __name__ == "__main__":
    unittest.main()
