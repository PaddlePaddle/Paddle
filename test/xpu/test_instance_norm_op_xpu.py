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

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import base
from paddle.base import Program, program_guard

paddle.enable_static()


def _reference_instance_norm_naive(x, scale, bias, epsilon, mean, var):
    x_shape = x.shape
    if len(x_shape) == 2:
        x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
    n, c, h, w = x.shape

    mean_tile = np.reshape(mean, (n, c, 1, 1))
    mean_tile = np.tile(mean_tile, (1, 1, h, w))
    var_tile = np.reshape(var, (n, c, 1, 1))
    var_tile = np.tile(var_tile, (1, 1, h, w))

    x_norm = (x - mean_tile) / np.sqrt(var_tile + epsilon).astype('float32')
    scale_tile = np.reshape(scale, (1, c, 1, 1))
    scale_tile = np.tile(scale_tile, (n, 1, h, w))
    bias_tile = np.reshape(bias, (1, c, 1, 1))
    bias_tile = np.tile(bias_tile, (n, 1, h, w))
    y = scale_tile * x_norm + bias_tile
    if len(x_shape) == 2:
        y = np.reshape(y, x_shape)
    return y, mean, var


def _cal_mean_variance(x, epsilon, mean_shape):
    mean = np.reshape(np.mean(x, axis=(2, 3)), mean_shape)
    var = np.reshape(np.var(x, axis=(2, 3)), mean_shape)
    return mean, var


class XPUTestInstanceNormOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'instance_norm'
        self.use_dynamic_create_class = False

    class XPUTestInstanceNormOp(XPUOpTest):
        def setUp(self):
            self.op_type = "instance_norm"
            self.dtype = self.in_type
            self.shape = [2, 3, 4, 5]
            self.epsilon = 1e-05
            self.no_grad_set = None
            self.set_attrs()
            self.atol = 1e-5
            if self.dtype == np.float16:
                self.atol = 1e-2

            np.random.seed(12345)
            epsilon = self.epsilon
            shape = self.shape
            n, c, h, w = shape[0], shape[1], shape[2], shape[3]
            scale_shape = [c]
            mean_shape = [n * c]

            x_np = np.random.random_sample(shape).astype(self.dtype)
            scale_np = np.random.random_sample(scale_shape).astype(np.float32)
            bias_np = np.random.random_sample(scale_shape).astype(np.float32)
            mean, variance = self.set_global_mean_var(mean_shape, x_np)

            (
                ref_y_np,
                ref_saved_mean,
                variance_tmp,
            ) = _reference_instance_norm_naive(
                x_np, scale_np, bias_np, epsilon, mean, variance
            )

            ref_saved_variance = 1 / np.sqrt(variance_tmp + epsilon)

            self.inputs = {'X': x_np, 'Scale': scale_np, 'Bias': bias_np}
            self.outputs = {
                'Y': ref_y_np,
                'SavedMean': ref_saved_mean,
                'SavedVariance': ref_saved_variance,
            }
            self.attrs = {'epsilon': epsilon, 'use_xpu': True}

        def set_global_mean_var(self, mean_shape, x):
            mean, variance = _cal_mean_variance(x, self.epsilon, mean_shape)
            return mean, variance

        def set_attrs(self):
            pass

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0), atol=self.atol)

        def test_check_grad(self):
            self.check_grad_with_place(
                paddle.XPUPlace(0),
                ['X', 'Scale', 'Bias'],
                ['Y'],
                self.no_grad_set,
            )

    class TestXPUInstanceNormOp1(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [10, 12, 32, 32]

    class TestXPUInstanceNormOp2(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [4, 5, 6, 7]

    class TestXPUInstanceNormOp3(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [1, 8, 16, 16]

    class TestXPUInstanceNormOp4(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [4, 16, 256, 128]

    class TestXPUInstanceNormOp5(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [10, 3, 512, 1]

    class TestXPUInstanceNormOp6(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [10, 12, 32, 32]
            self.no_grad_set = {'Scale', 'Bias'}

    class TestXPUInstanceNormOp7(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [4, 5, 6, 7]
            self.no_grad_set = {'Scale', 'Bias'}

    class TestXPUInstanceNormOp8(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [1, 8, 16, 16]
            self.no_grad_set = {'Scale', 'Bias'}

    class TestXPUInstanceNormOp9(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [4, 16, 256, 128]
            self.no_grad_set = {'Scale', 'Bias'}

    class TestXPUInstanceNormOp10(XPUTestInstanceNormOp):
        def set_attrs(self):
            self.shape = [10, 3, 512, 1]
            self.no_grad_set = {'Scale', 'Bias'}

    class TestInstanceNormOpError(XPUOpTest):
        def setUp(self):
            self.__class__.op_type = "instance_norm"
            self.__class__.no_need_check_grad = True
            self.dtype = self.in_type

        def test_errors(self):
            with program_guard(Program(), Program()):
                # the input of instance_norm must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], base.XPUPlace(0)
                )
                self.assertRaises(TypeError, paddle.static.nn.instance_norm, x1)

                # the input dtype of instance_norm must be float32
                x2 = paddle.static.data(
                    name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
                )
                self.assertRaises(TypeError, paddle.static.nn.instance_norm, x2)

                # the first dimension of input for instance_norm must between [2d, 5d]
                x3 = paddle.static.data(name='x', shape=[3], dtype="float32")
                self.assertRaises(
                    ValueError, paddle.static.nn.instance_norm, x3
                )


support_types = get_xpu_op_support_types('instance_norm')
for stype in support_types:
    create_test_class(globals(), XPUTestInstanceNormOp, stype)

if __name__ == "__main__":
    unittest.main()
