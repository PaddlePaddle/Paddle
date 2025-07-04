#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest

import numpy as np
from op_test import OpTest

import paddle
from paddle import base
from paddle.base import core


def _reference_instance_norm_naive(x, scale, bias, epsilon, mean, var):
    x_shape = x.shape
    if len(x_shape) < 4:
        x = np.reshape(x, (x.shape[0], x.shape[1], -1, 1))
    n, c, h, w = x.shape

    mean_tile = np.reshape(mean, (n, c, 1, 1))
    mean_tile = np.tile(mean_tile, (1, 1, h, w))
    var_tile = np.reshape(var, (n, c, 1, 1))
    var_tile = np.tile(var_tile, (1, 1, h, w))

    x_norm = (x - mean_tile) / np.sqrt(var_tile + epsilon)
    scale_tile = np.reshape(scale, (1, c, 1, 1))
    scale_tile = np.tile(scale_tile, (n, 1, h, w))
    bias_tile = np.reshape(bias, (1, c, 1, 1))
    bias_tile = np.tile(bias_tile, (n, 1, h, w))
    y = scale_tile * x_norm + bias_tile
    if len(x_shape) < 4:
        y = np.reshape(y, x_shape)
    return y, mean, var


def _reference_instance_norm_grad(x, d_y, scale, mean, var, epsilon):
    # d_scale = sum(d_y * (x-mean) / sqrt(var+epsilon))
    # d_offset = sum(d_y)
    # d_x = scale / sqrt(var+epsilon) * (d_y - np.mean(d_y, axis=(2,3)) - (x-mean)/sqrt(var+epsilon)* np.mean(y_grad * (x-mean)/sqrt(var+epsilon), axis=(2,3)))
    n, c, h, w = x.shape

    d_bias = np.sum(d_y, axis=(0, 2, 3))

    mean_tile = np.reshape(mean, (n, c, 1, 1))
    mean_tile = np.tile(mean_tile, (1, 1, h, w))
    var_tile = np.reshape(var, (n, c, 1, 1))
    var_tile = np.tile(var_tile, (1, 1, h, w))

    d_scale = np.sum(d_y * (x - mean_tile) * var_tile, axis=(0, 2, 3))
    var_inv = var_tile
    scale_tile = np.reshape(scale, (1, c, 1, 1))
    scale_tile = np.tile(scale_tile, (n, 1, h, w))

    d_x = (
        scale_tile
        * var_inv
        * (
            d_y
            - np.mean(d_y, axis=(2, 3), keepdims=True)
            - (x - mean_tile)
            * var_inv
            * np.mean(
                d_y * (x - mean_tile) * var_inv, axis=(2, 3), keepdims=True
            )
        )
    )
    return d_x, d_scale, d_bias


def _cal_mean_variance(x, epsilon, mean_shape):
    if len(x.shape) < 4:
        x = np.reshape(x, (x.shape[0], x.shape[1], -1, 1))
    mean = np.reshape(np.mean(x, axis=(2, 3)), mean_shape)
    var = np.reshape(np.var(x, axis=(2, 3)), mean_shape)
    return mean, var


def instance_norm_wrapper(x, weight=None, bias=None, esp=1e-05):
    return paddle.nn.functional.instance_norm(
        x, None, None, weight, bias, True, 0.9, esp
    )


class TestInstanceNormOp(OpTest):
    def setUp(self):
        self.op_type = "instance_norm"
        self.prim_op_type = "comp"
        self.python_api = instance_norm_wrapper
        self.public_python_api = instance_norm_wrapper
        self.python_out_sig = ['Y']
        self.fw_comp_rtol = 1e-6
        self.fw_comp_atol = 1e-6
        self.rev_comp_rtol = 1e-4
        self.rev_comp_atol = 1e-4
        self.cinn_rtol = 1e-4
        self.cinn_atol = 1e-4
        self.init_test_case()
        ref_y_np, ref_mean_np, ref_var_np_tmp = _reference_instance_norm_naive(
            self.x_np,
            self.scale_np,
            self.bias_np,
            self.epsilon,
            self.mean_np,
            self.var_np,
        )

        ref_var_np = 1 / np.sqrt(ref_var_np_tmp + self.epsilon)
        self.inputs = {
            'X': self.x_np,
            'Scale': self.scale_np,
            'Bias': self.bias_np,
        }
        self.attrs = {'epsilon': self.epsilon}
        self.outputs = {
            'Y': ref_y_np,
            'SavedMean': ref_mean_np,
            'SavedVariance': ref_var_np,
        }

    def test_check_output(self):
        self.check_output(check_prim=True, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Scale', 'Bias'],
            'Y',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )

    def init_test_case(self):
        x_shape = [2, 100, 4, 5]
        n, c, h, w = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        self.epsilon = 1e-05
        dtype = np.float32
        scale_shape = [c]
        mean_shape = [n * c]
        np.random.seed()
        self.x_np = np.random.random_sample(x_shape).astype(dtype)
        self.scale_np = np.random.random_sample(scale_shape).astype(dtype)
        self.bias_np = np.random.random_sample(scale_shape).astype(dtype)
        self.mean_np, self.var_np = _cal_mean_variance(
            self.x_np, self.epsilon, mean_shape
        )
        self.dtype = dtype


class TestInstanceNormFP64(TestInstanceNormOp):
    def init_test_case(self):
        x_shape = [2, 100, 4, 5]
        n, c, h, w = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        self.epsilon = 1e-5
        dtype = np.float64
        scale_shape = [c]
        mean_shape = [n * c]
        np.random.seed()
        self.x_np = np.random.random_sample(x_shape).astype(dtype)
        self.scale_np = np.ones(scale_shape).astype(dtype)
        self.bias_np = np.zeros(scale_shape).astype(dtype)
        self.mean_np, self.var_np = _cal_mean_variance(
            self.x_np, self.epsilon, mean_shape
        )
        self.cinn_atol = 1e-13
        self.cinn_rtol = 1e-13
        self.fw_comp_rtol = 1e-14
        self.fw_comp_atol = 1e-14
        self.rev_comp_rtol = 1e-13
        self.rev_comp_atol = 1e-13
        self.dtype = dtype


class TestInstanceNormCase1(TestInstanceNormOp):
    def init_test_case(self):
        x_shape = [2, 100, 4, 5]
        n, c, h, w = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        self.epsilon = 1e-05
        dtype = np.float32
        scale_shape = [c]
        mean_shape = [n * c]
        np.random.seed()
        self.x_np = np.random.random_sample(x_shape).astype(dtype)
        self.scale_np = np.ones(scale_shape).astype(dtype)
        self.bias_np = np.zeros(scale_shape).astype(dtype)
        self.mean_np, self.var_np = _cal_mean_variance(
            self.x_np, self.epsilon, mean_shape
        )


class TestInstanceNormCaseNCL(TestInstanceNormOp):
    def init_test_case(self):
        x_shape = [2, 100, 4]
        n, c = x_shape[0], x_shape[1]
        self.epsilon = 1e-05
        dtype = np.float32
        scale_shape = [c]
        mean_shape = [n * c]
        np.random.seed()
        self.x_np = np.random.random_sample(x_shape).astype(dtype)
        self.scale_np = np.ones(scale_shape).astype(dtype)
        self.bias_np = np.zeros(scale_shape).astype(dtype)
        self.mean_np, self.var_np = _cal_mean_variance(
            self.x_np, self.epsilon, mean_shape
        )
        self.fw_comp_atol = 1e-5

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Scale', 'Bias'],
            'Y',
            check_pir=True,
            check_prim_pir=True,
        )


class TestInstanceNormCaseNC(TestInstanceNormOp):
    def init_test_case(self):
        x_shape = [2, 100]
        n, c = x_shape[0], x_shape[1]
        self.epsilon = 1e-05
        dtype = np.float32
        scale_shape = [c]
        mean_shape = [n * c]
        np.random.seed()
        self.x_np = np.random.random_sample(x_shape).astype(dtype)
        self.scale_np = np.ones(scale_shape).astype(dtype)
        self.bias_np = np.zeros(scale_shape).astype(dtype)
        self.mean_np, self.var_np = _cal_mean_variance(
            self.x_np, self.epsilon, mean_shape
        )
        self.fw_comp_atol = 2e-5

    def test_check_output(self):
        self.check_output(atol=2e-5, check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X', 'Scale', 'Bias'],
            'Y',
            check_pir=True,
            check_prim_pir=True,
        )


class TestElasticNormOp(unittest.TestCase):
    def init_test_case(self):
        self.epsilon = 1e-5
        self.places = []
        if os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower() in [
            '1',
            'true',
            'on',
        ] or not (
            core.is_compiled_with_cuda()
            and core.op_support_gpu("instance_norm")
        ):
            self.places.append(core.CPUPlace())
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            self.places.append(core.CUDAPlace(0))

    def test_norm(self):
        self.init_test_case()
        inputs = np.random.random((2, 3, 5, 5)).astype(np.float32)
        shape = inputs.shape
        n, c, h, w = shape[0], shape[1], shape[2], shape[3]
        scale_shape = [c]
        mean_shape = [n * c]
        scale = np.ones(scale_shape).astype(np.float32)
        bias = np.zeros(scale_shape).astype(np.float32)
        mean, variance = _cal_mean_variance(inputs, self.epsilon, mean_shape)
        out_np, _, _ = _reference_instance_norm_naive(
            inputs, scale, bias, self.epsilon, mean, variance
        )

        for place in self.places:
            with base.dygraph.guard(place):
                instance_norm = paddle.nn.InstanceNorm2D(
                    5, weight_attr=False, bias_attr=False
                )
                outputs = instance_norm(paddle.to_tensor(inputs))
                np.testing.assert_allclose(
                    outputs.numpy(), out_np, rtol=1e-05, atol=1e-06
                )


class TestElasticNormOpCase2(unittest.TestCase):
    def init_test_case(self):
        self.epsilon = 1e-5
        self.places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu(
            "instance_norm"
        ):
            self.places.append(core.CUDAPlace(0))

    def test_norm(self):
        self.init_test_case()
        inputs = np.random.random((2, 3, 5, 5)).astype(np.float32)
        shape = inputs.shape
        n, c, h, w = shape[0], shape[1], shape[2], shape[3]
        scale_shape = [c]
        mean_shape = [n * c]
        scale = np.ones(scale_shape).astype(np.float32)
        bias = np.zeros(scale_shape).astype(np.float32)
        mean, variance = _cal_mean_variance(inputs, self.epsilon, mean_shape)
        out_np, _, _ = _reference_instance_norm_naive(
            inputs, scale, bias, self.epsilon, mean, variance
        )

        for place in self.places:
            with base.dygraph.guard(place):
                instance_norm = paddle.nn.InstanceNorm2D(
                    3, weight_attr=True, bias_attr=True
                )
                outputs = instance_norm(paddle.to_tensor(inputs))
                np.testing.assert_allclose(
                    outputs.numpy(), out_np, rtol=1e-05, atol=1e-06
                )


if __name__ == '__main__':
    unittest.main()
