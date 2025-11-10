# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_places

import paddle
from paddle import nn


def _reference_layer_norm_naive(x, scale, beta, epsilon, begin_norm_axis=1):
    x_shape = x.shape
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis : len(x_shape)], 1)
    x.shape = [N, D]

    mean = np.mean(x, axis=1)
    var = np.var(x, axis=1) + epsilon
    output = np.divide(
        (x - mean.reshape([N, 1])), (np.sqrt(var)).reshape([N, 1])
    )
    if scale is not None:
        output = scale.reshape([1, D]) * output
    if beta is not None:
        output = output + beta.reshape([1, D])

    x.shape, output.shape = x_shape, x_shape
    return output


class TestLayerNormOp(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_shape = [2, 6, 6, 3]
        self.epsilon = 1e-5
        self.begin_norm_axis = 1
        self.places = get_places()

    def test_basic_fp32(self):
        """test basic functionality with float32."""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                x_np = np.random.random(self.x_shape).astype('float32')
                scale_np = np.random.random(
                    self.x_shape[self.begin_norm_axis :]
                ).astype('float32')
                bias_np = np.random.random(
                    self.x_shape[self.begin_norm_axis :]
                ).astype('float32')
                scale = paddle.to_tensor(scale_np).reshape(-1)
                bias = paddle.to_tensor(bias_np).reshape(-1)

                ln = nn.LayerNorm(
                    normalized_shape=self.x_shape[self.begin_norm_axis :],
                    weight_attr=nn.initializer.Assign(scale),
                    bias_attr=nn.initializer.Assign(bias),
                    epsilon=self.epsilon,
                )

                x_pd = paddle.to_tensor(x_np)
                y_pd = ln(x_pd)
                expect_res = _reference_layer_norm_naive(
                    x_np, scale_np, bias_np, self.epsilon, self.begin_norm_axis
                )

                np.testing.assert_allclose(
                    y_pd.numpy(), expect_res, rtol=1e-5, atol=1e-4
                )

    def test_no_scale_no_bias_fp32(self):
        """test the case when both scale and bias are disabled (FP32)."""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                x_np = np.random.random(self.x_shape).astype('float32')
                x_pd = paddle.to_tensor(x_np)

                ln = nn.LayerNorm(
                    normalized_shape=self.x_shape[self.begin_norm_axis :],
                    elementwise_affine=False,
                    epsilon=self.epsilon,
                )
                y_pd = ln(x_pd)

                expect_res = _reference_layer_norm_naive(
                    x_np, None, None, self.epsilon, self.begin_norm_axis
                )
                np.testing.assert_allclose(
                    y_pd.numpy(), expect_res, rtol=1e-5, atol=1e-4
                )

    def test_with_scale_no_bias_fp32(self):
        """test the case when only scale is enabled (FP32)."""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                x_np = np.random.random(self.x_shape).astype('float32')
                scale_np = np.random.random(
                    self.x_shape[self.begin_norm_axis :]
                ).astype('float32')
                scale = paddle.to_tensor(scale_np).reshape(-1)

                ln = nn.LayerNorm(
                    normalized_shape=self.x_shape[self.begin_norm_axis :],
                    elementwise_affine=True,
                    bias_attr=False,
                    epsilon=self.epsilon,
                )
                with paddle.no_grad():
                    ln.weight.set_value(scale)

                x_pd = paddle.to_tensor(x_np)
                y_pd = ln(x_pd)

                expect_res = _reference_layer_norm_naive(
                    x_np, scale_np, None, self.epsilon, self.begin_norm_axis
                )
                np.testing.assert_allclose(
                    y_pd.numpy(), expect_res, rtol=1e-5, atol=1e-4
                )

    def test_no_scale_with_bias_fp32(self):
        """test the case when only bias is enabled (FP32)."""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                x_np = np.random.random(self.x_shape).astype('float32')
                bias_np = np.random.random(
                    self.x_shape[self.begin_norm_axis :]
                ).astype('float32')
                bias = paddle.to_tensor(bias_np).reshape(-1)

                ln = nn.LayerNorm(
                    normalized_shape=self.x_shape[self.begin_norm_axis :],
                    elementwise_affine=True,
                    weight_attr=False,
                    epsilon=self.epsilon,
                )
                with paddle.no_grad():
                    ln.bias.set_value(bias)

                x_pd = paddle.to_tensor(x_np)
                y_pd = ln(x_pd)

                expect_res = _reference_layer_norm_naive(
                    x_np, None, bias_np, self.epsilon, self.begin_norm_axis
                )
                np.testing.assert_allclose(
                    y_pd.numpy(), expect_res, rtol=1e-5, atol=1e-4
                )

    def test_bf16_forward_backward(self):
        """test forward and backward pass with bfloat16 precision."""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                x_np = np.random.random(self.x_shape).astype('float32')
                scale_np = np.random.random(
                    self.x_shape[self.begin_norm_axis :]
                ).astype('float32')
                bias_np = np.random.random(
                    self.x_shape[self.begin_norm_axis :]
                ).astype('float32')

                x = paddle.to_tensor(x_np).cast(paddle.bfloat16)
                x.stop_gradient = False

                scale = (
                    paddle.to_tensor(scale_np).cast(paddle.bfloat16).reshape(-1)
                )
                bias = (
                    paddle.to_tensor(bias_np).cast(paddle.bfloat16).reshape(-1)
                )

                ln = nn.LayerNorm(
                    normalized_shape=self.x_shape[self.begin_norm_axis :],
                    weight_attr=nn.initializer.Assign(scale),
                    bias_attr=nn.initializer.Assign(bias),
                    epsilon=self.epsilon,
                )

                y = ln(x)
                loss = y.sum()
                loss.backward()

                self.assertIsNotNone(x.grad)
                self.assertIsNotNone(ln.weight.grad)
                self.assertIsNotNone(ln.bias.grad)


class TestLayerNormParam(unittest.TestCase):
    def setUp(self):
        self.normalized_shape = [6]
        self.x_tensor = paddle.randn([2, 4, 4, 6])
        self.places = get_places()

    def test_elementwise_affine_false(self):
        """test that when elementwise_affine=False, no learnable parameters are created."""
        layer = nn.LayerNorm(
            normalized_shape=self.normalized_shape, elementwise_affine=False
        )
        self.assertIsNone(layer.weight)
        self.assertIsNone(layer.bias)

        out = layer(self.x_tensor)
        self.assertEqual(out.shape, self.x_tensor.shape)

    def test_elementwise_affine_true(self):
        """test that when elementwise_affine=True and attr=None, parameters are created with default initialization."""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                layer = nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                )
                self.assertIsNotNone(layer.weight)
                self.assertIsNotNone(layer.bias)

                expected_weight = paddle.ones([6])
                expected_bias = paddle.zeros([6])
                self.assertTrue(paddle.allclose(layer.weight, expected_weight))
                self.assertTrue(paddle.allclose(layer.bias, expected_bias))

    def test_bias_false(self):
        """test that when bias=False, the bias parameter is disabled even if elementwise_affine=True."""
        layer = nn.LayerNorm(
            normalized_shape=self.normalized_shape,
            elementwise_affine=True,
            bias=False,
        )
        self.assertIsNotNone(layer.weight)
        self.assertIsNone(layer.bias)

    def test_attr_custom_initialization(self):
        """test that weight_attr and bias_attr can be used to customize the initialization of the weight parameter."""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                weight_attr = paddle.nn.initializer.Constant(value=2.0)
                bias_attr = paddle.nn.initializer.Constant(value=3.0)
                layer = nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr,
                )

                expected_weight = paddle.full([6], 2.0)
                expected_bias = paddle.full([6], 3.0)
                self.assertTrue(paddle.allclose(layer.weight, expected_weight))
                self.assertTrue(paddle.allclose(layer.bias, expected_bias))

    def test_alias(self):
        """test parameter alias epsilon/eps"""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                layer_epsilon = nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                    epsilon=1e-5,
                )
                layer_eps = nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                    eps=1e-5,
                )

                out_epsilon = layer_epsilon(self.x_tensor)
                out_eps = layer_eps(self.x_tensor)

                np.testing.assert_array_equal(
                    out_epsilon.numpy(), out_eps.numpy()
                )

    def test_errors(self):
        """test for errors."""
        layer_norm = nn.LayerNorm(self.normalized_shape)
        x1 = np.random.random([3, *self.normalized_shape]).astype('float32')
        with self.assertRaises(ValueError):
            layer_norm(x1)
        with self.assertRaises(TypeError):
            nn.LayerNorm(self.normalized_shape, 1e-5, None, None, "name")
        with self.assertRaises(TypeError):
            nn.LayerNorm(
                self.normalized_shape, 1e-5, False, "cpu", paddle.float32
            )


if __name__ == '__main__':
    unittest.main()
