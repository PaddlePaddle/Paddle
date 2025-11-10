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

import unittest

import numpy as np
from op_test import get_places

import paddle
from paddle import base
from paddle.base import Program, program_guard


class TestDygraphLayerNormv2(unittest.TestCase):
    def test_dygraph(self):
        for p in get_places():
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    y = ln(paddle.to_tensor(x))
                return y.numpy()

            def compute_v2(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    y = ln(paddle.to_tensor(x))
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)

    def test_eager(self):
        for p in get_places():
            shape = [4, 10, 4, 4]

            def compute_v1(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    x1 = paddle.to_tensor(x)
                    x1.stop_gradient = False
                    y = ln(x1)
                    y.backward()
                    return y.numpy(), x1.gradient()

            def compute_v2(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    x1 = paddle.to_tensor(x)
                    x1.stop_gradient = False
                    y = ln(x1)
                    y.backward()
                    return y.numpy(), x1.gradient()

            x = np.random.randn(*shape).astype("float32")
            y1, g1 = compute_v1(x)
            y2, g2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)
            np.testing.assert_allclose(g1, g2, rtol=1e-05)

    def test_static(self):
        paddle.enable_static()
        for p in get_places():
            exe = base.Executor(p)
            shape = [4, 10, 16, 16]

            def compute_v1(x_np):
                with program_guard(Program(), Program()):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    x = paddle.static.data(
                        name='x', shape=x_np.shape, dtype=x_np.dtype
                    )
                    y = ln(x)
                    exe.run(base.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    x = paddle.static.data(
                        name='x', shape=x_np.shape, dtype=x_np.dtype
                    )
                    y = ln(x)
                    exe.run(base.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y1, y2, rtol=1e-05)


class TestLayerNormFunction(unittest.TestCase):
    def test_dygraph(self):
        for p in get_places():
            shape = [4, 10, 4, 4]

            def compute_v0(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[1:])
                    y = ln(paddle.to_tensor(x))
                return y.numpy()

            def compute_v1(x):
                with base.dygraph.guard(p):
                    x = paddle.to_tensor(x)
                    y = paddle.nn.functional.layer_norm(x, shape[1:])
                return y.numpy()

            def compute_v2(x):
                with base.dygraph.guard(p):
                    x = paddle.to_tensor(x)
                    y = paddle.nn.functional.layer_norm(x, tuple(shape[1:]))
                return y.numpy()

            def compute_v3(x):
                with base.dygraph.guard(p):
                    ln = paddle.nn.LayerNorm(shape[-1])
                    y = ln(paddle.to_tensor(x))
                return y.numpy()

            def compute_v4(x):
                with base.dygraph.guard(p):
                    x = paddle.to_tensor(x)
                    y = paddle.nn.functional.layer_norm(x, shape[-1])
                return y.numpy()

            x = np.random.randn(*shape).astype("float32")
            y0 = compute_v0(x)
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            np.testing.assert_allclose(y0, y1, rtol=1e-05)
            np.testing.assert_allclose(y0, y2, rtol=1e-05)
            y3 = compute_v3(x)
            y4 = compute_v4(x)
            np.testing.assert_allclose(y3, y4, rtol=1e-05)

            self.assertRaises(
                ValueError,
                paddle.nn.functional.layer_norm,
                x=x,
                normalized_shape=1.0,
            )


class TestLayerNormParam(unittest.TestCase):
    def setUp(self):
        self.normalized_shape = [6]
        self.x_shape = [2, 4, 4, 6]
        self.epsilon = 1e-5
        self.places = get_places()

    def test_elementwise_affine_false(self):
        """test that when elementwise_affine=False, weight and bias parameters are not created."""
        for p in self.places:
            with base.dygraph.guard(p):
                layer = paddle.nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=False,
                )
                self.assertIsNone(
                    layer.weight,
                    "Weight should be None when elementwise_affine=False",
                )
                self.assertIsNone(
                    layer.bias,
                    "Bias should be None when elementwise_affine=False",
                )

                x_tensor = paddle.randn(self.x_shape)
                out = layer(x_tensor)
                self.assertEqual(out.shape, x_tensor.shape)

    def test_elementwise_affine_true(self):
        """test that when elementwise_affine=True and attr=None, parameters are created with default initialization."""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                layer = paddle.nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                )
                self.assertIsNotNone(
                    layer.weight,
                    "Weight should not be None when elementwise_affine=True",
                )
                self.assertIsNotNone(
                    layer.bias,
                    "Weight should not be None when elementwise_affine=True",
                )

                expected_weight = paddle.ones(self.normalized_shape)
                expected_bias = paddle.zeros(self.normalized_shape)

                self.assertTrue(paddle.allclose(layer.weight, expected_weight))
                self.assertTrue(paddle.allclose(layer.bias, expected_bias))

    def test_bias_false(self):
        """test that when bias=False, the bias parameter is disabled even if elementwise_affine=True."""
        for p in self.places:
            with base.dygraph.guard(p):
                layer = paddle.nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                    bias=False,
                )
                self.assertIsNotNone(
                    layer.weight,
                    "Weight should exist when elementwise_affine=True",
                )
                self.assertIsNone(
                    layer.bias, "Bias should be None when bias_attr=False"
                )

    def test_weight_and_bias_false(self):
        """test that when weight_attr=False and bias_attr=False, both parameters are disabled."""
        for p in self.places:
            with base.dygraph.guard(p):
                layer = paddle.nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                    weight_attr=False,
                    bias_attr=False,
                )
                self.assertIsNotNone(
                    layer.weight,
                    "Weight should not be None when elementwise_affine=True although weight_attr=False",
                )
                self.assertIsNotNone(
                    layer.bias,
                    "Bias should not be None when elementwise_affine=True although bias_attr=False",
                )

    def test_custom_initialization(self):
        """test custom initialization using weight_attr and bias_attr."""
        for p in self.places:
            with base.dygraph.guard(p):
                weight_val = 2.5
                bias_val = -1.0
                weight_initializer = paddle.nn.initializer.Constant(
                    value=weight_val
                )
                bias_initializer = paddle.nn.initializer.Constant(
                    value=bias_val
                )

                layer = paddle.nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                    weight_attr=weight_initializer,
                    bias_attr=bias_initializer,
                )

                expected_weight = paddle.full(
                    self.normalized_shape, weight_val, dtype=layer.weight.dtype
                )
                expected_bias = paddle.full(
                    self.normalized_shape, bias_val, dtype=layer.bias.dtype
                )

                self.assertTrue(
                    paddle.allclose(layer.weight, expected_weight),
                    f"Weight initialization failed. Got {layer.weight.numpy()}, expected {expected_weight.numpy()}",
                )
                self.assertTrue(
                    paddle.allclose(layer.bias, expected_bias),
                    f"Bias initialization failed. Got {layer.bias.numpy()}, expected {expected_bias.numpy()}",
                )

    def test_alias(self):
        """test parameter alias epsilon/eps"""
        for place in self.places:
            with paddle.base.dygraph.guard(place):
                layer_epsilon = paddle.nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                    epsilon=1e-5,
                )
                layer_eps = paddle.nn.LayerNorm(
                    normalized_shape=self.normalized_shape,
                    elementwise_affine=True,
                    eps=1e-5,
                )

                x_tensor = paddle.randn(self.x_shape)
                out_epsilon = layer_epsilon(x_tensor)
                out_eps = layer_eps(x_tensor)

                np.testing.assert_array_equal(
                    out_epsilon.numpy(), out_eps.numpy()
                )

    def test_errors(self):
        """test for errors."""
        layer_norm = paddle.nn.LayerNorm(self.normalized_shape)
        x1 = np.random.random([3, *self.normalized_shape]).astype('float32')
        with self.assertRaises(TypeError):
            layer_norm(x1)
        with self.assertRaises(TypeError):
            paddle.nn.LayerNorm(self.normalized_shape, 1e-5, None, None, "name")
        with self.assertRaises(TypeError):
            paddle.nn.LayerNorm(
                self.normalized_shape, 1e-5, False, "cpu", paddle.float32
            )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
