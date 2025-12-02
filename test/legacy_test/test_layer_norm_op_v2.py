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
from utils import static_guard

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


class TestLayerNormParamDygraph(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.normalized_shape = [6]
        self.x_shape = [2, 4, 4, 6]
        self.places = get_places()

    def _run_test_on_places(self, test_func):
        """Helper to run the test function on all places."""
        for p in self.places:
            with base.dygraph.guard(p):
                test_func(p)

    def test_elementwise_affine_false(self):
        """test that when elementwise_affine=False, weight and bias parameters are not created."""

        def run_test(p):
            layer = paddle.nn.LayerNorm(
                normalized_shape=self.normalized_shape, elementwise_affine=False
            )
            assert layer.weight is None
            assert layer.bias is None

            x_tensor = paddle.randn(self.x_shape)
            out = layer(x_tensor)
            assert out.shape == self.x_shape

        self._run_test_on_places(run_test)

    def test_elementwise_affine_true(self):
        """test that when elementwise_affine=True and attr=None, parameters are created with default initialization."""

        def run_test(p):
            layer = paddle.nn.LayerNorm(
                normalized_shape=self.normalized_shape,
                elementwise_affine=True,
            )
            assert layer.weight is not None
            assert layer.bias is not None

            expected_weight = paddle.ones(self.normalized_shape)
            expected_bias = paddle.zeros(self.normalized_shape)

            np.testing.assert_allclose(
                layer.weight.numpy(), expected_weight.numpy()
            )
            np.testing.assert_allclose(
                layer.bias.numpy(), expected_bias.numpy()
            )

        self._run_test_on_places(run_test)

    def test_bias_false(self):
        """test that when bias=False, the bias parameter is disabled even if elementwise_affine=True."""

        def run_test(p):
            layer = paddle.nn.LayerNorm(
                normalized_shape=self.normalized_shape,
                elementwise_affine=True,
                bias=False,
            )
            assert layer.weight is not None
            assert layer.bias is None

        self._run_test_on_places(run_test)

    def test_weight_and_bias_false(self):
        """test that when weight_attr=False and bias_attr=False, both parameters are disabled."""

        def run_test(p):
            layer = paddle.nn.LayerNorm(
                normalized_shape=self.normalized_shape,
                elementwise_affine=True,
                weight_attr=False,
                bias_attr=False,
            )
            assert layer.weight is None
            assert layer.bias is None

        self._run_test_on_places(run_test)

    def test_alias(self):
        """test parameter alias epsilon/eps"""

        def run_test(p):
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

            np.testing.assert_array_equal(out_epsilon.numpy(), out_eps.numpy())

        self._run_test_on_places(run_test)

    def test_errors(self):
        """test for errors."""

        def run_test(p):
            with self.assertRaises(ValueError):
                layer_norm = paddle.nn.LayerNorm(self.normalized_shape)
                x1 = np.random.random([3, *self.normalized_shape]).astype(
                    'float32'
                )
                layer_norm(x1)

            with self.assertRaises(TypeError):
                paddle.nn.LayerNorm(
                    self.normalized_shape, 1e-5, None, None, "name"
                )

            with self.assertRaises(TypeError):
                paddle.nn.LayerNorm(
                    self.normalized_shape, 1e-5, False, "cpu", paddle.float32
                )

        self._run_test_on_places(run_test)


class TestLayerNormParamStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.normalized_shape = [6]
        self.x_shape = [2, 4, 4, 6]
        self.places = get_places()

    def test_static_elementwise_affine_false(self):
        """test elementwise_affine=False in static graph mode."""
        for p in self.places:
            with static_guard():
                main = base.Program()
                start = base.Program()
                with (
                    base.unique_name.guard(),
                    base.program_guard(main, start),
                ):
                    layer = paddle.nn.LayerNorm(
                        normalized_shape=self.normalized_shape,
                        elementwise_affine=False,
                    )
                    x = paddle.static.data(
                        name='x', shape=self.x_shape, dtype='float32'
                    )
                    out = layer(x)

                exe = base.Executor(p)
                exe.run(start)
                input_np = np.random.randn(*self.x_shape).astype('float32')
                result = exe.run(main, feed={'x': input_np}, fetch_list=[out])[
                    0
                ]

                assert result.shape == tuple(self.x_shape)

    def test_static_elementwise_affine_true(self):
        """test elementwise_affine=True with default init in static graph mode."""
        for p in self.places:
            with static_guard():
                main = base.Program()
                start = base.Program()
                with (
                    base.unique_name.guard(),
                    base.program_guard(main, start),
                ):
                    layer = paddle.nn.LayerNorm(
                        normalized_shape=self.normalized_shape,
                        elementwise_affine=True,
                    )

                exe = base.Executor(p)
                exe.run(start)
                weight_np, bias_np = exe.run(
                    main, fetch_list=[layer.weight, layer.bias]
                )

                assert weight_np is not None
                assert bias_np is not None

                expected_weight = np.ones(self.normalized_shape)
                expected_bias = np.zeros(self.normalized_shape)

                np.testing.assert_allclose(weight_np, expected_weight)
                np.testing.assert_allclose(bias_np, expected_bias)

    def test_static_bias_false(self):
        """test bias=False in static graph mode."""
        for p in self.places:
            with static_guard():
                main = base.Program()
                start = base.Program()
                with (
                    base.unique_name.guard(),
                    base.program_guard(main, start),
                ):
                    layer = paddle.nn.LayerNorm(
                        normalized_shape=self.normalized_shape,
                        elementwise_affine=True,
                        bias=False,
                    )
                    assert layer.bias is None

                exe = base.Executor(p)
                exe.run(start)
                weight_np = exe.run(main, fetch_list=[layer.weight])[0]
                assert weight_np is not None
                assert weight_np.shape == tuple(self.normalized_shape)

    def test_static_weight_and_bias_false(self):
        """test weight_attr=False and bias_attr=False in static graph mode."""
        for p in self.places:
            with static_guard():
                main = base.Program()
                start = base.Program()
                with (
                    base.unique_name.guard(),
                    base.program_guard(main, start),
                ):
                    layer = paddle.nn.LayerNorm(
                        normalized_shape=self.normalized_shape,
                        elementwise_affine=True,
                        weight_attr=False,
                        bias_attr=False,
                    )
                    assert layer.weight is None
                    assert layer.bias is None

    def test_static_alias(self):
        """test parameter alias epsilon/eps in static graph mode."""
        for p in self.places:
            with static_guard():
                main = base.Program()
                start = base.Program()
                with (
                    base.unique_name.guard(),
                    base.program_guard(main, start),
                ):
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

                    x = paddle.static.data(
                        name='x', shape=self.x_shape, dtype='float32'
                    )
                    out_epsilon = layer_epsilon(x)
                    out_eps = layer_eps(x)

                exe = base.Executor(p)
                exe.run(start)
                input_np = np.random.randn(*self.x_shape).astype('float32')
                out_eps_val, out_epsilon_val = exe.run(
                    main,
                    feed={'x': input_np},
                    fetch_list=[out_eps, out_epsilon],
                )

                np.testing.assert_array_equal(out_epsilon_val, out_eps_val)

    def test_static_errors(self):
        """test errors in static graph mode."""
        for p in self.places:
            with static_guard():
                main = base.Program()
                start = base.Program()
                with (
                    base.unique_name.guard(),
                    base.program_guard(main, start),
                ):
                    with self.assertRaises(TypeError):
                        paddle.nn.LayerNorm(
                            self.normalized_shape, 1e-5, None, None, "name"
                        )

                    with self.assertRaises(TypeError):
                        paddle.nn.LayerNorm(
                            self.normalized_shape,
                            1e-5,
                            False,
                            "cpu",
                            paddle.float32,
                        )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
