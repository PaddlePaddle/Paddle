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

import numpy as np

import paddle


class TestSinOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.rand(3, 4).astype(np.float32)
        self.test_types = ["decorator", "out", "out_decorator"]

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        if test_type == 'raw':
            result = paddle.sin(x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'decorator':
            result = paddle.sin(input=x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'out':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.sin(x, out=out)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'out_decorator':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.sin(input=x, out=out)
            out.mean().backward()
            return out, x.grad
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def test_all(self):
        out_std, grad_std = self.do_test('raw')
        for test_type in self.test_types:
            out, grad = self.do_test(test_type)
            np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-7)
            np.testing.assert_allclose(
                grad.numpy(), grad_std.numpy(), rtol=1e-7
            )


class TestSinSleefVectorized(unittest.TestCase):
    """Test sin with shapes that exercise Sleef vectorized paths.

    For AVX2:
    - float32: VEC_SIZE = 8, so shapes >= 8 trigger vectorized path
    - float64: VEC_SIZE = 4, so shapes >= 4 trigger vectorized path

    Test both:
    1. Shapes that are exact multiples of VEC_SIZE (only vectorized loop)
    2. Shapes with remainder (vectorized loop + scalar tail)
    """

    def setUp(self):
        paddle.disable_static()

    def test_sin_float32_vectorized_exact(self):
        """Test float32 sin with shape that's exact multiple of 8.
        Covers vsin_avx2_f32 main loop (lines 79-83).
        """
        # Shape 16 = 8 * 2, exercises only vectorized loop
        x_np = np.random.uniform(-np.pi, np.pi, size=(16,)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_sin_float32_vectorized_with_tail(self):
        """Test float32 sin with shape that has remainder when divided by 8.
        Covers vsin_avx2_f32 both main loop (79-83) and scalar tail (86-88).
        """
        # Shape 13 = 8 + 5, exercises both vectorized loop and scalar tail
        x_np = np.random.uniform(-np.pi, np.pi, size=(13,)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_sin_float64_vectorized_exact(self):
        """Test float64 sin with shape that's exact multiple of 4.
        Covers vsin_avx2_f64 main loop (lines 112-116).
        """
        # Shape 12 = 4 * 3, exercises only vectorized loop
        x_np = np.random.uniform(-np.pi, np.pi, size=(12,)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    def test_sin_float64_vectorized_with_tail(self):
        """Test float64 sin with shape that has remainder when divided by 4.
        Covers vsin_avx2_f64 both main loop (112-116) and scalar tail (118-120).
        """
        # Shape 11 = 4 * 2 + 3, exercises both vectorized loop and scalar tail
        x_np = np.random.uniform(-np.pi, np.pi, size=(11,)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    def test_sin_float32_large_shape(self):
        """Test float32 sin with large shape for comprehensive coverage."""
        x_np = np.random.uniform(-np.pi, np.pi, size=(1024,)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_sin_float64_large_shape(self):
        """Test float64 sin with large shape for comprehensive coverage."""
        x_np = np.random.uniform(-np.pi, np.pi, size=(1024,)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    def test_sin_float32_2d_shape(self):
        """Test float32 sin with 2D shape to verify flattened processing."""
        # Shape (4, 5) = 20 elements, exercises vectorized path
        x_np = np.random.uniform(-np.pi, np.pi, size=(4, 5)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_sin_float64_2d_shape(self):
        """Test float64 sin with 2D shape to verify flattened processing."""
        # Shape (3, 5) = 15 elements, exercises vectorized path with tail
        x_np = np.random.uniform(-np.pi, np.pi, size=(3, 5)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    def test_sin_float32_small_shape_fallback(self):
        """Test float32 sin with small shape (numel < 8) to cover Eigen fallback path.
        Covers VectorizedSinImpl fallback branch (lines 74-80 in activation_impl.h).
        """
        # Shape 5 < 8, triggers Eigen fallback instead of SIMD
        x_np = np.random.uniform(-np.pi, np.pi, size=(5,)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_sin_float64_small_shape_fallback(self):
        """Test float64 sin with small shape (numel < 8) to cover Eigen fallback path.
        Covers VectorizedSinImpl fallback branch (lines 74-80 in activation_impl.h).
        """
        # Shape 3 < 8, triggers Eigen fallback instead of SIMD
        x_np = np.random.uniform(-np.pi, np.pi, size=(3,)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.sin(x)
        expected = np.sin(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
