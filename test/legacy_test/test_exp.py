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


class TestExpOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.rand(3, 4).astype(np.float32)
        self.test_types = ["decorator", "out", "out_decorator"]

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        if test_type == 'raw':
            result = paddle.exp(x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'decorator':
            result = paddle.exp(input=x)
            result.mean().backward()
            return result, x.grad
        elif test_type == 'out':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.exp(x, out=out)
            out.mean().backward()
            return out, x.grad
        elif test_type == 'out_decorator':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.exp(input=x, out=out)
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


class TestExpSleefVectorized(unittest.TestCase):
    """Test exp with shapes that exercise Sleef vectorized paths.

    For AVX2:
    - float32: VEC_SIZE = 8, so shapes >= 8 trigger vectorized path
    - float64: VEC_SIZE = 4, so shapes >= 4 trigger vectorized path
      (Covers vexp_avx2_f64 in sleef_vectorized_math.h lines 335-348)

    Test both:
    1. Shapes that are exact multiples of VEC_SIZE (only vectorized loop)
    2. Shapes with remainder (vectorized loop + scalar tail)

    Note: If MKL is available at runtime, the mkl_exp path (lines 704-706, 722-724)
    will be triggered instead. Both paths are tested through this test.
    """

    def setUp(self):
        paddle.disable_static()

    def test_exp_float32_vectorized_exact(self):
        """Test float32 exp with shape that's exact multiple of 8.
        Covers vexp_avx2_f32 main loop (lines 323-327).
        """
        # Shape 16 = 8 * 2, exercises only vectorized loop
        x_np = np.random.uniform(-2, 2, size=(16,)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_exp_float32_vectorized_with_tail(self):
        """Test float32 exp with shape that has remainder when divided by 8.
        Covers vexp_avx2_f32 both main loop (323-327) and scalar tail (329-331).
        """
        # Shape 13 = 8 + 5, exercises both vectorized loop and scalar tail
        x_np = np.random.uniform(-2, 2, size=(13,)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_exp_float64_vectorized_exact(self):
        """Test float64 exp with shape that's exact multiple of 4.
        Covers vexp_avx2_f64 main loop (lines 339-343).
        This specifically tests the code at sleef_vectorized_math.h L335-348.
        """
        # Shape 12 = 4 * 3, exercises only vectorized loop
        x_np = np.random.uniform(-2, 2, size=(12,)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    def test_exp_float64_vectorized_with_tail(self):
        """Test float64 exp with shape that has remainder when divided by 4.
        Covers vexp_avx2_f64 both main loop (339-343) and scalar tail (345-347).
        This specifically tests the code at sleef_vectorized_math.h L335-348.
        """
        # Shape 11 = 4 * 2 + 3, exercises both vectorized loop and scalar tail
        x_np = np.random.uniform(-2, 2, size=(11,)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    def test_exp_float32_large_shape(self):
        """Test float32 exp with large shape for comprehensive coverage.
        Tests MKL VML path (mkl_exp) if available, otherwise Sleef path.
        """
        x_np = np.random.uniform(-2, 2, size=(1024,)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_exp_float64_large_shape(self):
        """Test float64 exp with large shape for comprehensive coverage.
        Tests MKL VML path (mkl_exp) if available, otherwise Sleef path.
        This specifically tests vexp_avx2_f64 at sleef_vectorized_math.h L335-348.
        """
        x_np = np.random.uniform(-2, 2, size=(1024,)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    def test_exp_float32_2d_shape(self):
        """Test float32 exp with 2D shape to verify flattened processing."""
        # Shape (4, 5) = 20 elements, exercises vectorized path
        x_np = np.random.uniform(-2, 2, size=(4, 5)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_exp_float64_2d_shape(self):
        """Test float64 exp with 2D shape to verify flattened processing.
        Covers vexp_avx2_f64 at sleef_vectorized_math.h L335-348.
        """
        # Shape (3, 5) = 15 elements, exercises vectorized path with tail
        x_np = np.random.uniform(-2, 2, size=(3, 5)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    def test_exp_float32_small_shape_fallback(self):
        """Test float32 exp with small shape (numel < 8) to cover Eigen fallback path."""
        # Shape 5 < 8, triggers Eigen fallback instead of SIMD
        x_np = np.random.uniform(-2, 2, size=(5,)).astype(np.float32)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-5, atol=1e-5
        )

    def test_exp_float64_small_shape_fallback(self):
        """Test float64 exp with small shape (numel < 8) to cover Eigen fallback path."""
        # Shape 3 < 8, triggers Eigen fallback instead of SIMD
        x_np = np.random.uniform(-2, 2, size=(3,)).astype(np.float64)
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )

    def test_exp_float64_boundary_values(self):
        """Test float64 exp with boundary values to ensure numerical stability.
        Covers vexp_avx2_f64 at sleef_vectorized_math.h L335-348.
        """
        # Test values near boundaries
        x_np = np.array(
            [0.0, 1.0, -1.0, 10.0, -10.0, 50.0, -50.0, 100.0], dtype=np.float64
        )
        x = paddle.to_tensor(x_np, place=paddle.CPUPlace())
        result = paddle.exp(x)
        expected = np.exp(x_np)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1e-10, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
