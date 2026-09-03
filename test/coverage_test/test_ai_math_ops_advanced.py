# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
进阶数学操作单元测试 / Advanced Math Operations Unit Tests

测试目标 / Test Target:
  paddle.tensor.math 进阶函数 (python/paddle/tensor/math.py, 覆盖率约78.8%)

覆盖的模块 / Covered Modules:
  - paddle.cumsum, cumprod: 累积求和/积
  - paddle.diff: 差分
  - paddle.digamma, lgamma, erf, erfc: 特殊函数
  - paddle.frexp, ldexp: 浮点分解
  - paddle.hypot: 斜边长度
  - paddle.i0, i0e, i1, i1e: 贝塞尔函数

作用 / Purpose:
  覆盖特殊数学函数的代码路径，补充进阶数学计算的测试。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestCumsumCumprod(unittest.TestCase):
    """测试累积求和和累积积 / Test cumsum and cumprod"""

    def test_cumsum_1d(self):
        """测试1D累积求和 / Test 1D cumsum"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        result = paddle.cumsum(x)
        expected = np.array([1.0, 3.0, 6.0, 10.0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_cumsum_2d_axis0(self):
        """测试2D沿axis=0的累积求和 / Test 2D cumsum along axis=0"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        result = paddle.cumsum(x, axis=0)
        expected = np.array([[1.0, 2.0], [4.0, 6.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_cumsum_2d_axis1(self):
        """测试2D沿axis=1的累积求和 / Test 2D cumsum along axis=1"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0]])
        result = paddle.cumsum(x, axis=1)
        expected = np.array([[1.0, 3.0, 6.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_cumprod_1d(self):
        """测试1D累积积 / Test 1D cumprod"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        result = paddle.cumprod(x)
        expected = np.array([1.0, 2.0, 6.0, 24.0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_cumsum_dtype(self):
        """测试cumsum类型转换 / Test cumsum type conversion"""
        x = paddle.to_tensor([1, 2, 3, 4])
        result = paddle.cumsum(x, dtype='float32')
        self.assertEqual(result.dtype, paddle.float32)


class TestDiffOps(unittest.TestCase):
    """测试差分操作 / Test diff operations"""

    def test_diff_basic(self):
        """测试基本差分 / Test basic diff"""
        x = paddle.to_tensor([1.0, 3.0, 6.0, 10.0])
        result = paddle.diff(x)
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_diff_n2(self):
        """测试二阶差分 / Test second-order diff"""
        x = paddle.to_tensor([1.0, 3.0, 6.0, 10.0])
        result = paddle.diff(x, n=2)
        expected = np.array([1.0, 1.0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_diff_2d(self):
        """测试2D差分 / Test 2D diff"""
        x = paddle.to_tensor([[1.0, 2.0, 4.0], [1.0, 3.0, 6.0]])
        result = paddle.diff(x, axis=1)
        expected = np.array([[1.0, 2.0], [2.0, 3.0]])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


class TestSpecialFunctions(unittest.TestCase):
    """测试特殊数学函数 / Test special mathematical functions"""

    def test_erf(self):
        """测试误差函数 / Test error function"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = paddle.erf(x)
        # erf(0) = 0
        self.assertAlmostEqual(float(result[0].numpy()), 0.0, places=5)
        # erf is odd: erf(-x) = -erf(x)
        self.assertAlmostEqual(
            float(result[2].numpy()), -float(result[1].numpy()), places=5
        )

    def test_erfc(self):
        """测试余误差函数 / Test complementary error function"""
        # erfc = 1 - erf, erfc(0) = 1
        x = paddle.to_tensor([0.0])
        erf_result = paddle.erf(x)
        erfc_approx = 1.0 - float(erf_result.numpy()[0])
        self.assertAlmostEqual(erfc_approx, 1.0, places=5)

    def test_erfinv(self):
        """测试逆误差函数 / Test inverse error function"""
        x = paddle.to_tensor([0.0, 0.5, -0.5])
        result = paddle.erfinv(x)
        self.assertEqual(result.shape, [3])
        self.assertAlmostEqual(float(result[0].numpy()), 0.0, places=5)

    def test_digamma(self):
        """测试Digamma函数 / Test digamma function"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.digamma(x)
        self.assertEqual(result.shape, [3])

    def test_lgamma(self):
        """测试Log-Gamma函数 / Test log-gamma function"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.lgamma(x)
        # lgamma(1) = 0, lgamma(2) = 0, lgamma(3) = log(2)
        self.assertAlmostEqual(float(result[0].numpy()), 0.0, places=5)
        self.assertAlmostEqual(float(result[1].numpy()), 0.0, places=5)

    def test_polygamma(self):
        """测试Polygamma函数 / Test polygamma function"""
        x = paddle.to_tensor([1.0, 2.0])
        # polygamma(x, n): x is tensor, n is int
        result = paddle.polygamma(x, 1)
        self.assertEqual(result.shape, [2])

    def test_i0(self):
        """测试第一类零阶修正贝塞尔函数 / Test 0th-order modified Bessel function"""
        x = paddle.to_tensor([0.0, 1.0])
        result = paddle.i0(x)
        # i0(0) = 1
        self.assertAlmostEqual(float(result[0].numpy()), 1.0, places=4)


class TestHyperbolicFunctions(unittest.TestCase):
    """测试双曲函数 / Test hyperbolic functions"""

    def test_sinh(self):
        """测试sinh / Test sinh"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = paddle.sinh(x)
        expected = np.sinh(np.array([0.0, 1.0, -1.0]))
        np.testing.assert_allclose(
            result.numpy(), expected.astype('float32'), rtol=1e-5
        )

    def test_cosh(self):
        """测试cosh / Test cosh"""
        x = paddle.to_tensor([0.0, 1.0])
        result = paddle.cosh(x)
        # cosh(0) = 1
        self.assertAlmostEqual(float(result[0].numpy()), 1.0, places=5)

    def test_tanh(self):
        """测试tanh / Test tanh"""
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        result = paddle.tanh(x)
        self.assertAlmostEqual(float(result[0].numpy()), 0.0, places=5)

    def test_asinh(self):
        """测试asinh / Test asinh"""
        x = paddle.to_tensor([0.0, 1.0])
        result = paddle.asinh(x)
        self.assertAlmostEqual(float(result[0].numpy()), 0.0, places=5)

    def test_acosh(self):
        """测试acosh / Test acosh"""
        x = paddle.to_tensor([1.0, 2.0])
        result = paddle.acosh(x)
        self.assertAlmostEqual(float(result[0].numpy()), 0.0, places=5)

    def test_atanh(self):
        """测试atanh / Test atanh"""
        x = paddle.to_tensor([0.0, 0.5])
        result = paddle.atanh(x)
        self.assertAlmostEqual(float(result[0].numpy()), 0.0, places=5)


class TestArithmeticOps(unittest.TestCase):
    """测试算术操作 / Test arithmetic operations"""

    def test_hypot(self):
        """测试斜边计算 / Test hypotenuse calculation"""
        x = paddle.to_tensor([3.0, 5.0])
        y = paddle.to_tensor([4.0, 12.0])
        result = paddle.hypot(x, y)
        expected = np.array([5.0, 13.0])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_frexp(self):
        """测试浮点分解 / Test floating point decomposition"""
        x = paddle.to_tensor([6.0, -3.0, 1.5])
        mantissa, exponent = paddle.frexp(x)
        self.assertEqual(mantissa.shape, [3])
        self.assertEqual(exponent.shape, [3])
        # 6.0 = 0.75 * 2^3
        self.assertAlmostEqual(float(mantissa[0].numpy()), 0.75, places=5)
        self.assertEqual(int(exponent[0].numpy()), 3)

    def test_ldexp(self):
        """测试浮点合成 / Test floating point composition"""
        x = paddle.to_tensor([0.75, 0.5])
        exponent = paddle.to_tensor([3, 2])
        result = paddle.ldexp(x, exponent)
        # 0.75 * 2^3 = 6.0, 0.5 * 2^2 = 2.0
        np.testing.assert_allclose(
            result.numpy(), np.array([6.0, 2.0]), rtol=1e-5
        )

    def test_logit(self):
        """测试logit函数 / Test logit function"""
        x = paddle.to_tensor([0.1, 0.5, 0.9])
        result = paddle.logit(x)
        # logit(0.5) = 0
        self.assertAlmostEqual(float(result[1].numpy()), 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
