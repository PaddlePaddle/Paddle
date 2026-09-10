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
复数张量操作单元测试 / Complex Tensor Operations Unit Tests

测试目标 / Test Target:
  paddle复数张量操作 (覆盖率较低)

覆盖的模块 / Covered Modules:
  - paddle.complex: 创建复数张量
  - paddle.real: 获取实部
  - paddle.imag: 获取虚部
  - paddle.angle: 获取相位角
  - paddle.conj: 共轭
  - paddle.as_complex: 转换为复数
  - paddle.as_real: 转换为实数

作用 / Purpose:
  覆盖复数张量操作的各种代码路径，提高复数运算的测试覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestComplexCreation(unittest.TestCase):
    """测试复数张量创建 / Test complex tensor creation"""

    def test_complex_basic(self):
        """测试基本复数张量创建 / Test basic complex tensor creation"""
        real = paddle.to_tensor([1.0, 2.0, 3.0])
        imag = paddle.to_tensor([4.0, 5.0, 6.0])
        z = paddle.complex(real, imag)
        self.assertEqual(z.dtype, paddle.complex64)
        self.assertEqual(z.shape, [3])

    def test_complex_from_numpy(self):
        """测试从numpy创建复数张量 / Test complex tensor from numpy"""
        np_z = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        z = paddle.to_tensor(np_z)
        self.assertEqual(z.dtype, paddle.complex64)

    def test_complex128(self):
        """测试complex128类型 / Test complex128 type"""
        real = paddle.to_tensor([1.0, 2.0], dtype='float64')
        imag = paddle.to_tensor([3.0, 4.0], dtype='float64')
        z = paddle.complex(real, imag)
        self.assertEqual(z.dtype, paddle.complex128)

    def test_complex_2d(self):
        """测试2D复数张量 / Test 2D complex tensor"""
        real = paddle.randn([3, 4])
        imag = paddle.randn([3, 4])
        z = paddle.complex(real, imag)
        self.assertEqual(z.shape, [3, 4])


class TestComplexDecomposition(unittest.TestCase):
    """测试复数分解操作 / Test complex decomposition"""

    def setUp(self):
        """初始化复数张量 / Initialize complex tensor"""
        self.real = paddle.to_tensor([1.0, 2.0, 3.0])
        self.imag = paddle.to_tensor([4.0, 5.0, 6.0])
        self.z = paddle.complex(self.real, self.imag)

    def test_real_part(self):
        """测试获取实部 / Test getting real part"""
        real = paddle.real(self.z)
        np.testing.assert_allclose(real.numpy(), self.real.numpy())

    def test_imag_part(self):
        """测试获取虚部 / Test getting imaginary part"""
        imag = paddle.imag(self.z)
        np.testing.assert_allclose(imag.numpy(), self.imag.numpy())

    def test_angle(self):
        """测试相位角 / Test angle (phase)"""
        # angle(1+1j) = pi/4
        z = paddle.to_tensor([1.0 + 1.0j])
        angle = paddle.angle(z)
        self.assertAlmostEqual(float(angle.numpy()), np.pi / 4, places=5)

    def test_abs_complex(self):
        """测试复数模 / Test complex absolute value"""
        # |3 + 4j| = 5
        z = paddle.to_tensor([3.0 + 4.0j])
        magnitude = paddle.abs(z)
        self.assertAlmostEqual(float(magnitude.numpy()), 5.0, places=4)

    def test_conj(self):
        """测试共轭 / Test conjugate"""
        conj = paddle.conj(self.z)
        # Real part same, imaginary part negated
        np.testing.assert_allclose(paddle.real(conj).numpy(), self.real.numpy())
        np.testing.assert_allclose(
            paddle.imag(conj).numpy(), -self.imag.numpy()
        )


class TestAsComplexReal(unittest.TestCase):
    """测试as_complex和as_real转换 / Test as_complex and as_real conversion"""

    def test_as_complex(self):
        """测试as_complex转换 / Test as_complex conversion"""
        x = paddle.randn([3, 2])  # Last dim must be 2
        z = paddle.as_complex(x)
        self.assertEqual(z.shape, [3])
        self.assertTrue(z.dtype in [paddle.complex64, paddle.complex128])

    def test_as_real(self):
        """测试as_real转换 / Test as_real conversion"""
        real = paddle.randn([3, 4])
        imag = paddle.randn([3, 4])
        z = paddle.complex(real, imag)
        x = paddle.as_real(z)
        # Last dim should be 2 (real, imag)
        self.assertEqual(x.shape, [3, 4, 2])

    def test_as_complex_as_real_roundtrip(self):
        """测试as_complex和as_real的往返转换 / Test as_complex/as_real roundtrip"""
        x = paddle.randn([3, 2])
        z = paddle.as_complex(x)
        x_recovered = paddle.as_real(z)
        np.testing.assert_allclose(x.numpy(), x_recovered.numpy(), rtol=1e-5)


class TestComplexArithmetic(unittest.TestCase):
    """测试复数算术运算 / Test complex arithmetic"""

    def test_complex_add(self):
        """测试复数加法 / Test complex addition"""
        z1 = paddle.to_tensor([1.0 + 2.0j, 3.0 + 4.0j])
        z2 = paddle.to_tensor([1.0 + 1.0j, 1.0 + 1.0j])
        result = z1 + z2
        expected = np.array([2.0 + 3.0j, 4.0 + 5.0j])
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_complex_multiply(self):
        """测试复数乘法 / Test complex multiplication"""
        z1 = paddle.to_tensor([1.0 + 1.0j])
        z2 = paddle.to_tensor([1.0 + 1.0j])
        result = z1 * z2
        # (1+i)*(1+i) = 1 + 2i - 1 = 2i
        expected = np.array([0.0 + 2.0j])
        np.testing.assert_allclose(result.numpy(), expected, atol=1e-5)

    def test_complex_matmul(self):
        """测试复数矩阵乘法 / Test complex matrix multiplication"""
        real = paddle.randn([3, 4])
        imag = paddle.randn([3, 4])
        z = paddle.complex(real, imag)
        real2 = paddle.randn([4, 5])
        imag2 = paddle.randn([4, 5])
        z2 = paddle.complex(real2, imag2)
        result = paddle.matmul(z, z2)
        self.assertEqual(result.shape, [3, 5])

    def test_complex_exp(self):
        """测试复数指数函数 / Test complex exponential"""
        # e^(i*pi) ≈ -1
        z = paddle.to_tensor([1j * np.pi], dtype='complex64')
        result = paddle.exp(z)
        self.assertAlmostEqual(
            float(paddle.real(result).numpy()[0]), -1.0, places=4
        )

    def test_complex_sum(self):
        """测试复数求和 / Test complex sum"""
        z = paddle.to_tensor([1.0 + 2.0j, 3.0 + 4.0j])
        result = z.sum()
        self.assertAlmostEqual(
            float(paddle.real(result).numpy()), 4.0, places=5
        )
        self.assertAlmostEqual(
            float(paddle.imag(result).numpy()), 6.0, places=5
        )


class TestComplexConversion(unittest.TestCase):
    """测试复数类型转换 / Test complex type conversion"""

    def test_complex64_to_complex128(self):
        """测试complex64转complex128 / Test complex64 to complex128"""
        z = paddle.to_tensor([1.0 + 2.0j], dtype='complex64')
        z128 = z.cast('complex128')
        self.assertEqual(z128.dtype, paddle.complex128)

    def test_real_to_complex(self):
        """测试实数创建复数 / Test creating complex from real"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        zeros = paddle.zeros_like(x)
        z = paddle.complex(x, zeros)
        # Imaginary part should be 0
        np.testing.assert_allclose(paddle.imag(z).numpy(), np.zeros(3))


if __name__ == '__main__':
    unittest.main()
