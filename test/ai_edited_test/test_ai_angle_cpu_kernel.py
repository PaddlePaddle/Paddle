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

# [AUTO-GENERATED]
# Target file: paddle/phi/kernels/cpu/angle_kernel.cc
# Tests for angle CPU kernel.
# Exercises the C++ AngleKernel via paddle.angle API.
#
# 本文件针对 angle_kernel.cc 中的角度计算 CPU 算子编写单元测试。
# 通过 paddle.angle API 来调用 C++ 内核，验证复数张量的角度计算结果。

import unittest

import numpy as np

import paddle


class TestAngleCPU(unittest.TestCase):
    """Test angle of complex tensors on CPU.
    测试 CPU 上复数张量的角度计算。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_angle_complex64_real(self):
        """Angle of real-valued complex numbers should be 0 or pi.
        纯实数的复数角度应为 0 或 pi。"""
        # Real positive numbers -> angle = 0
        x = paddle.to_tensor([1.0 + 0j, 2.0 + 0j, 3.0 + 0j], dtype="complex64")
        result = paddle.angle(x)
        np.testing.assert_array_almost_equal(result.numpy(), [0.0, 0.0, 0.0])

    def test_angle_complex64_imaginary(self):
        """Angle of purely imaginary numbers.
        纯虚数的复数角度测试。"""
        # Pure imaginary positive -> pi/2
        x = paddle.to_tensor([1j, 2j], dtype="complex64")
        result = paddle.angle(x)
        np.testing.assert_array_almost_equal(
            result.numpy(), [np.pi / 2, np.pi / 2]
        )

    def test_angle_complex64_negative_imaginary(self):
        """Angle of negative purely imaginary numbers -> -pi/2.
        负纯虚数的角度应为 -pi/2。"""
        x = paddle.to_tensor([-1j, -2j], dtype="complex64")
        result = paddle.angle(x)
        np.testing.assert_array_almost_equal(
            result.numpy(), [-np.pi / 2, -np.pi / 2]
        )

    def test_angle_complex64_negative_real(self):
        """Angle of negative real numbers -> pi.
        负实数的角度应为 pi。"""
        x = paddle.to_tensor([-1.0 + 0j, -3.0 + 0j], dtype="complex64")
        result = paddle.angle(x)
        np.testing.assert_array_almost_equal(result.numpy(), [np.pi, np.pi])

    def test_angle_complex64_general(self):
        """Angle of general complex numbers.
        一般复数角度测试。"""
        x = paddle.to_tensor([1.0 + 1j, 1.0 - 1j, -1.0 + 1j], dtype="complex64")
        result = paddle.angle(x)
        np.testing.assert_array_almost_equal(
            result.numpy(), [np.pi / 4, -np.pi / 4, 3 * np.pi / 4]
        )

    def test_angle_complex128(self):
        """Angle for complex128 dtype.
        complex128 数据类型的角度测试。"""
        x = paddle.to_tensor([1.0 + 1j, -1.0 + 0j], dtype="complex128")
        result = paddle.angle(x)
        np.testing.assert_array_almost_equal(result.numpy(), [np.pi / 4, np.pi])

    def test_angle_complex64_2d(self):
        """Angle for 2D complex tensor.
        二维复数张量的角度测试。"""
        x = paddle.to_tensor(
            [[1.0 + 0j, 0.0 + 1j], [-1.0 + 0j, 0.0 - 1j]], dtype="complex64"
        )
        result = paddle.angle(x)
        expected = np.array(
            [[0.0, np.pi / 2], [np.pi, -np.pi / 2]], dtype="float32"
        )
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_angle_output_dtype(self):
        """Angle output should be real-valued float.
        angle 的输出应为实数浮点类型。"""
        x = paddle.to_tensor([1.0 + 1j], dtype="complex64")
        result = paddle.angle(x)
        self.assertEqual(result.dtype, paddle.float32)

    def test_angle_shape(self):
        """Angle preserves input shape.
        angle 应保持输入的形状。"""
        x = paddle.to_tensor(
            paddle.randn([2, 3, 4]) + 1j * paddle.randn([2, 3, 4]),
            dtype="complex64",
        )
        result = paddle.angle(x)
        self.assertEqual(result.shape, [2, 3, 4])

    def test_angle_zero(self):
        """Angle of zero complex number.
        零复数的角度测试。"""
        x = paddle.to_tensor([0.0 + 0j], dtype="complex64")
        result = paddle.angle(x)
        # Angle of 0 is 0 in Paddle
        np.testing.assert_array_almost_equal(result.numpy(), [0.0])


if __name__ == "__main__":
    unittest.main()
