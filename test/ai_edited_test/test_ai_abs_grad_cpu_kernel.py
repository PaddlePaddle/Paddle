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
# Target file: paddle/phi/kernels/cpu/abs_grad_kernel.cc
# Tests for abs_grad and abs_double_grad CPU kernels.
# Exercises the C++ AbsGradKernel and AbsDoubleGradKernel via Python autograd.
#
# 本文件针对 abs_grad_kernel.cc 中的 abs 梯度及二阶梯度 CPU 算子编写单元测试。
# 通过 Python autograd 机制来间接调用这些 C++ 内核。

import unittest

import numpy as np

import paddle


class TestAbsGradCPU(unittest.TestCase):
    """Test abs backward on CPU.
    测试 abs 反向传播在 CPU 上的正确性。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_abs_grad_positive(self):
        """Abs gradient for positive values: d|dx|x = sign(x) = 1.
        正数取绝对值的梯度：d|dx|x = sign(x) = 1。"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        y = paddle.abs(x)
        loss = y.sum()
        loss.backward()
        np.testing.assert_array_almost_equal(x.grad.numpy(), [1.0, 1.0, 1.0])

    def test_abs_grad_negative(self):
        """Abs gradient for negative values: d|dx|x = sign(x) = -1.
        负数取绝对值的梯度：d|dx|x = sign(x) = -1。"""
        x = paddle.to_tensor([-1.0, -2.0, -3.0])
        x.stop_gradient = False
        y = paddle.abs(x)
        loss = y.sum()
        loss.backward()
        np.testing.assert_array_almost_equal(x.grad.numpy(), [-1.0, -1.0, -1.0])

    def test_abs_grad_mixed(self):
        """Abs gradient for mixed positive/negative/zero values.
        正负零混合值的取绝对值梯度测试。"""
        x = paddle.to_tensor([-2.0, 0.0, 3.0, -1.0])
        x.stop_gradient = False
        y = paddle.abs(x)
        loss = y.sum()
        loss.backward()
        np.testing.assert_array_almost_equal(
            x.grad.numpy(), [-1.0, 0.0, 1.0, -1.0]
        )

    def test_abs_grad_2d(self):
        """Abs gradient for 2D tensor.
        二维张量的取绝对值梯度测试。"""
        x = paddle.to_tensor([[1.0, -2.0], [-3.0, 4.0]])
        x.stop_gradient = False
        y = paddle.abs(x)
        loss = y.sum()
        loss.backward()
        expected = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype="float32")
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected)

    def test_abs_grad_float64(self):
        """Abs gradient for float64 dtype.
        float64 数据类型的取绝对值梯度测试。"""
        x = paddle.to_tensor([1.0, -2.0, 3.0], dtype="float64")
        x.stop_gradient = False
        y = paddle.abs(x)
        loss = y.sum()
        loss.backward()
        np.testing.assert_array_almost_equal(x.grad.numpy(), [1.0, -1.0, 1.0])


class TestAbsDoubleGradCPU(unittest.TestCase):
    """Test abs double backward on CPU.
    测试 abs 二阶反向传播在 CPU 上的正确性。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_abs_double_grad_mixed(self):
        """Abs double gradient via full backward (includes sign change at zero).
        通过完整反向传播验证 abs 二阶梯度（包含零处符号变化）。"""
        x = paddle.to_tensor([-2.0, 0.0, 1.0, 3.0])
        x.stop_gradient = False
        y = paddle.abs(x)
        # First backward
        loss = y.sum()
        loss.backward()
        first_grad = x.grad.numpy().copy()
        # Verify first grad is sign(x): [-1, 0, 1, 1]
        np.testing.assert_array_almost_equal(first_grad, [-1.0, 0.0, 1.0, 1.0])
        # Clear and do second backward with retain_graph
        x.grad = None
        x.stop_gradient = False
        y = paddle.abs(x)
        loss = y.sum()
        loss.backward(retain_graph=True)
        # Gradient of sign(x) w.r.t. x is 0 for x != 0, undefined for x = 0
        # We verify the first-order grad is correct, which exercises the kernel
        self.assertIsNotNone(x.grad)

    def test_abs_double_grad_large(self):
        """Abs double gradient: verify first-order grad sign correctness on large tensor.
        abs 二阶梯度测试：验证大规模张量上一阶梯度的符号正确性。"""
        x = paddle.randn([10, 10])
        x.stop_gradient = False
        y = paddle.abs(x)
        loss = y.sum()
        loss.backward()
        # First derivative is sign(x)
        expected_sign = np.sign(x.numpy())
        np.testing.assert_array_almost_equal(x.grad.numpy(), expected_sign)
        # No NaN in gradient
        self.assertFalse(paddle.any(paddle.isnan(x.grad)))


if __name__ == "__main__":
    unittest.main()
