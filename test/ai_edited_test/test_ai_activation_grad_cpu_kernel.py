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

# [AUTO-GENERATED] Do not edit manually.
# Target source: paddle/phi/kernels/cpu/activation_grad_kernel.cc
# Generated for exercising C++ CPU kernel: various activation gradient kernels
# (ReluGrad, TanhGrad, SigmoidGrad, LeakyReluGrad, ExpGrad, SqrtGrad, etc.)
#
# 测试激活函数梯度 CPU 内核
# Tests for Activation gradient CPU kernels

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


def numerical_gradient(f, x, eps=1e-5):
    """计算数值梯度用于验证 / Compute numerical gradient for verification"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


class TestReluGradKernel(unittest.TestCase):
    """ReLU 梯度内核测试 / ReLU gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_relu_grad_positive(self):
        """测试 ReLU 正数区域的梯度
        Test ReLU gradient for positive region
        """
        x_np = np.array([1.0, 2.0, 3.0], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = F.relu(x)
        out = y.sum()
        out.backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0, 1.0], atol=1e-6)

    def test_relu_grad_negative(self):
        """测试 ReLU 负数区域的梯度
        Test ReLU gradient for negative region
        """
        x_np = np.array([-1.0, -2.0, -3.0], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = F.relu(x)
        out = y.sum()
        out.backward()
        np.testing.assert_allclose(x.grad.numpy(), [0.0, 0.0, 0.0], atol=1e-6)

    def test_relu_grad_mixed(self):
        """测试 ReLU 混合区域的梯度
        Test ReLU gradient for mixed positive and negative
        """
        x_np = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = F.relu(x)
        out = y.sum()
        out.backward()
        np.testing.assert_allclose(
            x.grad.numpy(), [0.0, 0.0, 0.0, 1.0, 1.0], atol=1e-6
        )

    def test_relu_grad_2d(self):
        """测试 2D 张量的 ReLU 梯度
        Test ReLU gradient for 2D tensor
        """
        x_np = np.array([[-1, 2], [3, -4]], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = F.relu(x)
        out = y.sum()
        out.backward()
        np.testing.assert_allclose(
            x.grad.numpy(), [[0.0, 1.0], [1.0, 0.0]], atol=1e-6
        )


class TestTanhGradKernel(unittest.TestCase):
    """Tanh 梯度内核测试 / Tanh gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_tanh_grad_at_zero(self):
        """测试 Tanh 在零点的梯度为 1
        Test Tanh gradient at zero equals 1
        """
        x = paddle.to_tensor([0.0], dtype="float32", stop_gradient=False)
        y = paddle.tanh(x)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0], atol=1e-5)

    def test_tanh_grad_shape(self):
        """测试 Tanh 梯度的输出形状
        Test Tanh gradient output shape
        """
        x = paddle.randn([5, 10], dtype="float32")
        x.stop_gradient = False
        y = paddle.tanh(x)
        y.sum().backward()
        self.assertEqual(x.grad.shape, (5, 10))

    def test_tanh_grad_large_input(self):
        """测试 Tanh 对大输入的梯度（接近零）
        Test Tanh gradient for large input (should be near zero)
        """
        x = paddle.to_tensor([100.0], dtype="float32", stop_gradient=False)
        y = paddle.tanh(x)
        y.sum().backward()
        # tanh(100) ≈ 1, sech^2(100) ≈ 0
        self.assertAlmostEqual(x.grad.numpy()[0], 0.0, places=3)


class TestSigmoidGradKernel(unittest.TestCase):
    """Sigmoid 梯度内核测试 / Sigmoid gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_sigmoid_grad_at_zero(self):
        """测试 Sigmoid 在零点的梯度为 0.25
        Test Sigmoid gradient at zero equals 0.25
        """
        x = paddle.to_tensor([0.0], dtype="float32", stop_gradient=False)
        y = F.sigmoid(x)
        y.sum().backward()
        # sigmoid(0) = 0.5, gradient = 0.5 * (1 - 0.5) = 0.25
        np.testing.assert_allclose(x.grad.numpy(), [0.25], atol=1e-5)

    def test_sigmoid_grad_large_positive(self):
        """测试 Sigmoid 对大正数的梯度（接近零）
        Test Sigmoid gradient for large positive (near zero)
        """
        x = paddle.to_tensor([100.0], dtype="float32", stop_gradient=False)
        y = F.sigmoid(x)
        y.sum().backward()
        self.assertAlmostEqual(x.grad.numpy()[0], 0.0, places=3)

    def test_sigmoid_grad_large_negative(self):
        """测试 Sigmoid 对大负数的梯度（接近零）
        Test Sigmoid gradient for large negative (near zero)
        """
        x = paddle.to_tensor([-100.0], dtype="float32", stop_gradient=False)
        y = F.sigmoid(x)
        y.sum().backward()
        self.assertAlmostEqual(x.grad.numpy()[0], 0.0, places=3)


class TestLeakyReluGradKernel(unittest.TestCase):
    """LeakyReLU 梯度内核测试 / LeakyReLU gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_leaky_relu_grad_positive(self):
        """测试 LeakyReLU 正数区域的梯度
        Test LeakyReLU gradient for positive region
        """
        x = paddle.to_tensor([1.0, 2.0], dtype="float32", stop_gradient=False)
        y = F.leaky_relu(x, negative_slope=0.01)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0], atol=1e-5)

    def test_leaky_relu_grad_negative(self):
        """测试 LeakyReLU 负数区域的梯度
        Test LeakyReLU gradient for negative region
        """
        x = paddle.to_tensor([-1.0, -2.0], dtype="float32", stop_gradient=False)
        y = F.leaky_relu(x, negative_slope=0.01)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [0.01, 0.01], atol=1e-5)

    def test_leaky_relu_grad_custom_slope(self):
        """测试自定义斜率的 LeakyReLU 梯度
        Test LeakyReLU gradient with custom negative slope
        """
        slope = 0.2
        x = paddle.to_tensor(
            [-1.0, 0.0, 1.0], dtype="float32", stop_gradient=False
        )
        y = F.leaky_relu(x, negative_slope=slope)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [slope, 1.0, 1.0], atol=1e-5)


class TestExpGradKernel(unittest.TestCase):
    """Exp 梯度内核测试 / Exp gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_exp_grad_at_zero(self):
        """测试 Exp 在零点的梯度为 1
        Test Exp gradient at zero equals 1
        """
        x = paddle.to_tensor([0.0], dtype="float32", stop_gradient=False)
        y = paddle.exp(x)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0], atol=1e-5)

    def test_exp_grad(self):
        """测试 Exp 梯度 = exp(x)
        Test Exp gradient equals exp(x)
        """
        x_np = np.array([0.0, 1.0, 2.0], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.exp(x)
        y.sum().backward()
        expected = np.exp(x_np)
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-5)


class TestSqrtGradKernel(unittest.TestCase):
    """Sqrt 梯度内核测试 / Sqrt gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_sqrt_grad(self):
        """测试 Sqrt 梯度 = 1/(2*sqrt(x))
        Test Sqrt gradient = 1/(2*sqrt(x))
        """
        x_np = np.array([1.0, 4.0, 9.0], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.sqrt(x)
        y.sum().backward()
        expected = 1.0 / (2.0 * np.sqrt(x_np))
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-5)


class TestLogGradKernel(unittest.TestCase):
    """Log 梯度内核测试 / Log gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_log_grad(self):
        """测试 Log 梯度 = 1/x
        Test Log gradient = 1/x
        """
        x_np = np.array([1.0, 2.0, 3.0], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.log(x)
        y.sum().backward()
        expected = 1.0 / x_np
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-5)


class TestSquareGradKernel(unittest.TestCase):
    """Square 梯度内核测试 / Square gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_square_grad(self):
        """测试 Square 梯度 = 2*x
        Test Square gradient = 2*x
        """
        x_np = np.array([1.0, 2.0, -3.0, 0.5], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.square(x)
        y.sum().backward()
        expected = 2.0 * x_np
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-5)


class TestActivationGradFloat64(unittest.TestCase):
    """float64 类型激活函数梯度测试 / Activation gradient tests with float64"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_relu_grad_float64(self):
        """测试 float64 类型的 ReLU 梯度
        Test ReLU gradient with float64
        """
        x_np = np.array([-1.0, 0.0, 1.0, 2.0], dtype="float64")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = F.relu(x)
        y.sum().backward()
        np.testing.assert_allclose(
            x.grad.numpy(), [0.0, 0.0, 1.0, 1.0], atol=1e-10
        )
        self.assertEqual(x.grad.dtype, paddle.float64)

    def test_tanh_grad_float64(self):
        """测试 float64 类型的 Tanh 梯度
        Test Tanh gradient with float64
        """
        x_np = np.array([0.5, -0.5], dtype="float64")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.tanh(x)
        y.sum().backward()
        expected = 1.0 - np.tanh(x_np) ** 2
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-10)
        self.assertEqual(x.grad.dtype, paddle.float64)

    def test_sigmoid_grad_float64(self):
        """测试 float64 类型的 Sigmoid 梯度
        Test Sigmoid gradient with float64
        """
        x_np = np.array([0.0, 1.0, -1.0], dtype="float64")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = F.sigmoid(x)
        y.sum().backward()
        sig = 1.0 / (1.0 + np.exp(-x_np))
        expected = sig * (1.0 - sig)
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-10)
        self.assertEqual(x.grad.dtype, paddle.float64)


class TestHardShrinkGradKernel(unittest.TestCase):
    """HardShrink 梯度内核测试 / HardShrink gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_hardshrink_grad_inside_threshold(self):
        """测试阈值内区域的梯度为零
        Test HardShrink gradient inside threshold is zero
        """
        x = paddle.to_tensor(
            [-0.2, 0.1, 0.3], dtype="float32", stop_gradient=False
        )
        y = F.hardshrink(x, threshold=0.5)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [0.0, 0.0, 0.0], atol=1e-6)

    def test_hardshrink_grad_outside_threshold(self):
        """测试阈值外区域的梯度为一
        Test HardShrink gradient outside threshold is one
        """
        x = paddle.to_tensor([-1.0, 2.0], dtype="float32", stop_gradient=False)
        y = F.hardshrink(x, threshold=0.5)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0], atol=1e-6)

    def test_hardshrink_grad_at_boundary(self):
        """测试边界处的梯度（边界值被保留，梯度为1）
        Test HardShrink gradient at boundary: value is kept, grad=1
        """
        x = paddle.to_tensor([0.5, -0.5], dtype="float32", stop_gradient=False)
        y = F.hardshrink(x, threshold=0.5)
        y.sum().backward()
        # At boundary exactly (>= threshold), value is kept, grad=1
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0], atol=1e-6)


class TestSoftShrinkGradKernel(unittest.TestCase):
    """SoftShrink 梯度内核测试 / SoftShrink gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_softshrink_grad_inside_threshold(self):
        """测试阈值内区域的梯度为零
        Test SoftShrink gradient inside threshold is zero
        """
        x = paddle.to_tensor([0.1, -0.2], dtype="float32", stop_gradient=False)
        y = F.softshrink(x, threshold=0.5)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [0.0, 0.0], atol=1e-6)

    def test_softshrink_grad_outside_threshold(self):
        """测试阈值外区域的梯度为一
        Test SoftShrink gradient outside threshold is one
        """
        x = paddle.to_tensor([1.0, -2.0], dtype="float32", stop_gradient=False)
        y = F.softshrink(x, threshold=0.5)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0], atol=1e-6)


class TestRelu6GradKernel(unittest.TestCase):
    """ReLU6 梯度内核测试 / ReLU6 gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_relu6_grad(self):
        """测试 ReLU6 梯度
        Test ReLU6 gradient
        """
        x_np = np.array([-1.0, 0.0, 3.0, 6.0, 7.0], dtype="float32")
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = F.relu6(x)
        y.sum().backward()
        # grad = 1 for 0 < x < 6, 0 otherwise
        expected = [0.0, 0.0, 1.0, 0.0, 0.0]
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-6)


class TestEluGradKernel(unittest.TestCase):
    """ELU 梯度内核测试 / ELU gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_elu_grad_positive(self):
        """测试 ELU 正数区域的梯度为 1
        Test ELU gradient for positive region equals 1
        """
        x = paddle.to_tensor([1.0, 2.0], dtype="float32", stop_gradient=False)
        y = F.elu(x, alpha=1.0)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0, 1.0], atol=1e-5)

    def test_elu_grad_zero(self):
        """测试 ELU 在零点的梯度为 1
        Test ELU gradient at zero equals 1
        """
        x = paddle.to_tensor([0.0], dtype="float32", stop_gradient=False)
        y = F.elu(x, alpha=1.0)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0], atol=1e-5)


class TestSiluGradKernel(unittest.TestCase):
    """SiLU 梯度内核测试 / SiLU gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_silu_grad_at_zero(self):
        """测试 SiLU 在零点的梯度
        Test SiLU gradient at zero
        """
        x = paddle.to_tensor([0.0], dtype="float32", stop_gradient=False)
        y = F.silu(x)
        y.sum().backward()
        # silu(0) = 0, sigmoid(0) = 0.5
        # grad = sigmoid(0) + 0 * (1 - sigmoid(0)) = 0.5
        np.testing.assert_allclose(x.grad.numpy(), [0.5], atol=1e-5)

    def test_silu_grad_shape(self):
        """测试 SiLU 梯度形状
        Test SiLU gradient shape
        """
        x = paddle.randn([4, 8], dtype="float32")
        x.stop_gradient = False
        y = F.silu(x)
        y.sum().backward()
        self.assertEqual(x.grad.shape, (4, 8))


class TestMishGradKernel(unittest.TestCase):
    """Mish 梯度内核测试 / Mish gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_mish_grad_at_zero(self):
        """测试 Mish 在零点的梯度
        Test Mish gradient at zero
        """
        x = paddle.to_tensor([0.0], dtype="float32", stop_gradient=False)
        y = F.mish(x)
        y.sum().backward()
        # mish(0) = 0, softplus(0) = ln(2)
        # grad at 0 = sigmoid(softplus(0)) * (tanh(softplus(0)) + softplus(0) * sech^2(softplus(0)))
        # Approximate value is 0.6 for float32 precision
        self.assertAlmostEqual(x.grad.numpy()[0], 0.6, places=1)

    def test_mish_grad_shape(self):
        """测试 Mish 梯度形状
        Test Mish gradient shape
        """
        x = paddle.randn([3, 5], dtype="float32")
        x.stop_gradient = False
        y = F.mish(x)
        y.sum().backward()
        self.assertEqual(x.grad.shape, (3, 5))


class TestPowGradKernel(unittest.TestCase):
    """Pow 梯度内核测试 / Pow gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_pow_grad(self):
        """测试 Pow 梯度
        Test Pow gradient
        """
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
        factor = 2.0
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.pow(x, factor)
        y.sum().backward()
        expected = factor * x_np ** (factor - 1)
        np.testing.assert_allclose(x.grad.numpy(), expected, atol=1e-4)

    def test_pow_grad_zero_exponent(self):
        """测试零指数的 Pow 梯度（应为零）
        Test Pow gradient with zero exponent (should be zero)
        """
        x = paddle.to_tensor(
            [1.0, 2.0, 3.0], dtype="float32", stop_gradient=False
        )
        y = paddle.pow(x, 0.0)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [0.0, 0.0, 0.0], atol=1e-6)


class TestHardSigmoidGradKernel(unittest.TestCase):
    """HardSigmoid 梯度内核测试 / HardSigmoid gradient kernel tests"""

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_hardsigmoid_grad_inside(self):
        """测试 HardSigmoid 在中间区域的梯度
        Test HardSigmoid gradient in middle region
        """
        x = paddle.to_tensor([0.0], dtype="float32", stop_gradient=False)
        y = F.hardsigmoid(x)
        y.sum().backward()
        # hard_sigmoid(x) = slope * x + offset, grad = slope = 1/6
        np.testing.assert_allclose(x.grad.numpy(), [1.0 / 6.0], atol=1e-5)

    def test_hardsigmoid_grad_outside(self):
        """测试 HardSigmoid 在外侧区域的梯度为零
        Test HardSigmoid gradient outside region is zero
        """
        x = paddle.to_tensor(
            [100.0, -100.0], dtype="float32", stop_gradient=False
        )
        y = F.hardsigmoid(x)
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [0.0, 0.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
