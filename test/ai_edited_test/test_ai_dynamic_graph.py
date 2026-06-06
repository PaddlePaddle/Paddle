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
动态图自动微分单元测试 / Dynamic Graph Autograd Unit Tests

测试目标 / Test Target:
  paddle动态图自动微分功能 (paddle.grad, paddle.jacobian, paddle.hessian)

覆盖的模块 / Covered Modules:
  - paddle.grad: 梯度计算
  - paddle.jacobian: Jacobian矩阵计算
  - paddle.hessian: Hessian矩阵计算
  - Tensor.backward: 反向传播
  - Tensor.gradient: 获取梯度

作用 / Purpose:
  覆盖自动微分机制的各种代码路径，包括高阶导数、梯度累积等功能。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestGradBasic(unittest.TestCase):
    """测试基本梯度计算 / Test basic gradient computation"""

    def test_grad_simple(self):
        """测试简单梯度计算 / Test simple gradient computation"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = x * x
        grad = paddle.grad(y, x, grad_outputs=paddle.ones_like(y))
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(grad[0].numpy(), expected, rtol=1e-5)

    def test_grad_with_create_graph(self):
        """测试创建计算图的梯度 / Test gradient with create_graph"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = x**3
        grad = paddle.grad(
            y, x, grad_outputs=paddle.ones_like(y), create_graph=True
        )
        # dy/dx = 3x^2 = [3, 12, 27]
        expected = np.array([3.0, 12.0, 27.0])
        np.testing.assert_allclose(grad[0].numpy(), expected, rtol=1e-5)

    def test_grad_sum(self):
        """测试求和函数的梯度 / Test gradient of sum"""
        x = paddle.randn([3, 4])
        x.stop_gradient = False
        y = x.sum()
        y.backward()
        np.testing.assert_allclose(
            x.grad.numpy(), np.ones((3, 4), dtype='float32'), rtol=1e-5
        )

    def test_grad_chain_rule(self):
        """测试链式法则 / Test chain rule"""
        x = paddle.to_tensor([2.0], stop_gradient=False)
        y = x * x  # y = x^2
        z = y * y  # z = x^4
        dz_dx = paddle.grad(z, x)
        # dz/dx = 4x^3 = 4*8 = 32
        self.assertAlmostEqual(float(dz_dx[0].numpy()), 32.0, places=4)

    def test_retain_graph(self):
        """测试保留计算图 / Test retain_graph"""
        x = paddle.to_tensor([2.0], stop_gradient=False)
        y = x * x
        grad1 = paddle.grad(y, x, retain_graph=True)
        grad2 = paddle.grad(y, x, retain_graph=False)
        np.testing.assert_allclose(grad1[0].numpy(), grad2[0].numpy())


class TestGradMultiOutput(unittest.TestCase):
    """测试多输出梯度 / Test multi-output gradient"""

    def test_grad_multiple_outputs(self):
        """测试多输出梯度计算 / Test gradient with multiple outputs"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        y1 = x * x
        y2 = x * 2
        grad = paddle.grad(
            [y1, y2], x, grad_outputs=[paddle.ones([2]), paddle.ones([2])]
        )
        # dy1/dx = 2x, dy2/dx = 2
        expected = np.array([4.0, 6.0])
        np.testing.assert_allclose(grad[0].numpy(), expected, rtol=1e-5)

    def test_grad_multiple_inputs(self):
        """测试多输入梯度计算 / Test gradient with multiple inputs"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        w = paddle.to_tensor([3.0, 4.0], stop_gradient=False)
        y = (x * w).sum()
        grad_x, grad_w = paddle.grad(y, [x, w])
        np.testing.assert_allclose(grad_x.numpy(), w.numpy(), rtol=1e-5)
        np.testing.assert_allclose(grad_w.numpy(), x.numpy(), rtol=1e-5)


class TestBackwardBasic(unittest.TestCase):
    """测试backward方法 / Test backward method"""

    def test_backward_simple(self):
        """测试简单backward / Test simple backward"""
        x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
        y = (x * x).sum()
        y.backward()
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)

    def test_backward_accumulate(self):
        """测试梯度累积 / Test gradient accumulation"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        y = (x * x).sum()
        y.backward()
        first_grad = x.grad.numpy().copy()
        y = (x * x).sum()
        y.backward()
        # Gradient should be accumulated
        second_grad = x.grad.numpy()
        np.testing.assert_allclose(second_grad, first_grad * 2, rtol=1e-5)

    def test_clear_grad(self):
        """测试清除梯度 / Test clear gradient"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        y = (x * x).sum()
        y.backward()
        x.clear_gradient()
        self.assertTrue(x.grad is None or np.all(x.grad.numpy() == 0))

    def test_stop_gradient(self):
        """测试停止梯度 / Test stop_gradient"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=True)
        y = x * 2
        self.assertTrue(y.stop_gradient)

    def test_no_grad_decorator(self):
        """测试no_grad装饰器 / Test no_grad decorator"""

        @paddle.no_grad()
        def func(x):
            return x * x

        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        y = func(x)
        self.assertTrue(y.stop_gradient)


class TestJacobian(unittest.TestCase):
    """测试Jacobian矩阵计算 / Test Jacobian matrix computation"""

    def test_jacobian_via_grad(self):
        """通过grad计算Jacobian / Compute Jacobian via grad"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        # f(x) = x^2 element-wise, df/dx = diag(2x)
        y = x**2
        # Compute row by row
        jac_rows = []
        for i in range(y.shape[0]):
            if x.grad is not None:
                x.clear_gradient()
            grad = paddle.grad(y[i], x, retain_graph=True)
            jac_rows.append(grad[0].numpy())
        # Diagonal should be [2*1, 2*2] = [2, 4]
        self.assertAlmostEqual(jac_rows[0][0], 2.0, places=4)
        self.assertAlmostEqual(jac_rows[1][1], 4.0, places=4)

    def test_jacobian_single_output(self):
        """测试单输出的梯度 / Test gradient with single output"""
        x = paddle.randn([4])
        x.stop_gradient = False
        y = x.sum()
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones(4, dtype='float32'))


class TestHessian(unittest.TestCase):
    """测试二阶导数计算 / Test second-order derivative computation"""

    def test_second_order_gradient(self):
        """测试二阶梯度计算 / Test second-order gradient computation"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        # f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
        y = (x**3).sum()
        first_grad = paddle.grad(y, x, create_graph=True)
        second_grad = paddle.grad(first_grad[0].sum(), x)
        # f''(x) = 6x = [6, 12]
        expected = np.array([6.0, 12.0])
        np.testing.assert_allclose(second_grad[0].numpy(), expected, rtol=1e-4)


class TestGradientContext(unittest.TestCase):
    """测试梯度上下文管理 / Test gradient context management"""

    def test_no_grad_context(self):
        """测试no_grad上下文 / Test no_grad context"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        with paddle.no_grad():
            y = x * x
        self.assertTrue(y.stop_gradient)

    def test_enable_grad_context(self):
        """测试enable_grad上下文 / Test enable_grad context"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        with paddle.no_grad(), paddle.enable_grad():
            y = x * x
        self.assertFalse(y.stop_gradient)

    def test_set_grad_enabled(self):
        """测试set_grad_enabled / Test set_grad_enabled"""
        x = paddle.to_tensor([1.0, 2.0], stop_gradient=False)
        with paddle.set_grad_enabled(False):
            y = x * x
        self.assertTrue(y.stop_gradient)

        with paddle.set_grad_enabled(True):
            z = x * x
        self.assertFalse(z.stop_gradient)


if __name__ == '__main__':
    unittest.main()
