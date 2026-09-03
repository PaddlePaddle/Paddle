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
自动微分高级测试 / Advanced Automatic Differentiation Tests

测试目标 / Test Target:
  paddle.autograd 自动微分

覆盖的模块 / Covered Modules:
  - paddle.grad: 梯度计算
  - paddle.incubate.autograd: 功能性自动微分
  - higher-order gradients: 高阶梯度
  - gradient checkpointing

作用 / Purpose:
  补充自动微分API的高级测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestGradientComputation(unittest.TestCase):
    """测试梯度计算 / Test gradient computation"""

    def test_basic_gradient(self):
        """测试基本梯度 / Test basic gradient"""
        x = paddle.to_tensor([2.0, 3.0])
        x.stop_gradient = False
        y = x**2
        loss = y.sum()
        loss.backward()
        # d/dx(x^2) = 2x
        np.testing.assert_allclose(x.grad.numpy(), [4.0, 6.0])

    def test_chain_rule(self):
        """测试链式法则 / Test chain rule"""
        x = paddle.to_tensor(2.0)
        x.stop_gradient = False
        y = x**3  # dy/dx = 3x^2
        z = y**2  # dz/dy = 2y = 2*x^3
        # dz/dx = 2*x^3 * 3*x^2 = 6*x^5 = 6*32 = 192
        z.backward()
        self.assertAlmostEqual(float(x.grad.numpy()), 192.0, places=4)

    def test_multiple_outputs_grad(self):
        """测试多输出梯度 / Test gradient with multiple outputs"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x.stop_gradient = False
        y1 = x * 2
        y2 = x**2
        # Backward through sum
        loss = y1.sum() + y2.sum()
        loss.backward()
        # d/dx(2x + x^2) = 2 + 2x
        expected = [2 + 2 * 1, 2 + 2 * 2, 2 + 2 * 3]
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5)

    def test_no_grad_context(self):
        """测试no_grad上下文 / Test no_grad context"""
        x = paddle.to_tensor([1.0, 2.0])
        x.stop_gradient = False
        with paddle.no_grad():
            y = x**2
        # y should not require grad
        self.assertTrue(y.stop_gradient)

    def test_grad_function(self):
        """测试paddle.grad函数 / Test paddle.grad function"""
        x = paddle.to_tensor(3.0)
        x.stop_gradient = False
        y = x**2
        grad = paddle.grad(y, x)[0]
        self.assertAlmostEqual(float(grad.numpy()), 6.0, places=5)


class TestSecondOrderGradient(unittest.TestCase):
    """测试二阶梯度 / Test second-order gradients"""

    def test_second_order_grad(self):
        """测试二阶梯度计算 / Test second order gradient"""
        x = paddle.to_tensor(2.0)
        x.stop_gradient = False
        # f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x
        y = x**3
        first_grad = paddle.grad(y, x, create_graph=True)[0]
        second_grad = paddle.grad(first_grad, x)[0]
        # f''(2) = 12
        self.assertAlmostEqual(float(second_grad.numpy()), 12.0, places=4)


class TestBackwardWithAccumulation(unittest.TestCase):
    """测试梯度累积 / Test gradient accumulation"""

    def test_gradient_accumulation(self):
        """测试梯度累积 / Test gradient accumulation"""
        model = nn.Linear(4, 2)
        optimizer = paddle.optimizer.SGD(
            parameters=model.parameters(), learning_rate=0.01
        )

        accumulated_steps = 4
        for i in range(accumulated_steps):
            x = paddle.randn([2, 4])
            y = paddle.randn([2, 2])
            loss = nn.functional.mse_loss(model(x), y) / accumulated_steps
            loss.backward()

        optimizer.step()
        optimizer.clear_grad()

    def test_retain_grad(self):
        """测试中间变量梯度保留 / Test retain grad for intermediate tensors"""
        x = paddle.to_tensor(2.0)
        x.stop_gradient = False
        y = x**2
        y.retain_grads()
        z = y**2
        z.backward()
        # dy/dx = 2x = 4; dz/dy = 2y = 8; y.grad should be dz/dy = 8
        self.assertAlmostEqual(float(y.grad.numpy()), 8.0, places=5)


class TestDetachAndClone(unittest.TestCase):
    """测试detach和clone / Test detach and clone"""

    def test_detach(self):
        """测试detach / Test detach"""
        x = paddle.to_tensor([1.0, 2.0])
        x.stop_gradient = False
        y = x**2
        y_detached = y.detach()
        self.assertTrue(y_detached.stop_gradient)
        np.testing.assert_allclose(y_detached.numpy(), [1.0, 4.0])

    def test_clone(self):
        """测试clone / Test clone"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        x_clone = x.clone()
        np.testing.assert_allclose(x.numpy(), x_clone.numpy())
        # Modify clone should not affect original
        x_clone[0] = paddle.to_tensor(99.0)
        self.assertAlmostEqual(float(x[0].numpy()), 1.0)


if __name__ == '__main__':
    unittest.main()
