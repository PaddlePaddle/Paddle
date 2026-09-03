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

# [AUTO-GENERATED] Tests for phi/kernels/cpu/asgd_kernel.cc
# asgd_kernel.cc: CPU ASGD (Average Stochastic Gradient Descent) optimizer kernel
# Update rule: d_out = d - y + grad; y_out = grad; param_out = param - (lr/n) * d_out
# where n = min(m, batch_num) and m is the step counter (1-indexed).
# y is a ring buffer of size batch_num: y = ys[m % batch_num]

import unittest

import numpy as np

import paddle


class TestASGDOptimizer(unittest.TestCase):
    """Test suite for paddle.optimizer.ASGD CPU kernel.

    测试 paddle.optimizer.ASGD 优化器的 CPU 内核，涵盖单步更新、多步更新、
    不同学习率、batch_num 参数等场景。
    ASGD (Average Stochastic Gradient Descent) 实现如下更新规则：
      d_out = d - y + grad
      y_out = grad
      param_out = param - (lr/n) * d_out
    其中 n = min(m, batch_num)，m 是步数计数器（从 1 开始）。
    """

    def setUp(self):
        paddle.set_device('cpu')

    def test_asgd_single_step(self):
        """Test ASGD single optimization step.

        测试 ASGD 单步优化。
        Step 1: d=0, y=0, grad=ones, n=min(1,1)=1
          d_out = 0 - 0 + grad = grad
          param_out = param - lr/1 * grad
        """
        param = paddle.create_parameter(shape=[3], dtype='float32')
        param.set_value(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.01, batch_num=1, parameters=[param]
        )

        loss = param.sum()
        loss.backward()
        optimizer.step()

        # grad = [1,1,1], n=1
        # param_out = [1,2,3] - 0.01 * [1,1,1] = [0.99, 1.99, 2.99]
        expected = np.array([0.99, 1.99, 2.99], dtype=np.float32)
        np.testing.assert_allclose(param.numpy(), expected, rtol=1e-5)

    def test_asgd_multiple_steps_batch1(self):
        """Test ASGD with batch_num=1 (n always 1).

        测试 batch_num=1 时多步优化（n 始终为 1）。
        With batch_num=1, n = min(m, 1) = 1 for all steps.
        Also d_out = d - y + grad; when batch_num=1, y is always the last grad.
        So d_out = d - grad_prev + grad_current.
        When all grads are same, d_out = d (stable).
        """
        param = paddle.create_parameter(shape=[3], dtype='float32')
        param.set_value(np.array([10.0, 10.0, 10.0], dtype=np.float32))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.1, batch_num=1, parameters=[param]
        )

        for step in range(3):
            optimizer.clear_grad()
            loss = param.sum()
            loss.backward()
            optimizer.step()
            # Each step: param -= 0.1/1 * 1 = param - 0.1
            expected = 10.0 - 0.1 * (step + 1)
            np.testing.assert_allclose(param.numpy(), [expected] * 3, rtol=1e-5)

    def test_asgd_batch_num_capped(self):
        """Test ASGD with batch_num=2 (n capped at 2).

        测试 batch_num=2 时 n 上限为 2。
        Step 1: n=min(1,2)=1, param -= lr/1 * d_out
        Step 2: n=min(2,2)=2, param -= lr/2 * d_out
        Step 3: n=min(3,2)=2, param -= lr/2 * d_out
        """
        param = paddle.create_parameter(shape=[1], dtype='float32')
        param.set_value(np.array([100.0], dtype=np.float32))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=1.0, batch_num=2, parameters=[param]
        )

        # Step 1: d=0, y=0, grad=1, n=1
        # d_out = 0-0+1 = 1
        optimizer.clear_grad()
        loss = param.sum()
        loss.backward()
        optimizer.step()
        # param = 100 - 1/1 * 1 = 99
        np.testing.assert_allclose(param.numpy(), [99.0], rtol=1e-5)

        # Step 2: d=1, y(ys[1%2]=ys[1])=0 (ring buffer, ys[1] was 0), grad=1
        # d_out = 1 - 0 + 1 = 2, n=min(2,2)=2
        optimizer.clear_grad()
        loss = param.sum()
        loss.backward()
        optimizer.step()
        # param = 99 - 1/2 * 2 = 98
        np.testing.assert_allclose(param.numpy(), [98.0], rtol=1e-5)

    def test_asgd_different_lr(self):
        """Test ASGD with different learning rate values.

        测试不同学习率的 ASGD 优化。
        """
        for lr in [0.001, 0.1, 1.0]:
            param = paddle.create_parameter(shape=[1], dtype='float32')
            param.set_value(np.array([1.0], dtype=np.float32))
            optimizer = paddle.optimizer.ASGD(
                learning_rate=lr, batch_num=1, parameters=[param]
            )
            loss = param.sum()
            loss.backward()
            optimizer.step()
            # param = 1 - lr/1 * 1 = 1 - lr
            expected = 1.0 - lr
            np.testing.assert_allclose(param.numpy(), [expected], rtol=1e-5)

    def test_asgd_float64(self):
        """Test ASGD with float64 parameters.

        测试 float64 参数的 ASGD 优化。
        """
        param = paddle.create_parameter(shape=[2], dtype='float64')
        param.set_value(np.array([1.0, 2.0], dtype=np.float64))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.01, batch_num=1, parameters=[param]
        )

        loss = param.sum()
        loss.backward()
        optimizer.step()

        expected = np.array([0.99, 1.99], dtype=np.float64)
        np.testing.assert_allclose(param.numpy(), expected, rtol=1e-10)

    def test_asgd_2d_param(self):
        """Test ASGD with 2D parameter tensor.

        测试 2D 参数张量的 ASGD 优化。
        """
        param = paddle.create_parameter(shape=[2, 3], dtype='float32')
        param.set_value(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.01, batch_num=1, parameters=[param]
        )

        loss = param.sum()
        loss.backward()
        optimizer.step()

        # grad = ones(2,3), n=1
        # param -= 0.01 * ones
        expected = np.array(
            [[0.99, 1.99, 2.99], [3.99, 4.99, 5.99]], dtype=np.float32
        )
        np.testing.assert_allclose(param.numpy(), expected, rtol=1e-5)

    def test_asgd_different_grad(self):
        """Test ASGD with different gradient patterns.

        测试不同梯度模式的 ASGD 优化。
        """
        param = paddle.create_parameter(shape=[4], dtype='float32')
        param.set_value(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.1, batch_num=1, parameters=[param]
        )

        # Create loss with different gradients for each element
        loss = (param * paddle.to_tensor([1.0, 2.0, 3.0, 4.0])).sum()
        loss.backward()
        optimizer.step()

        # grad = [1,2,3,4]
        # param_out = [1,2,3,4] - 0.1 * [1,2,3,4] = [0.9, 1.8, 2.7, 3.6]
        expected = np.array([0.9, 1.8, 2.7, 3.6], dtype=np.float32)
        np.testing.assert_allclose(param.numpy(), expected, rtol=1e-5)

    def test_asgd_convergence(self):
        """Test ASGD convergence towards minimum of a simple quadratic.

        测试 ASGD 在简单二次函数上向最小值收敛。
        Minimize (x-5)^2, minimum at x=5, gradient = 2*(x-5).
        """
        param = paddle.create_parameter(shape=[1], dtype='float32')
        param.set_value(np.array([0.0], dtype=np.float32))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.1, batch_num=1, parameters=[param]
        )

        for _ in range(50):
            loss = (param - 5.0) ** 2
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # Should converge near x=5
        np.testing.assert_allclose(param.numpy(), [5.0], atol=0.5)

    def test_asgd_state_dict_keys(self):
        """Test ASGD state dict contains expected keys (d, y, m counter).

        测试 ASGD 状态字典包含预期的键（d、y、m 计数器）。
        """
        param = paddle.create_parameter(shape=[2], dtype='float32')
        param.set_value(np.array([1.0, 2.0], dtype=np.float32))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.01, batch_num=1, parameters=[param]
        )

        loss = param.sum()
        loss.backward()
        optimizer.step()

        # Save state
        state_dict = optimizer.state_dict()
        keys = list(state_dict.keys())
        # State dict should have d, y, and m (counter) for the parameter
        self.assertTrue(
            any('d_0' in k for k in keys), f"No 'd' key found in {keys}"
        )
        self.assertTrue(
            any('y_0' in k for k in keys), f"No 'y' key found in {keys}"
        )
        self.assertTrue(
            any('m_0' in k for k in keys),
            f"No 'm' (counter) key found in {keys}",
        )

    def test_asgd_m_counter_increments(self):
        """Test that ASGD m counter increments by 1 each step.

        测试 ASGD 的 m 计数器每步递增 1。
        """
        param = paddle.create_parameter(shape=[1], dtype='float32')
        param.set_value(np.array([100.0], dtype=np.float32))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.1, batch_num=1, parameters=[param]
        )

        for step in range(4):
            optimizer.clear_grad()
            loss = param.sum()
            loss.backward()
            optimizer.step()
            state = optimizer.state_dict()
            m_key = next(k for k in state if 'm_0' in k)
            m_val = state[m_key].numpy()[0]
            self.assertEqual(
                m_val,
                step + 1,
                f"Step {step + 1}: expected m={step + 1}, got {m_val}",
            )

    def test_asgd_accumulated_grad(self):
        """Test ASGD with gradient accumulation (no clear_grad between steps).

        测试不带梯度清除的 ASGD（梯度累积）。
        Without clear_grad, gradients accumulate across steps.
        """
        param = paddle.create_parameter(shape=[2], dtype='float32')
        param.set_value(np.array([10.0, 10.0], dtype=np.float32))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.1, batch_num=1, parameters=[param]
        )

        # Step 1: grad=ones
        loss1 = param.sum()
        loss1.backward()
        optimizer.step()
        param_after_1 = param.numpy().copy()

        # Step 2 without clear: grad accumulates (grad of 9.9+9.9 = 2)
        loss2 = param.sum()
        loss2.backward()
        optimizer.step()
        param_after_2 = param.numpy()

        # Without clear_grad, grad is larger so update should be larger
        delta_1 = 10.0 - param_after_1[0]
        delta_2 = param_after_1[0] - param_after_2[0]
        # Accumulated grad leads to larger or equal step
        self.assertGreaterEqual(delta_2, delta_1 - 1e-6)

    def test_asgd_initial_d_y_zeros(self):
        """Test that initial d and y are zero vectors.

        测试初始 d 和 y 为零向量。
        """
        param = paddle.create_parameter(shape=[3], dtype='float32')
        param.set_value(np.array([1.0, 1.0, 1.0], dtype=np.float32))
        optimizer = paddle.optimizer.ASGD(
            learning_rate=0.01, batch_num=1, parameters=[param]
        )

        loss = param.sum()
        loss.backward()
        optimizer.step()

        state = optimizer.state_dict()
        d_key = next(k for k in state if 'd_0' in k)
        d_val = state[d_key].numpy()
        # d = 0 - 0 + grad = grad = [1,1,1]
        np.testing.assert_allclose(d_val, [1.0, 1.0, 1.0], rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
