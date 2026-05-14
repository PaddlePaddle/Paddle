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
# Target source: paddle/phi/kernels/cpu/adam_kernel.cc
# Generated for exercising C++ CPU kernel: AdamDenseKernel, MergedAdamKernel
#
# 测试 Adam 优化器 CPU 内核
# Tests for Adam optimizer CPU kernel

import unittest

import numpy as np

import paddle


class TestAdamDenseKernelBasic(unittest.TestCase):
    """基本 Adam 优化器测试 / Basic Adam optimizer tests"""

    def setUp(self):
        """初始化测试环境 / Setup test environment"""
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        """清理测试环境 / Teardown test environment"""
        paddle.enable_static()

    def test_adam_single_step_float32(self):
        """测试 float32 类型单步 Adam 更新
        Test single step Adam update with float32
        """
        np.random.seed(42)
        param_np = np.random.randn(10, 5).astype("float32")
        grad_np = np.random.randn(10, 5).astype("float32")

        param = paddle.to_tensor(param_np, stop_gradient=False)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001,
            parameters=[param],
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
        )

        loss = paddle.sum(
            param * paddle.to_tensor(grad_np, stop_gradient=False)
        )
        # Manual gradient to simulate a simple scenario
        loss.backward()
        optimizer.step()

        # 参数应该已经更新 / Parameters should have been updated
        result = param.numpy()
        self.assertFalse(np.allclose(result, param_np, atol=0))
        self.assertEqual(result.shape, (10, 5))

    def test_adam_multi_step(self):
        """测试多步 Adam 优化更新
        Test multi-step Adam optimization updates
        """
        np.random.seed(123)
        w_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

        w = paddle.to_tensor(w_np, stop_gradient=False)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=[w], beta1=0.9, beta2=0.999
        )

        for _ in range(5):
            loss = paddle.sum(w**2)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        result = w.numpy()
        # After optimization, values should move toward zero
        norm_after = np.linalg.norm(result)
        norm_before = np.linalg.norm(w_np)
        self.assertLess(norm_after, norm_before)

    def test_adam_different_lr(self):
        """测试不同学习率下的 Adam 行为
        Test Adam behavior with different learning rates
        """
        np.random.seed(42)
        param_init = np.array([2.0, -2.0, 1.0, -1.0], dtype="float32")

        results = {}
        for lr in [0.001, 0.01, 0.1]:
            p = paddle.to_tensor(param_init.copy(), stop_gradient=False)
            opt = paddle.optimizer.Adam(learning_rate=lr, parameters=[p])
            for _ in range(10):
                loss = paddle.sum(p**2)
                loss.backward()
                opt.step()
                opt.clear_grad()
            results[lr] = np.linalg.norm(p.numpy())

        # Higher LR should converge faster (smaller norm)
        self.assertLess(results[0.1], results[0.01])
        self.assertLess(results[0.01], results[0.001])

    def test_adam_beta_parameters(self):
        """测试不同 beta 参数下的 Adam 行为
        Test Adam behavior with different beta parameters
        """
        np.random.seed(42)
        param_init = np.array([1.0, -1.0], dtype="float32")

        p = paddle.to_tensor(param_init.copy(), stop_gradient=False)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=[p],
            beta1=0.85,
            beta2=0.95,
        )

        for _ in range(20):
            loss = paddle.sum(p**2)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # Should converge toward zero
        self.assertLess(np.abs(p.numpy()).max(), np.abs(param_init).max())

    def test_adam_large_tensor(self):
        """测试大型张量的 Adam 更新
        Test Adam update with large tensor
        """
        np.random.seed(42)
        param_np = np.random.randn(100, 100).astype("float32")
        grad_np = np.random.randn(100, 100).astype("float32")

        param = paddle.to_tensor(param_np, stop_gradient=False)
        grad = paddle.to_tensor(grad_np)

        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=[param]
        )

        loss = paddle.sum(param * grad)
        loss.backward()
        optimizer.step()

        result = param.numpy()
        self.assertEqual(result.shape, (100, 100))
        self.assertFalse(np.allclose(result, param_np))


class TestAdamDenseKernelFloat64(unittest.TestCase):
    """float64 类型的 Adam 优化器测试
    Tests for Adam optimizer with float64 dtype
    """

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_adam_float64(self):
        """测试 float64 类型 Adam 更新
        Test Adam update with float64
        """
        np.random.seed(42)
        param_np = np.random.randn(5, 3).astype("float64")

        param = paddle.to_tensor(param_np, stop_gradient=False)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=[param]
        )

        loss = paddle.sum(param**2)
        loss.backward()
        optimizer.step()

        result = param.numpy()
        self.assertEqual(result.dtype, np.float64)
        self.assertFalse(np.allclose(result, param_np, atol=0))


class TestAdamDenseKernelEdgeCases(unittest.TestCase):
    """Adam 优化器边界情况测试
    Edge case tests for Adam optimizer
    """

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        paddle.enable_static()

    def test_adam_zero_gradient(self):
        """测试零梯度的 Adam 更新（参数不变）
        Test Adam update with zero gradient (params should not change significantly)
        """
        np.random.seed(42)
        param_np = np.random.randn(3, 3).astype("float32")

        param = paddle.to_tensor(param_np, stop_gradient=False)
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.001, parameters=[param]
        )

        # Zero gradient
        loss = paddle.sum(paddle.zeros_like(param) * param)
        loss.backward()
        optimizer.step()

        result = param.numpy()
        # With zero gradient, params should remain mostly unchanged
        np.testing.assert_allclose(result, param_np, atol=1e-6)

    def test_adam_very_small_lr(self):
        """测试极小学习率的 Adam 更新
        Test Adam update with very small learning rate
        """
        np.random.seed(42)
        param_np = np.array([1.0, 2.0], dtype="float32")

        param = paddle.to_tensor(param_np.copy(), stop_gradient=False)
        optimizer = paddle.optimizer.Adam(
            learning_rate=1e-10, parameters=[param]
        )

        for _ in range(5):
            loss = paddle.sum(param**2)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        # With very small LR, params should barely change
        np.testing.assert_allclose(param.numpy(), param_np, atol=1e-5)

    def test_adam_single_element(self):
        """测试单元素张量的 Adam 更新
        Test Adam update with single element tensor
        """
        param = paddle.to_tensor([5.0], dtype="float32", stop_gradient=False)
        optimizer = paddle.optimizer.Adam(learning_rate=0.1, parameters=[param])

        loss = param**2
        loss.backward()
        optimizer.step()

        result = param.numpy()
        self.assertEqual(result.shape, (1,))
        self.assertLess(abs(result[0]), 5.0)

    def test_adam_1d_tensor(self):
        """测试一维张量的 Adam 更新
        Test Adam update with 1D tensor
        """
        param = paddle.to_tensor(
            [1.0, 2.0, 3.0, 4.0], dtype="float32", stop_gradient=False
        )
        optimizer = paddle.optimizer.Adam(
            learning_rate=0.01, parameters=[param]
        )

        loss = paddle.sum(param**2)
        loss.backward()
        optimizer.step()

        result = param.numpy()
        self.assertEqual(result.shape, (4,))


if __name__ == "__main__":
    unittest.main()
