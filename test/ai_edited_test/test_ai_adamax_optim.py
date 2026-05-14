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
# Target file: paddle/optimizer/adamax.py
# Coverage target: 82.9% -> improve coverage on uncovered lines
# 测试 Adamax 优化器的各项功能，包括参数验证、参数组、类型检查等
# Tests for Adamax optimizer covering parameter validation, param groups, type checks, etc.

import unittest

import paddle
from paddle import nn


class TestAdamaxOptimizer(unittest.TestCase):
    """Adamax 优化器测试类 / Adamax optimizer test class"""

    def setUp(self):
        """测试前置准备 / Set up test fixtures"""
        paddle.set_device("cpu")

    def tearDown(self):
        """测试后清理 / Clean up after tests"""
        paddle.set_device("cpu")

    def test_adamax_basic_step(self):
        """测试 Adamax 优化器基本训练步骤 / Test Adamax basic training step"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adamax = paddle.optimizer.Adamax(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        loss.backward()
        adamax.step()
        adamax.clear_grad()

    def test_adamax_with_param_groups(self):
        """测试 Adamax 使用参数组 / Test Adamax with parameter groups"""
        linear_1 = nn.Linear(10, 10)
        linear_2 = nn.Linear(10, 10)
        x = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
        out = linear_1(x)
        out = linear_2(out)
        loss = paddle.mean(out)
        adamax = paddle.optimizer.Adamax(
            learning_rate=0.1,
            parameters=[
                {"params": linear_1.parameters()},
                {
                    "params": linear_2.parameters(),
                    "weight_decay": 0.001,
                    "learning_rate": 0.1,
                    "beta1": 0.8,
                },
            ],
            weight_decay=0.01,
            beta1=0.9,
        )
        loss.backward()
        adamax.step()
        adamax.clear_grad()

    def test_adamax_with_tensor_beta(self):
        """测试 Adamax 使用 Tensor beta / Test Adamax with Tensor beta"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")
        adamax = paddle.optimizer.Adamax(
            learning_rate=0.1,
            parameters=linear.parameters(),
            beta1=beta1,
            beta2=beta2,
            weight_decay=0.01,
        )
        loss.backward()
        adamax.step()
        adamax.clear_grad()

    def test_adamax_invalid_beta1(self):
        """测试 Adamax 无效 beta1 / Test Adamax invalid beta1"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Adamax(
                learning_rate=0.1,
                beta1=1.5,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_adamax_invalid_beta2(self):
        """测试 Adamax 无效 beta2 / Test Adamax invalid beta2"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Adamax(
                learning_rate=0.1,
                beta2=-0.5,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_adamax_invalid_epsilon(self):
        """测试 Adamax 无效 epsilon / Test Adamax invalid epsilon"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Adamax(
                learning_rate=0.1,
                epsilon=-1.0,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_adamax_default_dict(self):
        """测试 Adamax 默认参数字典 / Test Adamax default parameter dictionary"""
        linear = nn.Linear(10, 10)
        adamax = paddle.optimizer.Adamax(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        self.assertIn("beta1", adamax._default_dict)
        self.assertIn("beta2", adamax._default_dict)
        self.assertIn("epsilon", adamax._default_dict)

    def test_adamax_weight_decay(self):
        """测试 Adamax 权重衰减 / Test Adamax with weight decay"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adamax = paddle.optimizer.Adamax(
            learning_rate=0.1,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        loss.backward()
        adamax.step()
        adamax.clear_grad()

    def test_adamax_state_dict(self):
        """测试 Adamax 状态字典 / Test Adamax state_dict"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adamax = paddle.optimizer.Adamax(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        loss.backward()
        adamax.step()
        state = adamax.state_dict()
        self.assertIsInstance(state, dict)
        has_moment = any("moment" in k for k in state.keys())
        self.assertTrue(has_moment)

    def test_adamax_accumulator_strings(self):
        """测试 Adamax 累加器字符串常量 / Test Adamax accumulator string constants"""
        self.assertEqual(paddle.optimizer.Adamax._moment_acc_str, "moment")
        self.assertEqual(paddle.optimizer.Adamax._inf_norm_acc_str, "inf_norm")
        self.assertEqual(
            paddle.optimizer.Adamax._beta1_pow_acc_str, "beta1_pow_acc"
        )

    def test_adamax_beta1_boundary(self):
        """测试 Adamax beta1 边界值 / Test Adamax beta1 boundary value"""
        linear = nn.Linear(10, 10)
        # beta1 = 0 should be valid
        adamax = paddle.optimizer.Adamax(
            learning_rate=0.1,
            beta1=0.0,
            parameters=linear.parameters(),
        )
        self.assertEqual(adamax._beta1, 0.0)

    def test_adamax_beta2_boundary(self):
        """测试 Adamax beta2 边界值 / Test Adamax beta2 boundary value"""
        linear = nn.Linear(10, 10)
        # beta2 = 0 should be valid
        adamax = paddle.optimizer.Adamax(
            learning_rate=0.1,
            beta2=0.0,
            parameters=linear.parameters(),
        )
        self.assertEqual(adamax._beta2, 0.0)


if __name__ == "__main__":
    unittest.main()
