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
# Target file: paddle/optimizer/radam.py
# Coverage target: 87.2% -> improve coverage on uncovered lines
# 测试 RAdam 优化器的各项功能，包括参数验证、参数组、类型检查等
# Tests for RAdam optimizer covering parameter validation, param groups, type checks, etc.

import unittest

import paddle
from paddle import nn


class TestRAdamOptimizer(unittest.TestCase):
    """RAdam 优化器测试类 / RAdam optimizer test class"""

    def setUp(self):
        """测试前置准备 / Set up test fixtures"""
        paddle.set_device("cpu")

    def tearDown(self):
        """测试后清理 / Clean up after tests"""
        paddle.set_device("cpu")

    def test_radam_basic_step(self):
        """测试 RAdam 优化器基本训练步骤 / Test RAdam basic training step"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        radam = paddle.optimizer.RAdam(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        loss.backward()
        radam.step()
        radam.clear_grad()

    def test_radam_with_param_groups(self):
        """测试 RAdam 使用参数组 / Test RAdam with parameter groups"""
        linear_1 = nn.Linear(10, 10)
        linear_2 = nn.Linear(10, 10)
        x = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
        out = linear_1(x)
        out = linear_2(out)
        loss = paddle.mean(out)
        radam = paddle.optimizer.RAdam(
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
        radam.step()
        radam.clear_grad()

    def test_radam_invalid_learning_rate(self):
        """测试 RAdam 无效学习率 / Test RAdam invalid learning rate"""
        with self.assertRaises(ValueError):
            paddle.optimizer.RAdam(
                learning_rate=-0.1,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_radam_invalid_epsilon(self):
        """测试 RAdam 无效 epsilon / Test RAdam invalid epsilon"""
        with self.assertRaises(ValueError):
            paddle.optimizer.RAdam(
                learning_rate=0.1,
                epsilon=-1.0,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_radam_invalid_beta1(self):
        """测试 RAdam 无效 beta1 / Test RAdam invalid beta1"""
        with self.assertRaises(ValueError):
            paddle.optimizer.RAdam(
                learning_rate=0.1,
                beta1=1.5,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_radam_invalid_beta2(self):
        """测试 RAdam 无效 beta2 / Test RAdam invalid beta2"""
        with self.assertRaises(ValueError):
            paddle.optimizer.RAdam(
                learning_rate=0.1,
                beta2=-0.5,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_radam_with_weight_decay(self):
        """测试 RAdam 权重衰减 / Test RAdam with weight decay"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        radam = paddle.optimizer.RAdam(
            learning_rate=0.1,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        loss.backward()
        radam.step()
        radam.clear_grad()

    def test_radam_default_dict(self):
        """测试 RAdam 默认参数字典 / Test RAdam default parameter dictionary"""
        linear = nn.Linear(10, 10)
        radam = paddle.optimizer.RAdam(
            learning_rate=0.1,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            parameters=linear.parameters(),
        )
        self.assertIn("beta1", radam._default_dict)
        self.assertIn("beta2", radam._default_dict)
        self.assertIn("epsilon", radam._default_dict)

    def test_radam_accumulator_strings(self):
        """测试 RAdam 累加器字符串常量 / Test RAdam accumulator string constants"""
        self.assertEqual(paddle.optimizer.RAdam._beta1_pow_acc_str, "beta1_pow")
        self.assertEqual(paddle.optimizer.RAdam._beta2_pow_acc_str, "beta2_pow")
        self.assertEqual(paddle.optimizer.RAdam._rho_acc_str, "rho")
        self.assertEqual(paddle.optimizer.RAdam._moment1_acc_str, "moment1")
        self.assertEqual(paddle.optimizer.RAdam._moment2_acc_str, "moment2")

    def test_radam_state_dict(self):
        """测试 RAdam 状态字典 / Test RAdam state_dict"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        radam = paddle.optimizer.RAdam(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        loss.backward()
        radam.step()
        state = radam.state_dict()
        self.assertIsInstance(state, dict)
        has_moment = any("moment1" in k for k in state.keys())
        self.assertTrue(has_moment)

    def test_radam_multi_precision(self):
        """测试 RAdam multi_precision / Test RAdam multi_precision"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        radam = paddle.optimizer.RAdam(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        radam._multi_precision = True
        loss.backward()
        radam.step()
        radam.clear_grad()

    def test_radam_boundary_values(self):
        """测试 RAdam 边界值 / Test RAdam boundary values"""
        linear = nn.Linear(10, 10)
        # beta1 = 0 should be valid
        radam1 = paddle.optimizer.RAdam(
            learning_rate=0.1,
            beta1=0.0,
            parameters=linear.parameters(),
        )
        self.assertEqual(radam1._beta1, 0.0)

        # beta2 = 0 should be valid
        linear2 = nn.Linear(10, 10)
        radam2 = paddle.optimizer.RAdam(
            learning_rate=0.1,
            beta2=0.0,
            parameters=linear2.parameters(),
        )
        self.assertEqual(radam2._beta2, 0.0)

        # epsilon = 0 should be valid
        linear3 = nn.Linear(10, 10)
        radam3 = paddle.optimizer.RAdam(
            learning_rate=0.1,
            epsilon=0.0,
            parameters=linear3.parameters(),
        )
        self.assertEqual(radam3._epsilon, 0.0)

    def test_radam_update_param_group(self):
        """测试 RAdam 参数组更新 / Test RAdam parameter group update"""
        linear_1 = nn.Linear(10, 10)
        linear_2 = nn.Linear(10, 10)
        x = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
        out = linear_1(x)
        out = linear_2(out)
        loss = paddle.mean(out)
        radam = paddle.optimizer.RAdam(
            learning_rate=0.1,
            parameters=[
                {"params": linear_1.parameters()},
                {
                    "params": linear_2.parameters(),
                    "epsilon": 1e-6,
                    "beta1": 0.85,
                    "beta2": 0.99,
                },
            ],
        )
        loss.backward()
        radam.step()
        radam.clear_grad()

    def test_radam_name(self):
        """测试 RAdam name 参数 / Test RAdam name parameter"""
        linear = nn.Linear(10, 10)
        radam = paddle.optimizer.RAdam(
            learning_rate=0.1,
            parameters=linear.parameters(),
            name="test_radam",
        )
        self.assertEqual(radam.type, "radam")


if __name__ == "__main__":
    unittest.main()
