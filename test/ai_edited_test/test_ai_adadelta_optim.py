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
# Target file: paddle/optimizer/adadelta.py
# Coverage target: 84.7% -> improve coverage on uncovered lines
# 测试 Adadelta 优化器的各项功能，包括参数验证、参数组、rho/epsilon 参数等
# Tests for Adadelta optimizer covering parameter validation, param groups, rho/epsilon params, etc.

import unittest

import paddle
from paddle import nn


class TestAdadeltaOptimizer(unittest.TestCase):
    """Adadelta 优化器测试类 / Adadelta optimizer test class"""

    def setUp(self):
        """测试前置准备 / Set up test fixtures"""
        paddle.set_device("cpu")

    def tearDown(self):
        """测试后清理 / Clean up after tests"""
        paddle.set_device("cpu")

    def test_adadelta_basic_step(self):
        """测试 Adadelta 优化器基本训练步骤 / Test Adadelta basic training step"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adadelta = paddle.optimizer.Adadelta(
            learning_rate=0.1,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        loss.backward()
        adadelta.step()
        adadelta.clear_grad()

    def test_adadelta_with_param_groups(self):
        """测试 Adadelta 使用参数组 / Test Adadelta with parameter groups"""
        linear_1 = nn.Linear(10, 10)
        linear_2 = nn.Linear(10, 10)
        x = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
        out = linear_1(x)
        out = linear_2(out)
        loss = paddle.mean(out)
        adadelta = paddle.optimizer.Adadelta(
            learning_rate=0.1,
            parameters=[
                {"params": linear_1.parameters()},
                {
                    "params": linear_2.parameters(),
                    "weight_decay": 0.001,
                    "learning_rate": 0.1,
                },
            ],
            weight_decay=0.01,
        )
        loss.backward()
        adadelta.step()
        adadelta.clear_grad()

    def test_adadelta_invalid_learning_rate(self):
        """测试 Adadelta 无效学习率 / Test Adadelta invalid learning rate"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Adadelta(
                learning_rate=None,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_adadelta_invalid_epsilon(self):
        """测试 Adadelta 无效 epsilon / Test Adadelta invalid epsilon"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Adadelta(
                learning_rate=0.1,
                epsilon=None,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_adadelta_invalid_rho(self):
        """测试 Adadelta 无效 rho / Test Adadelta invalid rho"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Adadelta(
                learning_rate=0.1,
                rho=None,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_adadelta_custom_rho_epsilon(self):
        """测试 Adadelta 自定义 rho 和 epsilon / Test Adadelta with custom rho and epsilon"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adadelta = paddle.optimizer.Adadelta(
            learning_rate=0.1,
            rho=0.9,
            epsilon=1e-7,
            parameters=linear.parameters(),
        )
        loss.backward()
        adadelta.step()
        adadelta.clear_grad()

    def test_adadelta_default_dict(self):
        """测试 Adadelta 默认参数字典 / Test Adadelta default parameter dictionary"""
        linear = nn.Linear(10, 10)
        adadelta = paddle.optimizer.Adadelta(
            learning_rate=0.1,
            rho=0.9,
            epsilon=1e-7,
            parameters=linear.parameters(),
        )
        self.assertIn("epsilon", adadelta._default_dict)
        self.assertIn("rho", adadelta._default_dict)
        self.assertEqual(adadelta._default_dict["rho"], 0.9)
        self.assertEqual(adadelta._default_dict["epsilon"], 1e-7)

    def test_adadelta_accumulator_strings(self):
        """测试 Adadelta 累加器字符串常量 / Test Adadelta accumulator string constants"""
        self.assertEqual(
            paddle.optimizer.Adadelta._avg_squared_grad_acc_str,
            "_avg_squared_grad",
        )
        self.assertEqual(
            paddle.optimizer.Adadelta._avg_squared_update_acc_str,
            "_avg_squared_update",
        )

    def test_adadelta_state_dict(self):
        """测试 Adadelta 状态字典 / Test Adadelta state_dict"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adadelta = paddle.optimizer.Adadelta(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        loss.backward()
        adadelta.step()
        state = adadelta.state_dict()
        self.assertIsInstance(state, dict)
        has_avg_grad = any("_avg_squared_grad" in k for k in state.keys())
        self.assertTrue(has_avg_grad)

    def test_adadelta_multi_precision(self):
        """测试 Adadelta multi_precision / Test Adadelta multi_precision"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adadelta = paddle.optimizer.Adadelta(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        adadelta._multi_precision = True
        loss.backward()
        adadelta.step()
        adadelta.clear_grad()

    def test_adadelta_update_param_group(self):
        """测试 Adadelta 参数组更新 / Test Adadelta parameter group update"""
        linear_1 = nn.Linear(10, 10)
        linear_2 = nn.Linear(10, 10)
        x = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
        out = linear_1(x)
        out = linear_2(out)
        loss = paddle.mean(out)
        adadelta = paddle.optimizer.Adadelta(
            learning_rate=0.1,
            parameters=[
                {"params": linear_1.parameters()},
                {
                    "params": linear_2.parameters(),
                    "epsilon": 1e-5,
                    "rho": 0.85,
                },
            ],
        )
        loss.backward()
        adadelta.step()
        adadelta.clear_grad()

    def test_adadelta_name(self):
        """测试 Adadelta name 参数 / Test Adadelta name parameter"""
        linear = nn.Linear(10, 10)
        adadelta = paddle.optimizer.Adadelta(
            learning_rate=0.1,
            parameters=linear.parameters(),
            name="test_adadelta",
        )
        self.assertEqual(adadelta.type, "adadelta")


if __name__ == "__main__":
    unittest.main()
