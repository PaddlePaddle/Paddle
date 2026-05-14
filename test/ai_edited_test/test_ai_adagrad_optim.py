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
# Target file: paddle/optimizer/adagrad.py
# Coverage target: 83.9% -> improve coverage on uncovered lines
# 测试 Adagrad 优化器的各项功能，包括参数验证、参数组、初始累加器值等
# Tests for Adagrad optimizer covering parameter validation, param groups, initial accumulator value, etc.

import unittest

import paddle
from paddle import nn


class TestAdagradOptimizer(unittest.TestCase):
    """Adagrad 优化器测试类 / Adagrad optimizer test class"""

    def setUp(self):
        """测试前置准备 / Set up test fixtures"""
        paddle.set_device("cpu")

    def tearDown(self):
        """测试后清理 / Clean up after tests"""
        paddle.set_device("cpu")

    def test_adagrad_basic_step(self):
        """测试 Adagrad 优化器基本训练步骤 / Test Adagrad basic training step"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adagrad = paddle.optimizer.Adagrad(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        loss.backward()
        adagrad.step()
        adagrad.clear_grad()

    def test_adagrad_with_param_groups(self):
        """测试 Adagrad 使用参数组 / Test Adagrad with parameter groups"""
        linear_1 = nn.Linear(10, 10)
        linear_2 = nn.Linear(10, 10)
        x = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
        out = linear_1(x)
        out = linear_2(out)
        loss = paddle.mean(out)
        adagrad = paddle.optimizer.Adagrad(
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
        adagrad.step()
        adagrad.clear_grad()

    def test_adagrad_with_weight_decay(self):
        """测试 Adagrad 权重衰减 / Test Adagrad with weight decay"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adagrad = paddle.optimizer.Adagrad(
            learning_rate=0.1,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        loss.backward()
        adagrad.step()
        adagrad.clear_grad()

    def test_adagrad_initial_accumulator_value(self):
        """测试 Adagrad 初始累加器值 / Test Adagrad with initial accumulator value"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adagrad = paddle.optimizer.Adagrad(
            learning_rate=0.1,
            parameters=linear.parameters(),
            initial_accumulator_value=0.1,
        )
        loss.backward()
        adagrad.step()
        adagrad.clear_grad()

    def test_adagrad_default_dict(self):
        """测试 Adagrad 默认参数字典 / Test Adagrad default parameter dictionary"""
        linear = nn.Linear(10, 10)
        adagrad = paddle.optimizer.Adagrad(
            learning_rate=0.1,
            parameters=linear.parameters(),
            initial_accumulator_value=0.1,
        )
        self.assertIn("epsilon", adagrad._default_dict)
        self.assertIn("initial_accumulator_value", adagrad._default_dict)
        self.assertEqual(
            adagrad._default_dict["initial_accumulator_value"], 0.1
        )

    def test_adagrad_accumulator_strings(self):
        """测试 Adagrad 累加器字符串常量 / Test Adagrad accumulator string constants"""
        self.assertEqual(paddle.optimizer.Adagrad._moment_acc_str, "moment")

    def test_adagrad_state_dict(self):
        """测试 Adagrad 状态字典 / Test Adagrad state_dict"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adagrad = paddle.optimizer.Adagrad(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        loss.backward()
        adagrad.step()
        state = adagrad.state_dict()
        self.assertIsInstance(state, dict)
        has_moment = any("moment" in k for k in state.keys())
        self.assertTrue(has_moment)

    def test_adagrad_multi_precision(self):
        """测试 Adagrad multi_precision / Test Adagrad multi_precision"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adagrad = paddle.optimizer.Adagrad(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        adagrad._multi_precision = True
        loss.backward()
        adagrad.step()
        adagrad.clear_grad()

    def test_adagrad_update_param_group(self):
        """测试 Adagrad 参数组更新 / Test Adagrad parameter group update"""
        linear_1 = nn.Linear(10, 10)
        linear_2 = nn.Linear(10, 10)
        x = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
        out = linear_1(x)
        out = linear_2(out)
        loss = paddle.mean(out)
        adagrad = paddle.optimizer.Adagrad(
            learning_rate=0.1,
            parameters=[
                {"params": linear_1.parameters()},
                {
                    "params": linear_2.parameters(),
                    "initial_accumulator_value": 0.5,
                },
            ],
        )
        loss.backward()
        adagrad.step()
        adagrad.clear_grad()


if __name__ == "__main__":
    unittest.main()
