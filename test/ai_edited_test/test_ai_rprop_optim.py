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
# Target file: paddle/optimizer/rprop.py
# Coverage target: 85.7% -> improve coverage on uncovered lines
# 测试 Rprop 优化器的各项功能，包括参数验证、learning_rate_range、etas、multi_precision 等
# Tests for Rprop optimizer covering parameter validation, learning_rate_range, etas, multi_precision, etc.

import unittest

import paddle
from paddle import nn


class TestRpropOptimizer(unittest.TestCase):
    """Rprop 优化器测试类 / Rprop optimizer test class"""

    def setUp(self):
        """测试前置准备 / Set up test fixtures"""
        paddle.set_device("cpu")

    def tearDown(self):
        """测试后清理 / Clean up after tests"""
        paddle.set_device("cpu")

    def test_rprop_basic_step(self):
        """测试 Rprop 优化器基本训练步骤 / Test Rprop basic training step"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        rprop = paddle.optimizer.Rprop(
            learning_rate=0.01,
            learning_rate_range=(0.0001, 0.1),
            parameters=linear.parameters(),
            etas=(0.5, 1.2),
        )
        loss.backward()
        rprop.step()
        rprop.clear_grad()

    def test_rprop_invalid_learning_rate(self):
        """测试 Rprop 无效学习率 / Test Rprop invalid learning rate"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Rprop(
                learning_rate=None,
                learning_rate_range=(0.0001, 0.1),
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_rprop_invalid_lr_range(self):
        """测试 Rprop 无效学习率范围 / Test Rprop invalid learning rate range"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Rprop(
                learning_rate=0.01,
                learning_rate_range=(0.1, 0.001),
                parameters=paddle.randn([10, 10], dtype="float32"),
            )

    def test_rprop_invalid_etas(self):
        """测试 Rprop 无效 etas / Test Rprop invalid etas"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Rprop(
                learning_rate=0.01,
                learning_rate_range=(0.0001, 0.1),
                parameters=paddle.randn([10, 10], dtype="float32"),
                etas=(1.0, 0.5),
            )

    def test_rprop_accumulator_strings(self):
        """测试 Rprop 累加器字符串常量 / Test Rprop accumulator string constants"""
        self.assertEqual(paddle.optimizer.Rprop._prevs_acc_str, "prevs")
        self.assertEqual(
            paddle.optimizer.Rprop._learning_rates_acc_str, "learning_rates"
        )

    def test_rprop_multiple_steps(self):
        """测试 Rprop 多步训练 / Test Rprop multiple training steps"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        rprop = paddle.optimizer.Rprop(
            learning_rate=0.01,
            learning_rate_range=(0.0001, 0.1),
            parameters=linear.parameters(),
            etas=(0.5, 1.2),
        )
        for _ in range(5):
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()
            rprop.step()
            rprop.clear_grad()

    def test_rprop_multi_precision(self):
        """测试 Rprop multi_precision / Test Rprop multi_precision"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        rprop = paddle.optimizer.Rprop(
            learning_rate=0.01,
            learning_rate_range=(0.0001, 0.1),
            parameters=linear.parameters(),
            multi_precision=True,
        )
        loss.backward()
        rprop.step()
        rprop.clear_grad()

    def test_rprop_default_lr_range(self):
        """测试 Rprop 默认学习率范围 / Test Rprop default learning rate range"""
        linear = nn.Linear(10, 10)
        rprop = paddle.optimizer.Rprop(
            learning_rate=0.01,
            parameters=linear.parameters(),
        )
        self.assertEqual(rprop._learning_rate_range, [(1e-5, 50)])

    def test_rprop_default_etas(self):
        """测试 Rprop 默认 etas / Test Rprop default etas"""
        linear = nn.Linear(10, 10)
        rprop = paddle.optimizer.Rprop(
            learning_rate=0.01,
            parameters=linear.parameters(),
        )
        self.assertEqual(rprop._etas, [(0.5, 1.2)])

    def test_rprop_state_dict(self):
        """测试 Rprop 状态字典 / Test Rprop state_dict"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        rprop = paddle.optimizer.Rprop(
            learning_rate=0.01,
            learning_rate_range=(0.0001, 0.1),
            parameters=linear.parameters(),
        )
        loss.backward()
        rprop.step()
        state = rprop.state_dict()
        self.assertIsInstance(state, dict)

    def test_rprop_name(self):
        """测试 Rprop name 参数 / Test Rprop name parameter"""
        linear = nn.Linear(10, 10)
        rprop = paddle.optimizer.Rprop(
            learning_rate=0.01,
            learning_rate_range=(0.0001, 0.1),
            parameters=linear.parameters(),
            name="test_rprop",
        )
        self.assertEqual(rprop.type, "rprop")


if __name__ == "__main__":
    unittest.main()
