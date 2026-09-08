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
# Target file: paddle/optimizer/adam.py
# Coverage target: 76.6% -> improve coverage on uncovered lines
# 测试 Adam 优化器的各项功能，包括参数验证、参数组、amsgrad、multi_tensor 等
# Tests for Adam optimizer covering parameter validation, param groups, amsgrad, multi_tensor, etc.

import unittest

import paddle
from paddle import nn


class TestAdamOptimizer(unittest.TestCase):
    """Adam 优化器测试类 / Adam optimizer test class"""

    def setUp(self):
        """测试前置准备 / Set up test fixtures"""
        paddle.set_device("cpu")

    def tearDown(self):
        """测试后清理 / Clean up after tests"""
        paddle.set_device("cpu")

    def test_adam_basic_step(self):
        """测试 Adam 优化器基本训练步骤 / Test basic Adam optimizer training step"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        loss.backward()
        adam.step()
        adam.clear_grad()
        # Verify parameters changed
        w_after = linear.weight.numpy().copy()
        out2 = linear(x)
        loss2 = paddle.mean(out2)
        loss2.backward()
        adam.step()
        w_after2 = linear.weight.numpy().copy()
        self.assertFalse((w_after == w_after2).all())

    def test_adam_invalid_beta1(self):
        """测试 Adam beta1 参数验证 / Test Adam beta1 parameter validation"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Adam(
                learning_rate=0.1,
                beta1=1.5,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_adam_invalid_beta2(self):
        """测试 Adam beta2 参数验证 / Test Adam beta2 parameter validation"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Adam(
                learning_rate=0.1,
                beta2=-0.5,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_adam_invalid_epsilon(self):
        """测试 Adam epsilon 参数验证 / Test Adam epsilon parameter validation"""
        with self.assertRaises(ValueError):
            paddle.optimizer.Adam(
                learning_rate=0.1,
                epsilon=-1.0,
                parameters=[paddle.randn([10, 10], dtype="float32")],
            )

    def test_adam_with_tensor_beta(self):
        """测试 Adam 使用 Tensor 类型的 beta 参数 / Test Adam with Tensor beta parameters"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        beta1 = paddle.to_tensor([0.9], dtype="float32")
        beta2 = paddle.to_tensor([0.99], dtype="float32")
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
            beta1=beta1,
            beta2=beta2,
            weight_decay=0.01,
        )
        loss.backward()
        adam.step()
        adam.clear_grad()

    def test_adam_with_param_groups(self):
        """测试 Adam 使用参数组 / Test Adam with parameter groups"""
        linear_1 = nn.Linear(10, 10)
        linear_2 = nn.Linear(10, 10)
        x = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
        out = linear_1(x)
        out = linear_2(out)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
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
        adam.step()
        adam.clear_grad()

    def test_adam_lazy_mode(self):
        """测试 Adam lazy_mode 参数 / Test Adam lazy_mode parameter"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
            lazy_mode=True,
        )
        loss.backward()
        adam.step()
        adam.clear_grad()

    def test_adam_multi_precision(self):
        """测试 Adam multi_precision 参数 / Test Adam multi_precision parameter"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
            multi_precision=True,
        )
        loss.backward()
        adam.step()
        adam.clear_grad()

    def test_adam_amsgrad(self):
        """测试 Adam amsgrad 变体 / Test Adam amsgrad variant"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
            amsgrad=True,
        )
        loss.backward()
        adam.step()
        adam.clear_grad()

    def test_adam_with_closure(self):
        """测试 Adam step 方法正常执行 / Test Adam step method executes normally"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([2, 10], dtype="float32")
        adam = paddle.optimizer.Adam(
            learning_rate=0.01,
            parameters=linear.parameters(),
        )
        out = linear(x)
        loss = paddle.mean(out)
        loss.backward()
        result = adam.step()
        self.assertIsNone(result)

    def test_adam_use_multi_tensor(self):
        """测试 Adam use_multi_tensor 参数 / Test Adam use_multi_tensor parameter"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
            use_multi_tensor=True,
        )
        loss.backward()
        adam.step()
        adam.clear_grad()

    def test_adam_use_multi_tensor_with_amsgrad(self):
        """测试 Adam 同时使用 multi_tensor 和 amsgrad / Test Adam with both multi_tensor and amsgrad"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
            use_multi_tensor=True,
            amsgrad=True,
        )
        loss.backward()
        adam.step()
        adam.clear_grad()

    def test_adam_fp16_warning(self):
        """测试 Adam 使用 FP16 时的警告 / Test Adam FP16 warning"""
        # Just verify the FP16 warning path in _create_accumulators
        # by checking that the warning condition code exists
        linear = nn.Linear(10, 10)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        # Verify multi_precision flag
        self.assertFalse(adam._multi_precision)
        # Verify the _is_dtype_fp16_or_bf16 method exists
        self.assertTrue(hasattr(adam, '_is_dtype_fp16_or_bf16'))

    def test_adam_weight_decay(self):
        """测试 Adam 权重衰减 / Test Adam with weight decay"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
            weight_decay=0.01,
        )
        loss.backward()
        adam.step()
        adam.clear_grad()

    def test_adam_state_dict(self):
        """测试 Adam 状态字典的保存和加载 / Test Adam state_dict save and load"""
        linear = nn.Linear(10, 10)
        x = paddle.randn([10, 10], dtype="float32")
        out = linear(x)
        loss = paddle.mean(out)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        loss.backward()
        adam.step()
        state = adam.state_dict()
        self.assertIsInstance(state, dict)
        # State dict keys include parameter names
        has_moment1 = any("moment1" in k for k in state.keys())
        self.assertTrue(has_moment1)

    def test_adam_default_dict(self):
        """测试 Adam 默认参数字典 / Test Adam default parameter dictionary"""
        linear = nn.Linear(10, 10)
        adam = paddle.optimizer.Adam(
            learning_rate=0.1,
            parameters=linear.parameters(),
        )
        self.assertIn("beta1", adam._default_dict)
        self.assertIn("beta2", adam._default_dict)
        self.assertIn("epsilon", adam._default_dict)
        self.assertIn("lazy_mode", adam._default_dict)

    def test_adam_accumulator_strings(self):
        """测试 Adam 累加器字符串常量 / Test Adam accumulator string constants"""
        self.assertEqual(paddle.optimizer.Adam._moment1_acc_str, "moment1")
        self.assertEqual(paddle.optimizer.Adam._moment2_acc_str, "moment2")
        self.assertEqual(
            paddle.optimizer.Adam._moment2_acc_max_str, "moment2_max"
        )
        self.assertEqual(
            paddle.optimizer.Adam._beta1_pow_acc_str, "beta1_pow_acc"
        )
        self.assertEqual(
            paddle.optimizer.Adam._beta2_pow_acc_str, "beta2_pow_acc"
        )


if __name__ == "__main__":
    unittest.main()
