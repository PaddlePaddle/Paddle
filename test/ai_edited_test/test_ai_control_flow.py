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
动态图控制流测试 / Dynamic Graph Control Flow Tests

测试目标 / Test Target:
  paddle 动态图控制流

覆盖的模块 / Covered Modules:
  - paddle.cond: 条件执行
  - paddle.while_loop: 循环执行
  - paddle.jit.to_static: 动转静
  - 条件分支模型

作用 / Purpose:
  补充动态图控制流API的测试，提升覆盖率。
"""

import unittest

import paddle
from paddle import nn

paddle.disable_static()


class TestPaddleCond(unittest.TestCase):
    """测试条件控制流 / Test conditional control flow"""

    def test_cond_true(self):
        """测试条件为True / Test cond when true"""
        x = paddle.to_tensor(True)
        result = paddle.static.nn.cond(
            x, lambda: paddle.to_tensor([1.0]), lambda: paddle.to_tensor([0.0])
        )
        self.assertAlmostEqual(float(result.item()), 1.0)

    def test_cond_false(self):
        """测试条件为False / Test cond when false"""
        x = paddle.to_tensor(False)
        result = paddle.static.nn.cond(
            x, lambda: paddle.to_tensor([1.0]), lambda: paddle.to_tensor([0.0])
        )
        self.assertAlmostEqual(float(result.item()), 0.0)

    def test_cond_with_computation(self):
        """测试带计算的条件 / Test cond with computation"""
        x = paddle.to_tensor(3.0)
        cond = x > 2.0
        result = paddle.static.nn.cond(cond, lambda: x * 2, lambda: x * 0.5)
        self.assertAlmostEqual(float(result.item()), 6.0, places=5)


class TestWhileLoop(unittest.TestCase):
    """测试while循环 / Test while loop"""

    def test_while_loop_basic(self):
        """测试基本while循环 / Test basic while loop"""
        i = paddle.zeros([1], dtype='int64')
        limit = paddle.to_tensor([5], dtype='int64')

        def cond(i):
            return paddle.less_than(i, limit)

        def body(i):
            return [i + 1]

        out = paddle.static.nn.while_loop(cond, body, [i])
        self.assertEqual(int(out[0].item()), 5)


class TestDynamicShapeModel(unittest.TestCase):
    """测试动态形状模型 / Test dynamic shape model"""

    def test_variable_length_batch(self):
        """测试可变长度批次 / Test variable length batch"""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

        # Test with different batch sizes
        for batch_size in [1, 4, 16]:
            x = paddle.randn([batch_size, 4])
            output = model(x)
            self.assertEqual(output.shape[0], batch_size)
            self.assertEqual(output.shape[1], 2)

    def test_conditional_model(self):
        """测试条件模型 / Test conditional model"""

        class ConditionalNet(nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.fc2 = nn.Linear(4, 8)

            def forward(self, x, use_path1):
                if use_path1:
                    return self.fc1(x)
                else:
                    return self.fc2(x)

        model = ConditionalNet()
        x = paddle.randn([4, 4])

        out1 = model(x, True)
        out2 = model(x, False)

        self.assertEqual(out1.shape, [4, 8])
        self.assertEqual(out2.shape, [4, 8])


class TestToStaticConversion(unittest.TestCase):
    """测试动转静转换 / Test dynamic to static conversion"""

    def test_to_static_function(self):
        """测试函数动转静 / Test function to static"""

        @paddle.jit.to_static
        def linear_func(x, w, b):
            return paddle.matmul(x, w) + b

        x = paddle.randn([4, 4])
        w = paddle.randn([4, 2])
        b = paddle.zeros([2])
        result = linear_func(x, w, b)
        self.assertEqual(result.shape, [4, 2])

    def test_to_static_model(self):
        """测试模型动转静 / Test model to static"""

        class SimpleModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            @paddle.jit.to_static
            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()
        x = paddle.randn([4, 4])
        output = model(x)
        self.assertEqual(output.shape, [4, 2])


if __name__ == '__main__':
    unittest.main()
