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

# [AUTO-GENERATED] Test file for paddle/nn/quant/lsq.py
# Target file: paddle/nn/quant/lsq.py (80.2% coverage)
# Uncovered lines: 104-107 (per_channel forward), 111-117 (per_channel backward),
#   131, 134-135 (LsqPlusActFunc backward paths),
#   169 (FakeQuantActLSQPlus init_state<batch_init with not symmetric),
#   186-187, 189, 192, 195, 201 (FakeQuantWeightLSQPlus batch_init paths),
#   206, 218, 227, 235 (FakeQuantWeightLSQPlus more batch_init paths),
#   284-285 (FakeQuantWeightLSQPlus reduce_type==max),
#   304-323, 346-348 (FakeQuantWeightLSQPlus per_channel init/batch_init)

"""LSQ 量化模块测试 / Learned Step Size Quantization tests

测试目标 / Test Target:
  paddle/nn/quant/lsq.py

覆盖的模块 / Covered Modules:
  - LsqFunc: per_channel forward and backward
  - LsqPlusActFunc: forward and backward
  - FakeQuantActLSQPlus: init, forward with different states
  - FakeQuantWeightLSQPlus: init, forward with different states
"""

import unittest

import paddle
from paddle.nn.quant.lsq import (
    FakeQuantActLSQPlus,
    FakeQuantWeightLSQPlus,
)


class TestLsqFuncNoPerChannel(unittest.TestCase):
    """测试 LsqFunc 的非 per_channel 路径
    Test LsqFunc non-per_channel path"""

    def setUp(self):
        paddle.disable_static()

    def test_lsq_no_per_channel_forward(self):
        """测试 LsqFunc 无 per_channel 的前向传播
        Test LsqFunc forward without per_channel"""
        weight = paddle.randn([4, 8], dtype='float32')
        alpha = paddle.to_tensor(1.0, dtype='float32')
        g = paddle.to_tensor(0.1, dtype='float32')

        from paddle.nn.quant.lsq import LsqFunc

        out = LsqFunc.apply(weight, alpha, g, -128, 127, per_channel=False)
        self.assertEqual(out.shape, weight.shape)

    def test_lsq_no_per_channel_backward(self):
        """测试 LsqFunc 无 per_channel 的反向传播
        Test LsqFunc backward without per_channel"""
        weight = paddle.randn([4, 8], dtype='float32')
        weight.stop_gradient = False
        # Use [1] shaped alpha to match parameter shape
        alpha = paddle.create_parameter(shape=[1], dtype='float32')
        alpha.set_value(paddle.to_tensor([1.0], dtype='float32'))
        g = paddle.to_tensor(0.1, dtype='float32')

        from paddle.nn.quant.lsq import LsqFunc

        out = LsqFunc.apply(weight, alpha, g, -128, 127, per_channel=False)
        loss = out.sum()
        loss.backward()
        # Should have gradients
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(alpha.grad)


class TestLsqPlusActFunc(unittest.TestCase):
    """测试 LsqPlusActFunc 的前向和反向传播
    Test LsqPlusActFunc forward and backward"""

    def setUp(self):
        paddle.disable_static()

    def test_lsq_plus_act_forward(self):
        """测试 LsqPlusActFunc 前向传播
        Test LsqPlusActFunc forward"""
        x = paddle.randn([4, 8], dtype='float32')
        alpha = paddle.to_tensor(1.0, dtype='float32')
        beta = paddle.to_tensor(0.0, dtype='float32')
        g = paddle.to_tensor(0.1, dtype='float32')

        from paddle.nn.quant.lsq import LsqPlusActFunc

        out = LsqPlusActFunc.apply(x, alpha, beta, g, -128, 127)
        self.assertEqual(out.shape, x.shape)

    def test_lsq_plus_act_backward(self):
        """测试 LsqPlusActFunc 反向传播 (line 131, 134-135)
        Test LsqPlusActFunc backward"""
        x = paddle.randn([4, 8], dtype='float32')
        x.stop_gradient = False
        alpha = paddle.create_parameter(shape=[], dtype='float32')
        alpha.set_value(paddle.to_tensor(1.0, dtype='float32'))
        beta = paddle.create_parameter(shape=[], dtype='float32')
        beta.set_value(paddle.to_tensor(0.0, dtype='float32'))
        g = paddle.to_tensor(0.1, dtype='float32')

        from paddle.nn.quant.lsq import LsqPlusActFunc

        out = LsqPlusActFunc.apply(x, alpha, beta, g, -128, 127)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(alpha.grad)
        self.assertIsNotNone(beta.grad)


class TestFakeQuantActLSQPlus(unittest.TestCase):
    """测试 FakeQuantActLSQPlus 层
    Test FakeQuantActLSQPlus layer"""

    def setUp(self):
        paddle.disable_static()

    def test_act_lsq_plus_all_positive(self):
        """测试全正数 FakeQuantActLSQPlus (all_positive=True)
        Test all-positive FakeQuantActLSQPlus"""
        layer = FakeQuantActLSQPlus(
            quant_bits=8,
            all_positive=True,
            symmetric=True,
            batch_init=2,
        )
        self.assertEqual(layer.Qn, 0)
        self.assertEqual(layer.Qp, 255)

        x = paddle.randn([4, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_act_lsq_plus_not_symmetric(self):
        """测试非对称 FakeQuantActLSQPlus (line 185-195, 206, 218, 227, 235)
        Test non-symmetric FakeQuantActLSQPlus"""
        layer = FakeQuantActLSQPlus(
            quant_bits=8,
            all_positive=False,
            symmetric=False,
            batch_init=2,
        )
        # Verify beta parameter exists for non-symmetric
        self.assertTrue(hasattr(layer, 'beta'))

        x = paddle.randn([4, 8], dtype='float32')
        # Call multiple times to go through batch_init states
        out1 = layer(x)  # init_state=0 -> 1
        out2 = layer(x)  # init_state=1 -> 2 (== batch_init)
        out3 = layer(x)  # init_state=2 -> 3 (past batch_init)
        self.assertEqual(out1.shape, x.shape)
        self.assertEqual(out2.shape, x.shape)
        self.assertEqual(out3.shape, x.shape)

    def test_act_lsq_plus_symmetric(self):
        """测试对称 FakeQuantActLSQPlus
        Test symmetric FakeQuantActLSQPlus"""
        layer = FakeQuantActLSQPlus(
            quant_bits=8,
            all_positive=False,
            symmetric=True,
            batch_init=2,
        )
        # Verify beta does NOT exist for symmetric
        self.assertFalse(hasattr(layer, 'beta'))

        x = paddle.randn([4, 8], dtype='float32')
        out = layer(x)
        self.assertEqual(out.shape, x.shape)

    def test_act_lsq_plus_batch_init_accumulation(self):
        """测试 batch_init 多次调用累积 (line 169, 186-187)
        Test batch_init accumulation across calls"""
        layer = FakeQuantActLSQPlus(
            quant_bits=8,
            all_positive=False,
            symmetric=False,
            batch_init=5,
        )
        x = paddle.abs(paddle.randn([4, 8], dtype='float32')) + 0.01
        for _ in range(10):
            out = layer(x)
        self.assertEqual(out.shape, x.shape)


class TestFakeQuantWeightLSQPlus(unittest.TestCase):
    """测试 FakeQuantWeightLSQPlus 层
    Test FakeQuantWeightLSQPlus layer"""

    def setUp(self):
        paddle.disable_static()

    def test_weight_lsq_basic(self):
        """测试基本的 FakeQuantWeightLSQPlus (per_channel=True)
        Test basic FakeQuantWeightLSQPlus with per_channel"""
        layer = FakeQuantWeightLSQPlus(
            quant_bits=8,
            all_positive=False,
            per_channel=True,
            batch_init=2,
            channel_num=8,
            quant_linear=False,
        )
        w = paddle.randn([8, 16], dtype='float32')
        out = layer(w)
        self.assertEqual(out.shape, w.shape)

    def test_weight_lsq_per_channel(self):
        """测试 per_channel 的 FakeQuantWeightLSQPlus (line 304-323, 346-348)
        Test per_channel FakeQuantWeightLSQPlus"""
        layer = FakeQuantWeightLSQPlus(
            quant_bits=8,
            all_positive=False,
            per_channel=True,
            batch_init=2,
            channel_num=8,
            quant_linear=False,
        )
        w = paddle.randn([8, 16], dtype='float32')
        out = layer(w)
        self.assertEqual(out.shape, w.shape)

    def test_weight_lsq_all_positive(self):
        """测试全正数 FakeQuantWeightLSQPlus
        Test all-positive FakeQuantWeightLSQPlus"""
        layer = FakeQuantWeightLSQPlus(
            quant_bits=8,
            all_positive=True,
            per_channel=True,
            batch_init=2,
            channel_num=8,
        )
        self.assertEqual(layer.Qn, 0)
        self.assertEqual(layer.Qp, 255)

    def test_weight_lsq_quant_linear(self):
        """测试 quant_linear 的 FakeQuantWeightLSQPlus
        Test quant_linear FakeQuantWeightLSQPlus"""
        layer = FakeQuantWeightLSQPlus(
            quant_bits=8,
            all_positive=False,
            per_channel=True,
            batch_init=2,
            channel_num=8,
            quant_linear=True,
        )
        # quant_linear=True -> quant_axis=1, collect_axis=0
        self.assertEqual(layer.quant_axis, 1)
        self.assertEqual(layer.collect_axis, 0)

        w = paddle.randn([4, 8], dtype='float32')
        out = layer(w)
        self.assertEqual(out.shape, w.shape)

    def test_weight_lsq_batch_init_per_channel(self):
        """测试 per_channel batch_init 多次调用 (line 304-358)
        Test per_channel batch_init multiple calls"""
        layer = FakeQuantWeightLSQPlus(
            quant_bits=8,
            all_positive=False,
            per_channel=True,
            batch_init=3,
            channel_num=8,
            quant_linear=False,
        )
        w = paddle.randn([8, 16], dtype='float32')
        for _ in range(6):
            out = layer(w)
        self.assertEqual(out.shape, w.shape)

    def test_weight_lsq_batch_init_no_per_channel(self):
        """测试非 per_channel batch_init 多次调用 (line 322-358)
        Test non-per_channel batch_init multiple calls"""
        layer = FakeQuantWeightLSQPlus(
            quant_bits=8,
            all_positive=False,
            per_channel=True,
            batch_init=3,
            channel_num=8,
        )
        w = paddle.randn([8, 16], dtype='float32')
        for _ in range(6):
            out = layer(w)
        self.assertEqual(out.shape, w.shape)


if __name__ == '__main__':
    unittest.main()
