# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.loss
# 自动生成的单测，覆盖 paddle.nn.functional.loss 模块中未覆盖的代码

"""
测试模块：paddle.nn.functional.loss (log_loss, margin_ranking_loss, soft_margin_loss, multi_label_soft_margin_loss)
Test Module: paddle.nn.functional.loss

本测试覆盖以下功能：
This test covers the following functions:
1. log_loss - 对数损失 / Log loss function
2. margin_ranking_loss - 排序损失 / Margin ranking loss
3. soft_margin_loss - 软间隔损失 / Soft margin loss
4. multi_label_soft_margin_loss - 多标签软间隔损失 / Multi-label soft margin loss

覆盖的未覆盖行：166-181 (log_loss), margin_ranking_loss分支
"""

import unittest

import paddle
import paddle.nn.functional as F


class TestLogLoss(unittest.TestCase):
    """测试log_loss对数损失函数
    Test log_loss function"""

    def setUp(self):
        paddle.disable_static()

    def test_log_loss_basic(self):
        """基本log_loss / Basic log_loss"""
        input_data = paddle.to_tensor([[0.8], [0.3], [0.6]], dtype='float32')
        label_data = paddle.to_tensor([[1.0], [0.0], [1.0]], dtype='float32')
        loss = F.log_loss(input=input_data, label=label_data)
        self.assertEqual(list(loss.shape), [3, 1])
        self.assertTrue(float(loss.sum().numpy()) > 0)

    def test_log_loss_with_epsilon(self):
        """带epsilon的log_loss / Log loss with custom epsilon"""
        input_data = paddle.to_tensor([[0.99], [0.01]], dtype='float32')
        label_data = paddle.to_tensor([[1.0], [0.0]], dtype='float32')
        loss = F.log_loss(input=input_data, label=label_data, epsilon=1e-6)
        self.assertTrue(float(loss.sum().numpy()) > 0)


class TestMarginRankingLoss(unittest.TestCase):
    """测试margin_ranking_loss排序损失
    Test margin_ranking_loss"""

    def setUp(self):
        paddle.disable_static()

    def test_margin_ranking_loss_basic(self):
        """基本排序损失 / Basic margin ranking loss"""
        input1 = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        input2 = paddle.to_tensor([2.0, 1.0, 1.0], dtype='float32')
        label = paddle.to_tensor([1.0, 1.0, -1.0], dtype='float32')
        loss = F.margin_ranking_loss(input1, input2, label, margin=0.0)
        self.assertIsNotNone(loss)

    def test_margin_ranking_loss_with_margin(self):
        """带margin的排序损失 / Margin ranking loss with margin"""
        input1 = paddle.to_tensor([3.0, 2.0], dtype='float32')
        input2 = paddle.to_tensor([1.0, 4.0], dtype='float32')
        label = paddle.to_tensor([1.0, -1.0], dtype='float32')
        loss = F.margin_ranking_loss(input1, input2, label, margin=0.5)
        self.assertIsNotNone(loss)

    def test_margin_ranking_loss_reduction_none(self):
        """reduction=none / Margin ranking loss with no reduction"""
        input1 = paddle.randn([5], dtype='float32')
        input2 = paddle.randn([5], dtype='float32')
        label = paddle.sign(paddle.randn([5]))
        loss = F.margin_ranking_loss(input1, input2, label, reduction='none')
        self.assertEqual(list(loss.shape), [5])


class TestSoftMarginLoss(unittest.TestCase):
    """测试soft_margin_loss软间隔损失
    Test soft_margin_loss"""

    def setUp(self):
        paddle.disable_static()

    def test_soft_margin_loss_basic(self):
        """基本软间隔损失 / Basic soft margin loss"""
        input_data = paddle.to_tensor([0.5, -0.5, 1.0], dtype='float32')
        label = paddle.to_tensor([1.0, -1.0, 1.0], dtype='float32')
        loss = F.soft_margin_loss(input_data, label)
        self.assertIsNotNone(loss)
        self.assertTrue(float(loss.numpy()) > 0)

    def test_soft_margin_loss_reduction_none(self):
        """reduction=none / Soft margin loss with no reduction"""
        input_data = paddle.randn([3, 4], dtype='float32')
        label = paddle.sign(paddle.randn([3, 4]))
        loss = F.soft_margin_loss(input_data, label, reduction='none')
        self.assertEqual(list(loss.shape), [3, 4])

    def test_soft_margin_loss_reduction_sum(self):
        """reduction=sum / Soft margin loss with sum reduction"""
        input_data = paddle.randn([3, 4], dtype='float32')
        label = paddle.sign(paddle.randn([3, 4]))
        loss = F.soft_margin_loss(input_data, label, reduction='sum')
        self.assertEqual(list(loss.shape), [])


class TestMultiLabelSoftMarginLoss(unittest.TestCase):
    """测试多标签软间隔损失
    Test multi_label_soft_margin_loss"""

    def setUp(self):
        paddle.disable_static()

    def test_multi_label_basic(self):
        """基本多标签损失 / Basic multi-label loss"""
        input_data = paddle.to_tensor(
            [[0.5, -0.3, 0.8], [0.2, 0.1, -0.4]], dtype='float32'
        )
        label = paddle.to_tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype='float32'
        )
        loss = F.multi_label_soft_margin_loss(input_data, label)
        self.assertIsNotNone(loss)

    def test_multi_label_with_weight(self):
        """带权重的多标签损失 / Multi-label loss with weight"""
        input_data = paddle.randn([2, 4], dtype='float32')
        label = paddle.to_tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype='float32')
        weight = paddle.to_tensor([1.0, 2.0, 1.0, 2.0], dtype='float32')
        loss = F.multi_label_soft_margin_loss(input_data, label, weight=weight)
        self.assertIsNotNone(loss)


if __name__ == '__main__':
    unittest.main()
