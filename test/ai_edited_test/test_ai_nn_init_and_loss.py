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

# [AUTO-GENERATED] Test file for paddle.nn.initializer.orthogonal and paddle.nn.functional.loss
# 覆盖模块: paddle/nn/initializer/orthogonal.py, paddle/nn/functional/loss.py
# 未覆盖行: orthogonal: 157,161,167,181,187,193,204,210,218,219,226,234,240,241,247,254,256,264,271; loss: 169,170,171,173,175,181,280,284,289,290,291,338,584,593,594,595,596,597,598,601,602,603,610,706,712,719,720,721,722,731
# Covered module: paddle/nn/initializer/orthogonal.py, paddle/nn/functional/loss.py
# Uncovered lines: orthogonal: 157-271; loss: 169-731

import unittest

import paddle
import paddle.nn.functional as F


class TestOrthogonalInit(unittest.TestCase):
    """测试正交初始化器
    Test orthogonal initializer"""

    def test_orthogonal_init_default(self):
        """测试默认参数的正交初始化
        Test orthogonal initializer with default parameters"""
        init = paddle.nn.initializer.Orthogonal()
        linear = paddle.nn.Linear(10, 10)
        init(linear.weight)
        self.assertEqual(linear.weight.shape, [10, 10])

    def test_orthogonal_init_with_gain(self):
        """测试带 gain 参数的正交初始化
        Test orthogonal initializer with gain"""
        init = paddle.nn.initializer.Orthogonal(gain=2.0)
        linear = paddle.nn.Linear(10, 10)
        init(linear.weight)
        self.assertEqual(linear.weight.shape, [10, 10])

    def test_orthogonal_init_rectangular(self):
        """测试矩形矩阵的正交初始化
        Test orthogonal initializer with rectangular matrix"""
        init = paddle.nn.initializer.Orthogonal()
        # For Linear(20, 10), weight shape is [20, 10] (in_features x out_features)
        linear = paddle.nn.Linear(20, 10)
        init(linear.weight)
        # Paddle Linear weight shape is [in_features, out_features]
        self.assertEqual(list(linear.weight.shape), [20, 10])


class TestBCELoss(unittest.TestCase):
    """测试二元交叉熵损失函数
    Test Binary Cross Entropy loss function"""

    def test_bce_loss_basic(self):
        """测试基本的 BCELoss
        Test basic BCELoss"""
        input = paddle.to_tensor([0.9, 0.1, 0.8])
        label = paddle.to_tensor([1.0, 0.0, 1.0])
        result = F.binary_cross_entropy(input, label)
        self.assertEqual(result.shape, [])

    def test_bce_loss_with_weight(self):
        """测试带权重的 BCELoss
        Test BCELoss with weight"""
        input = paddle.to_tensor([0.9, 0.1, 0.8])
        label = paddle.to_tensor([1.0, 0.0, 1.0])
        weight = paddle.to_tensor([1.0, 2.0, 1.0])
        result = F.binary_cross_entropy(input, label, weight=weight)
        self.assertEqual(result.shape, [])

    def test_bce_loss_reduction_sum(self):
        """测试 reduction=sum 的 BCELoss
        Test BCELoss with reduction=sum"""
        input = paddle.to_tensor([0.9, 0.1, 0.8])
        label = paddle.to_tensor([1.0, 0.0, 1.0])
        result = F.binary_cross_entropy(input, label, reduction='sum')
        self.assertEqual(result.shape, [])

    def test_bce_loss_no_reduction(self):
        """测试 reduction=none 的 BCELoss
        Test BCELoss with reduction=none"""
        input = paddle.to_tensor([0.9, 0.1, 0.8])
        label = paddle.to_tensor([1.0, 0.0, 1.0])
        result = F.binary_cross_entropy(input, label, reduction='none')
        self.assertEqual(result.shape, [3])


class TestMSELoss(unittest.TestCase):
    """测试均方误差损失函数
    Test Mean Squared Error loss function"""

    def test_mse_loss_basic(self):
        """测试基本的 MSELoss
        Test basic MSELoss"""
        input = paddle.to_tensor([1.0, 2.0, 3.0])
        label = paddle.to_tensor([1.5, 2.5, 3.5])
        result = F.mse_loss(input, label)
        self.assertEqual(result.shape, [])

    def test_mse_loss_reduction_sum(self):
        """测试 reduction=sum 的 MSELoss
        Test MSELoss with reduction=sum"""
        input = paddle.to_tensor([1.0, 2.0, 3.0])
        label = paddle.to_tensor([1.5, 2.5, 3.5])
        result = F.mse_loss(input, label, reduction='sum')
        self.assertEqual(result.shape, [])

    def test_mse_loss_2d(self):
        """测试2D输入的 MSELoss
        Test MSELoss with 2D input"""
        input = paddle.randn([3, 5])
        label = paddle.randn([3, 5])
        result = F.mse_loss(input, label)
        self.assertEqual(result.shape, [])


class TestL1Loss(unittest.TestCase):
    """测试 L1 损失函数
    Test L1 loss function"""

    def test_l1_loss_basic(self):
        """测试基本的 L1Loss
        Test basic L1Loss"""
        input = paddle.to_tensor([1.0, 2.0, 3.0])
        label = paddle.to_tensor([1.5, 2.5, 3.5])
        result = F.l1_loss(input, label)
        self.assertEqual(result.shape, [])

    def test_l1_loss_reduction_sum(self):
        """测试 reduction=sum 的 L1Loss
        Test L1Loss with reduction=sum"""
        input = paddle.to_tensor([1.0, 2.0, 3.0])
        label = paddle.to_tensor([1.5, 2.5, 3.5])
        result = F.l1_loss(input, label, reduction='sum')
        self.assertEqual(result.shape, [])


class TestSmoothL1Loss(unittest.TestCase):
    """测试 Smooth L1 损失函数
    Test Smooth L1 loss function"""

    def test_smooth_l1_loss_basic(self):
        """测试基本的 SmoothL1Loss
        Test basic SmoothL1Loss"""
        input = paddle.to_tensor([1.0, 2.0, 3.0])
        label = paddle.to_tensor([1.5, 2.5, 3.5])
        result = F.smooth_l1_loss(input, label)
        self.assertEqual(result.shape, [])

    def test_smooth_l1_loss_custom_delta(self):
        """测试自定义 delta 的 SmoothL1Loss
        Test SmoothL1Loss with custom delta"""
        input = paddle.to_tensor([1.0, 2.0, 3.0])
        label = paddle.to_tensor([1.5, 2.5, 3.5])
        result = F.smooth_l1_loss(input, label, delta=0.5)
        self.assertEqual(result.shape, [])


class TestKLDivLoss(unittest.TestCase):
    """测试 KL 散度损失函数
    Test KL Divergence loss function"""

    def test_kldiv_loss_basic(self):
        """测试基本的 KLDivLoss
        Test basic KLDivLoss"""
        input = paddle.to_tensor([0.5, 0.5])
        label = paddle.to_tensor([0.3, 0.7])
        result = F.kl_div(input, label)
        self.assertEqual(result.shape, [])


class TestNLLLoss(unittest.TestCase):
    """测试负对数似然损失函数
    Test Negative Log Likelihood loss function"""

    def test_nll_loss_basic(self):
        """测试基本的 NLLLoss
        Test basic NLLLoss"""
        input = paddle.randn([3, 5])
        label = paddle.to_tensor([0, 2, 1], dtype='int64')
        result = F.nll_loss(input, label)
        self.assertEqual(result.shape, [])

    def test_nll_loss_with_weight(self):
        """测试带权重的 NLLLoss
        Test NLLLoss with weight"""
        input = paddle.randn([3, 5])
        label = paddle.to_tensor([0, 2, 1], dtype='int64')
        weight = paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        result = F.nll_loss(input, label, weight=weight)
        self.assertEqual(result.shape, [])


if __name__ == '__main__':
    unittest.main()
