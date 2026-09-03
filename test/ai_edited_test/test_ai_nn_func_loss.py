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
# Target: paddle/nn/functional/loss.py
# Coverage target: improve coverage for loss functions (dice_loss, log_loss, binary_cross_entropy,
#   binary_cross_entropy_with_logits, smooth_l1_loss, mse_loss, nll_loss, kl_div, cross_entropy,
#   margin_ranking_loss, l1_loss, hinge_embedding_loss, huber_loss, poisson_nll_loss, etc.)
"""
Tests for paddle.nn.functional.loss module.
测试 paddle.nn.functional.loss 模块的单元测试。
"""

import unittest

import numpy as np

import paddle
from paddle.nn import functional as F


class TestDiceLoss(unittest.TestCase):
    """Tests for dice_loss function. / dice_loss 函数的测试。"""

    def test_dice_loss_basic(self):
        """Test dice_loss with basic input. / 测试基本输入的 dice_loss。"""
        x = paddle.randn((3, 224, 224, 2), dtype='float32')
        label = paddle.randint(high=2, size=(3, 224, 224, 1), dtype='int64')
        predictions = F.softmax(x)
        loss = F.dice_loss(input=predictions, label=label)
        self.assertEqual(loss.shape, [])

    def test_dice_loss_custom_epsilon(self):
        """Test dice_loss with custom epsilon. / 测试自定义 epsilon 的 dice_loss。"""
        x = paddle.randn((2, 10, 10, 2), dtype='float32')
        label = paddle.randint(high=2, size=(2, 10, 10, 1), dtype='int64')
        predictions = F.softmax(x)
        loss = F.dice_loss(input=predictions, label=label, epsilon=1e-3)
        self.assertEqual(loss.shape, [])


class TestNpairLoss(unittest.TestCase):
    """Tests for npair_loss function. / npair_loss 函数的测试。"""

    def test_npair_loss_basic(self):
        """Test npair_loss with basic inputs. / 测试基本输入的 npair_loss。"""
        paddle.seed(2023)
        anchor = paddle.rand(shape=(6, 4), dtype='float32')
        positive = paddle.rand(shape=(6, 4), dtype='float32')
        labels = paddle.rand(shape=(6,), dtype='float32')
        loss = F.npair_loss(anchor, positive, labels, l2_reg=0.002)
        self.assertEqual(loss.shape, [])


class TestSquareErrorCost(unittest.TestCase):
    """Tests for square_error_cost function. / square_error_cost 函数的测试。"""

    def test_square_error_cost(self):
        """Test square_error_cost. / 测试 square_error_cost。"""
        input_t = paddle.to_tensor([1.1, 1.9], dtype='float32')
        label = paddle.to_tensor([1.0, 2.0], dtype='float32')
        output = F.square_error_cost(input_t, label)
        result = output.numpy()
        np.testing.assert_allclose(result, [0.01, 0.01], rtol=1e-5)


class TestBinaryCrossEntropy(unittest.TestCase):
    """Tests for binary_cross_entropy function. / binary_cross_entropy 函数的测试。"""

    def test_bce_mean(self):
        """Test BCE with mean reduction. / 测试 mean 归约的 BCE。"""
        input_t = paddle.to_tensor([0.5, 0.6, 0.7], dtype='float32')
        label = paddle.to_tensor([1.0, 0.0, 1.0], dtype='float32')
        loss = F.binary_cross_entropy(input_t, label, reduction='mean')
        self.assertEqual(loss.shape, [])

    def test_bce_sum(self):
        """Test BCE with sum reduction. / 测试 sum 归约的 BCE。"""
        input_t = paddle.to_tensor([0.5, 0.6, 0.7], dtype='float32')
        label = paddle.to_tensor([1.0, 0.0, 1.0], dtype='float32')
        loss = F.binary_cross_entropy(input_t, label, reduction='sum')
        self.assertEqual(loss.shape, [])

    def test_bce_none(self):
        """Test BCE with none reduction. / 测试 none 归约的 BCE。"""
        input_t = paddle.to_tensor([0.5, 0.6, 0.7], dtype='float32')
        label = paddle.to_tensor([1.0, 0.0, 1.0], dtype='float32')
        loss = F.binary_cross_entropy(input_t, label, reduction='none')
        self.assertEqual(loss.shape, [3])

    def test_bce_with_weight(self):
        """Test BCE with weight. / 测试带权重的 BCE。"""
        input_t = paddle.to_tensor([0.5, 0.6, 0.7], dtype='float32')
        label = paddle.to_tensor([1.0, 0.0, 1.0], dtype='float32')
        weight = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        loss = F.binary_cross_entropy(
            input_t, label, weight=weight, reduction='mean'
        )
        self.assertEqual(loss.shape, [])

    def test_bce_invalid_reduction(self):
        """Test BCE raises ValueError for invalid reduction. / 测试无效归约方式时 BCE 抛出 ValueError。"""
        input_t = paddle.to_tensor([0.5], dtype='float32')
        label = paddle.to_tensor([1.0], dtype='float32')
        with self.assertRaises(ValueError):
            F.binary_cross_entropy(input_t, label, reduction='invalid')


class TestBinaryCrossEntropyWithLogits(unittest.TestCase):
    """Tests for binary_cross_entropy_with_logits function. / binary_cross_entropy_with_logits 函数的测试。"""

    def test_bce_logits_default(self):
        """Test BCE with logits default. / 测试默认参数的 BCE with logits。"""
        logit = paddle.to_tensor([0.1, 0.2, 0.3], dtype='float32')
        label = paddle.to_tensor([1.0, 0.0, 1.0], dtype='float32')
        loss = F.binary_cross_entropy_with_logits(logit, label)
        self.assertEqual(loss.shape, [])

    def test_bce_logits_none_reduction(self):
        """Test BCE with logits none reduction. / 测试 none 归约的 BCE with logits。"""
        logit = paddle.to_tensor([0.1, 0.2], dtype='float32')
        label = paddle.to_tensor([1.0, 0.0], dtype='float32')
        loss = F.binary_cross_entropy_with_logits(
            logit, label, reduction='none'
        )
        self.assertEqual(loss.shape, [2])

    def test_bce_logits_alias(self):
        """Test BCE with logits using alias names. / 测试别名的 BCE with logits。"""
        logit = paddle.to_tensor([0.1, 0.2], dtype='float32')
        target = paddle.to_tensor([1.0, 0.0], dtype='float32')
        loss = F.binary_cross_entropy_with_logits(input=logit, target=target)
        self.assertEqual(loss.shape, [])


class TestSmoothL1Loss(unittest.TestCase):
    """Tests for smooth_l1_loss function. / smooth_l1_loss 函数的测试。"""

    def test_smooth_l1_default(self):
        """Test smooth_l1_loss with default delta. / 测试默认 delta 的 smooth_l1_loss。"""
        input_t = paddle.to_tensor([0.1, 0.2, 0.8], dtype='float32')
        label = paddle.to_tensor([0.0, 0.0, 1.0], dtype='float32')
        loss = F.smooth_l1_loss(input_t, label)
        self.assertEqual(loss.shape, [])

    def test_smooth_l1_none_reduction(self):
        """Test smooth_l1_loss with none reduction. / 测试 none 归约的 smooth_l1_loss。"""
        input_t = paddle.to_tensor([0.1, 0.2], dtype='float32')
        label = paddle.to_tensor([0.0, 0.0], dtype='float32')
        loss = F.smooth_l1_loss(input_t, label, reduction='none')
        self.assertEqual(loss.shape, [2])

    def test_smooth_l1_sum(self):
        """Test smooth_l1_loss with sum reduction. / 测试 sum 归约的 smooth_l1_loss。"""
        input_t = paddle.to_tensor([0.1, 0.2], dtype='float32')
        label = paddle.to_tensor([0.0, 0.0], dtype='float32')
        loss = F.smooth_l1_loss(input_t, label, reduction='sum')
        self.assertEqual(loss.shape, [])

    def test_smooth_l1_invalid_reduction(self):
        """Test smooth_l1_loss raises ValueError for invalid reduction. / 测试无效归约时 smooth_l1_loss 抛出 ValueError。"""
        input_t = paddle.to_tensor([0.1], dtype='float32')
        label = paddle.to_tensor([0.0], dtype='float32')
        with self.assertRaises(ValueError):
            F.smooth_l1_loss(input_t, label, reduction='invalid')


class TestMseLoss(unittest.TestCase):
    """Tests for mse_loss function. / mse_loss 函数的测试。"""

    def test_mse_default(self):
        """Test mse_loss with default reduction. / 测试默认归约的 mse_loss。"""
        input_t = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        label = paddle.to_tensor([1.1, 2.1, 2.9], dtype='float32')
        loss = F.mse_loss(input_t, label)
        self.assertEqual(loss.shape, [])

    def test_mse_none_reduction(self):
        """Test mse_loss with none reduction. / 测试 none 归约的 mse_loss。"""
        input_t = paddle.to_tensor([1.0, 2.0], dtype='float32')
        label = paddle.to_tensor([1.1, 2.1], dtype='float32')
        loss = F.mse_loss(input_t, label, reduction='none')
        self.assertEqual(loss.shape, [2])

    def test_mse_sum(self):
        """Test mse_loss with sum reduction. / 测试 sum 归约的 mse_loss。"""
        input_t = paddle.to_tensor([1.0, 2.0], dtype='float32')
        label = paddle.to_tensor([1.1, 2.1], dtype='float32')
        loss = F.mse_loss(input_t, label, reduction='sum')
        self.assertEqual(loss.shape, [])


class TestL1Loss(unittest.TestCase):
    """Tests for l1_loss function. / l1_loss 函数的测试。"""

    def test_l1_default(self):
        """Test l1_loss with default reduction. / 测试默认归约的 l1_loss。"""
        input_t = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
        label = paddle.to_tensor([1.1, 2.1, 2.9], dtype='float32')
        loss = F.l1_loss(input_t, label)
        self.assertEqual(loss.shape, [])

    def test_l1_none_reduction(self):
        """Test l1_loss with none reduction. / 测试 none 归约的 l1_loss。"""
        input_t = paddle.to_tensor([1.0, 2.0], dtype='float32')
        label = paddle.to_tensor([1.1, 2.1], dtype='float32')
        loss = F.l1_loss(input_t, label, reduction='none')
        self.assertEqual(loss.shape, [2])

    def test_l1_sum(self):
        """Test l1_loss with sum reduction. / 测试 sum 归约的 l1_loss。"""
        input_t = paddle.to_tensor([1.0, 2.0], dtype='float32')
        label = paddle.to_tensor([1.1, 2.1], dtype='float32')
        loss = F.l1_loss(input_t, label, reduction='sum')
        self.assertEqual(loss.shape, [])


class TestNLLLoss(unittest.TestCase):
    """Tests for nll_loss function. / nll_loss 函数的测试。"""

    def test_nll_mean(self):
        """Test nll_loss with mean reduction. / 测试 mean 归约的 nll_loss。"""
        logit = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        loss = F.nll_loss(logit, label, reduction='mean')
        self.assertEqual(loss.shape, [])

    def test_nll_none(self):
        """Test nll_loss with none reduction. / 测试 none 归约的 nll_loss。"""
        logit = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        loss = F.nll_loss(logit, label, reduction='none')
        self.assertEqual(loss.shape, [3])

    def test_nll_sum(self):
        """Test nll_loss with sum reduction. / 测试 sum 归约的 nll_loss。"""
        logit = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        loss = F.nll_loss(logit, label, reduction='sum')
        self.assertEqual(loss.shape, [])

    def test_nll_with_weight(self):
        """Test nll_loss with weight. / 测试带权重的 nll_loss。"""
        logit = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        weight = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype='float32')
        loss = F.nll_loss(logit, label, weight=weight)
        self.assertEqual(loss.shape, [])

    def test_nll_ignore_index(self):
        """Test nll_loss with ignore_index. / 测试带 ignore_index 的 nll_loss。"""
        logit = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, -100, 4], dtype='int64')
        loss = F.nll_loss(logit, label, ignore_index=-100)
        self.assertEqual(loss.shape, [])


class TestCrossEntropy(unittest.TestCase):
    """Tests for cross_entropy function. / cross_entropy 函数的测试。"""

    def test_cross_entropy_hard_label(self):
        """Test cross_entropy with hard labels. / 测试硬标签的 cross_entropy。"""
        logits = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        loss = F.cross_entropy(logits, label)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_soft_label(self):
        """Test cross_entropy with soft labels. / 测试软标签的 cross_entropy。"""
        logits = paddle.randn([3, 5], dtype='float32')
        label = paddle.randn([3, 5], dtype='float32')
        label = F.softmax(label)
        loss = F.cross_entropy(logits, label, soft_label=True)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_with_weight(self):
        """Test cross_entropy with weight. / 测试带权重的 cross_entropy。"""
        logits = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        weight = paddle.to_tensor([1.0, 2.0, 1.0, 1.0, 1.0], dtype='float32')
        loss = F.cross_entropy(logits, label, weight=weight)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_none_reduction(self):
        """Test cross_entropy with none reduction. / 测试 none 归约的 cross_entropy。"""
        logits = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        loss = F.cross_entropy(logits, label, reduction='none')
        self.assertEqual(loss.shape, [3])

    def test_cross_entropy_sum_reduction(self):
        """Test cross_entropy with sum reduction. / 测试 sum 归约的 cross_entropy。"""
        logits = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        loss = F.cross_entropy(logits, label, reduction='sum')
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_label_smoothing(self):
        """Test cross_entropy with label smoothing. / 测试带标签平滑的 cross_entropy。"""
        logits = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        loss = F.cross_entropy(logits, label, label_smoothing=0.1)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_ignore_index(self):
        """Test cross_entropy with ignore_index. / 测试带 ignore_index 的 cross_entropy。"""
        logits = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, -100, 4], dtype='int64')
        loss = F.cross_entropy(logits, label, ignore_index=-100)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_invalid_reduction(self):
        """Test cross_entropy raises ValueError for invalid reduction. / 测试无效归约时 cross_entropy 抛出 ValueError。"""
        logits = paddle.randn([3, 5], dtype='float32')
        label = paddle.to_tensor([1, 2, 4], dtype='int64')
        with self.assertRaises(ValueError):
            F.cross_entropy(logits, label, reduction='invalid')

    def test_cross_entropy_zero_dim(self):
        """Test cross_entropy raises ValueError for zero-dim input. / 测试零维输入时 cross_entropy 抛出 ValueError。"""
        logits = paddle.to_tensor(0.5, dtype='float32')
        label = paddle.to_tensor(0, dtype='int64')
        with self.assertRaises(ValueError):
            F.cross_entropy(logits, label)


class TestMarginRankingLoss(unittest.TestCase):
    """Tests for margin_ranking_loss function. / margin_ranking_loss 函数的测试。"""

    def test_margin_ranking_loss_default(self):
        """Test margin_ranking_loss with default margin. / 测试默认 margin 的 margin_ranking_loss。"""
        input_t = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
        other = paddle.to_tensor([[2, 1], [2, 4]], dtype='float32')
        label = paddle.to_tensor([[1, -1], [-1, -1]], dtype='float32')
        loss = F.margin_ranking_loss(input_t, other, label)
        self.assertEqual(loss.shape, [])

    def test_margin_ranking_loss_none(self):
        """Test margin_ranking_loss with none reduction. / 测试 none 归约的 margin_ranking_loss。"""
        input_t = paddle.to_tensor([1.0, 2.0], dtype='float32')
        other = paddle.to_tensor([2.0, 1.0], dtype='float32')
        label = paddle.to_tensor([1.0, -1.0], dtype='float32')
        loss = F.margin_ranking_loss(input_t, other, label, reduction='none')
        self.assertEqual(loss.shape, [2])


class TestKLDivLoss(unittest.TestCase):
    """Tests for kl_div function. / kl_div 函数的测试。"""

    def test_kl_div_default(self):
        """Test kl_div with default params. / 测试默认参数的 kl_div。"""
        input_t = paddle.randn([3, 5], dtype='float32')
        label = paddle.randn([3, 5], dtype='float32')
        loss = F.kl_div(input_t, label)
        self.assertEqual(loss.shape, [])

    def test_kl_div_none_reduction(self):
        """Test kl_div with none reduction. / 测试 none 归约的 kl_div。"""
        input_t = paddle.randn([3, 5], dtype='float32')
        label = paddle.randn([3, 5], dtype='float32')
        loss = F.kl_div(input_t, label, reduction='none')
        self.assertEqual(loss.shape, [3, 5])

    def test_kl_div_sum(self):
        """Test kl_div with sum reduction. / 测试 sum 归约的 kl_div。"""
        input_t = paddle.randn([3, 5], dtype='float32')
        label = paddle.randn([3, 5], dtype='float32')
        loss = F.kl_div(input_t, label, reduction='sum')
        self.assertEqual(loss.shape, [])


class TestHingeEmbeddingLoss(unittest.TestCase):
    """Tests for hinge_embedding_loss function. / hinge_embedding_loss 函数的测试。"""

    def test_hinge_embedding_default(self):
        """Test hinge_embedding_loss with default margin. / 测试默认 margin 的 hinge_embedding_loss。"""
        input_t = paddle.to_tensor([0.5, -0.5, 0.5], dtype='float32')
        label = paddle.to_tensor([1.0, 1.0, -1.0], dtype='float32')
        loss = F.hinge_embedding_loss(input_t, label)
        self.assertEqual(loss.shape, [])

    def test_hinge_embedding_none(self):
        """Test hinge_embedding_loss with none reduction. / 测试 none 归约的 hinge_embedding_loss。"""
        input_t = paddle.to_tensor([0.5, -0.5], dtype='float32')
        label = paddle.to_tensor([1.0, -1.0], dtype='float32')
        loss = F.hinge_embedding_loss(input_t, label, reduction='none')
        self.assertEqual(loss.shape, [2])


class TestHingeEmbeddingLossExtra(unittest.TestCase):
    """Tests for hinge_embedding_loss extra cases. / hinge_embedding_loss 额外测试。"""

    def test_hinge_embedding_sum(self):
        """Test hinge_embedding_loss with sum reduction. / 测试 sum 归约的 hinge_embedding_loss。"""
        input_t = paddle.to_tensor([0.5, -0.5], dtype='float32')
        label = paddle.to_tensor([1.0, -1.0], dtype='float32')
        loss = F.hinge_embedding_loss(input_t, label, reduction='sum')
        self.assertEqual(loss.shape, [])


class TestSoftMarginLoss(unittest.TestCase):
    """Tests for soft_margin_loss function. / soft_margin_loss 函数的测试。"""

    def test_soft_margin_loss_default(self):
        """Test soft_margin_loss with default reduction. / 测试默认归约的 soft_margin_loss。"""
        input_t = paddle.to_tensor([0.3, 0.7, -0.5], dtype='float32')
        label = paddle.to_tensor([1.0, -1.0, 1.0], dtype='float32')
        loss = F.soft_margin_loss(input_t, label)
        self.assertEqual(loss.shape, [])

    def test_soft_margin_loss_none(self):
        """Test soft_margin_loss with none reduction. / 测试 none 归约的 soft_margin_loss。"""
        input_t = paddle.to_tensor([0.3, 0.7], dtype='float32')
        label = paddle.to_tensor([1.0, -1.0], dtype='float32')
        loss = F.soft_margin_loss(input_t, label, reduction='none')
        self.assertEqual(loss.shape, [2])


class TestTripletMarginWithDistanceLoss(unittest.TestCase):
    """Tests for triplet_margin_with_distance_loss function. / triplet_margin_with_distance_loss 函数的测试。"""

    def test_triplet_margin_distance_default(self):
        """Test triplet_margin_with_distance_loss default. / 测试默认参数的 triplet_margin_with_distance_loss。"""
        anchor = paddle.randn([4, 8], dtype='float32')
        positive = paddle.randn([4, 8], dtype='float32')
        negative = paddle.randn([4, 8], dtype='float32')
        distance_function = lambda x, y: paddle.norm(x - y, p=2)
        loss = F.triplet_margin_with_distance_loss(
            anchor, positive, negative, distance_function=distance_function
        )
        self.assertEqual(loss.shape, [])


if __name__ == '__main__':
    unittest.main()
