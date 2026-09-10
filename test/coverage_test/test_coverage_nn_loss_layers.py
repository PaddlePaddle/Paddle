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
损失函数层单元测试 / Loss Function Layer Unit Tests

测试目标 / Test Target:
  paddle.nn 损失层 (python/paddle/nn/functional/loss.py, 覆盖率约77.5%)

覆盖的模块 / Covered Modules:
  - paddle.nn.CrossEntropyLoss: 交叉熵损失
  - paddle.nn.MSELoss: 均方误差损失
  - paddle.nn.L1Loss: L1损失
  - paddle.nn.BCELoss: 二元交叉熵损失
  - paddle.nn.BCEWithLogitsLoss: 带logits的BCE
  - paddle.nn.KLDivLoss: KL散度损失
  - paddle.nn.NLLLoss: 负对数似然损失
  - paddle.nn.SmoothL1Loss: Smooth L1损失
  - paddle.nn.HingeEmbeddingLoss: Hinge嵌入损失
  - paddle.nn.CosineEmbeddingLoss: 余弦嵌入损失

作用 / Purpose:
  覆盖各类损失函数层的代码路径，测试reduction参数、权重、ignore_index等。
"""

import unittest

import paddle
import paddle.nn.functional as F
from paddle import nn

paddle.disable_static()


class TestCrossEntropyLoss(unittest.TestCase):
    """测试交叉熵损失 / Test CrossEntropyLoss"""

    def test_cross_entropy_basic(self):
        """测试基本交叉熵损失 / Test basic cross entropy loss"""
        ce = nn.CrossEntropyLoss()
        pred = paddle.randn([4, 5])
        target = paddle.to_tensor([0, 1, 2, 3])
        loss = ce(pred, target)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_mean(self):
        """测试mean reduction / Test mean reduction"""
        ce = nn.CrossEntropyLoss(reduction='mean')
        pred = paddle.randn([4, 5])
        target = paddle.to_tensor([0, 1, 2, 3])
        loss = ce(pred, target)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_sum(self):
        """测试sum reduction / Test sum reduction"""
        ce = nn.CrossEntropyLoss(reduction='sum')
        pred = paddle.randn([4, 5])
        target = paddle.to_tensor([0, 1, 2, 3])
        loss = ce(pred, target)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_none(self):
        """测试none reduction / Test none reduction"""
        ce = nn.CrossEntropyLoss(reduction='none')
        pred = paddle.randn([4, 5])
        target = paddle.to_tensor([0, 1, 2, 3])
        loss = ce(pred, target)
        self.assertEqual(loss.shape, [4])

    def test_cross_entropy_with_weight(self):
        """测试带权重的交叉熵 / Test cross entropy with weights"""
        weight = paddle.to_tensor([1.0, 2.0, 1.0, 2.0, 1.0])
        ce = nn.CrossEntropyLoss(weight=weight)
        pred = paddle.randn([4, 5])
        target = paddle.to_tensor([0, 1, 2, 3])
        loss = ce(pred, target)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_ignore_index(self):
        """测试ignore_index参数 / Test ignore_index parameter"""
        ce = nn.CrossEntropyLoss(ignore_index=255)
        pred = paddle.randn([4, 5])
        target = paddle.to_tensor([0, 1, 255, 3])
        loss = ce(pred, target)
        self.assertEqual(loss.shape, [])

    def test_cross_entropy_soft_labels(self):
        """测试软标签 / Test soft labels"""
        ce = nn.CrossEntropyLoss(soft_label=True)
        pred = paddle.randn([4, 5])
        target = paddle.to_tensor(
            [[0.2, 0.2, 0.2, 0.2, 0.2]] * 4, dtype='float32'
        )
        loss = ce(pred, target)
        self.assertEqual(loss.shape, [])


class TestMSELoss(unittest.TestCase):
    """测试均方误差损失 / Test MSELoss"""

    def test_mse_basic(self):
        """测试基本MSE / Test basic MSE"""
        mse = nn.MSELoss()
        pred = paddle.randn([4, 5])
        target = paddle.randn([4, 5])
        loss = mse(pred, target)
        self.assertEqual(loss.shape, [])
        self.assertTrue(float(loss.numpy()) >= 0)

    def test_mse_sum_reduction(self):
        """测试sum reduction MSE / Test sum reduction MSE"""
        mse = nn.MSELoss(reduction='sum')
        pred = paddle.randn([4, 5])
        target = paddle.randn([4, 5])
        loss = mse(pred, target)
        self.assertEqual(loss.shape, [])

    def test_mse_none_reduction(self):
        """测试none reduction MSE / Test none reduction MSE"""
        mse = nn.MSELoss(reduction='none')
        pred = paddle.randn([4, 5])
        target = paddle.randn([4, 5])
        loss = mse(pred, target)
        self.assertEqual(loss.shape, [4, 5])

    def test_mse_perfect_prediction(self):
        """测试完美预测的MSE / Test MSE with perfect prediction"""
        mse = nn.MSELoss()
        x = paddle.randn([4, 5])
        loss = mse(x, x)
        self.assertAlmostEqual(float(loss.numpy()), 0.0, places=5)


class TestL1Loss(unittest.TestCase):
    """测试L1损失 / Test L1Loss"""

    def test_l1_basic(self):
        """测试基本L1损失 / Test basic L1 loss"""
        l1 = nn.L1Loss()
        pred = paddle.to_tensor([1.0, 2.0, 3.0])
        target = paddle.to_tensor([1.5, 2.5, 2.5])
        loss = l1(pred, target)
        self.assertAlmostEqual(float(loss.numpy()), 0.5, places=5)

    def test_l1_sum(self):
        """测试sum reduction L1 / Test sum reduction L1"""
        l1 = nn.L1Loss(reduction='sum')
        pred = paddle.to_tensor([1.0, 2.0, 3.0])
        target = paddle.to_tensor([1.0, 2.0, 3.0])
        loss = l1(pred, target)
        self.assertAlmostEqual(float(loss.numpy()), 0.0, places=5)


class TestBCELoss(unittest.TestCase):
    """测试二元交叉熵损失 / Test BCELoss"""

    def test_bce_basic(self):
        """测试基本BCE损失 / Test basic BCE loss"""
        bce = nn.BCELoss()
        pred = paddle.to_tensor([0.3, 0.7, 0.5])
        target = paddle.to_tensor([0.0, 1.0, 1.0])
        loss = bce(pred, target)
        self.assertEqual(loss.shape, [])
        self.assertTrue(float(loss.numpy()) >= 0)

    def test_bce_with_logits(self):
        """测试带Logits的BCE / Test BCE with logits"""
        bce_logits = nn.BCEWithLogitsLoss()
        pred = paddle.randn([4, 5])
        target = paddle.randint(0, 2, [4, 5]).astype('float32')
        loss = bce_logits(pred, target)
        self.assertEqual(loss.shape, [])

    def test_bce_with_weight(self):
        """测试带权重的BCE / Test BCE with weight"""
        weight = paddle.to_tensor([1.0, 2.0, 1.0])
        bce = nn.BCELoss(weight=weight)
        pred = paddle.to_tensor([0.3, 0.7, 0.5])
        target = paddle.to_tensor([0.0, 1.0, 1.0])
        loss = bce(pred, target)
        self.assertEqual(loss.shape, [])


class TestKLDivLoss(unittest.TestCase):
    """测试KL散度损失 / Test KLDivLoss"""

    def test_kl_div_basic(self):
        """测试基本KL散度 / Test basic KL divergence"""
        kl = nn.KLDivLoss(reduction='batchmean')
        # Input should be log probability
        log_pred = F.log_softmax(paddle.randn([4, 5]), axis=-1)
        # Target should be probability
        target = F.softmax(paddle.randn([4, 5]), axis=-1)
        loss = kl(log_pred, target)
        self.assertEqual(loss.shape, [])


class TestNLLLoss(unittest.TestCase):
    """测试负对数似然损失 / Test NLLLoss"""

    def test_nll_basic(self):
        """测试基本NLL损失 / Test basic NLL loss"""
        nll = nn.NLLLoss()
        # Log probabilities
        log_pred = F.log_softmax(paddle.randn([4, 5]), axis=-1)
        target = paddle.to_tensor([0, 1, 2, 3])
        loss = nll(log_pred, target)
        self.assertEqual(loss.shape, [])

    def test_nll_ignore_index(self):
        """测试NLL的ignore_index / Test NLL with ignore_index"""
        nll = nn.NLLLoss(ignore_index=-100)
        log_pred = F.log_softmax(paddle.randn([4, 5]), axis=-1)
        target = paddle.to_tensor([0, 1, -100, 3])
        loss = nll(log_pred, target)
        self.assertEqual(loss.shape, [])


class TestSmoothL1Loss(unittest.TestCase):
    """测试Smooth L1损失 / Test SmoothL1Loss"""

    def test_smooth_l1_basic(self):
        """测试基本Smooth L1 / Test basic Smooth L1"""
        smooth_l1 = nn.SmoothL1Loss()
        pred = paddle.randn([4, 5])
        target = paddle.randn([4, 5])
        loss = smooth_l1(pred, target)
        self.assertEqual(loss.shape, [])

    def test_smooth_l1_delta(self):
        """测试带delta的Smooth L1 / Test Smooth L1 with delta"""
        smooth_l1 = nn.SmoothL1Loss(delta=2.0)
        pred = paddle.randn([4, 5])
        target = paddle.randn([4, 5])
        loss = smooth_l1(pred, target)
        self.assertTrue(float(loss.numpy()) >= 0)


class TestHingeAndCosineEmbeddingLoss(unittest.TestCase):
    """测试Hinge和Cosine嵌入损失 / Test Hinge and Cosine Embedding Loss"""

    def test_hinge_embedding_loss(self):
        """测试Hinge嵌入损失 / Test Hinge embedding loss"""
        hinge = nn.HingeEmbeddingLoss(margin=1.0)
        x = paddle.randn([4])
        y = paddle.to_tensor([1, -1, 1, -1], dtype='float32')
        loss = hinge(x, y)
        self.assertEqual(loss.shape, [])

    def test_cosine_embedding_loss(self):
        """测试余弦嵌入损失 / Test Cosine embedding loss"""
        cosine = nn.CosineEmbeddingLoss(margin=0.5)
        x1 = paddle.randn([4, 10])
        x2 = paddle.randn([4, 10])
        y = paddle.to_tensor([1, -1, 1, -1], dtype='float32')
        loss = cosine(x1, x2, y)
        self.assertEqual(loss.shape, [])

    def test_margin_ranking_loss(self):
        """测试Margin排名损失 / Test Margin ranking loss"""
        margin_loss = nn.MarginRankingLoss(margin=0.5)
        x1 = paddle.randn([4])
        x2 = paddle.randn([4])
        y = paddle.to_tensor([1, -1, 1, -1], dtype='float32')
        loss = margin_loss(x1, x2, y)
        self.assertEqual(loss.shape, [])


if __name__ == '__main__':
    unittest.main()
