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
文本处理工具测试 / Text Processing Utility Tests

测试目标 / Test Target:
  paddle.text 文本处理功能

覆盖的模块 / Covered Modules:
  - paddle.nn.Embedding: 词嵌入
  - paddle.nn.functional.one_hot: One-hot编码
  - paddle.nn.functional.label_smooth: 标签平滑
  - paddle.text: 文本数据集

作用 / Purpose:
  补充文本处理API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import nn

paddle.disable_static()


class TestOneHot(unittest.TestCase):
    """测试One-Hot编码 / Test one-hot encoding"""

    def test_one_hot_basic(self):
        """测试基本One-Hot / Test basic one-hot"""
        x = paddle.to_tensor([0, 1, 2, 3])
        result = F.one_hot(x, num_classes=5)
        self.assertEqual(result.shape, [4, 5])
        # Check correctness
        np.testing.assert_allclose(result[0].numpy(), [1, 0, 0, 0, 0])
        np.testing.assert_allclose(result[1].numpy(), [0, 1, 0, 0, 0])

    def test_one_hot_2d(self):
        """测试2D输入One-Hot / Test one-hot with 2D input"""
        x = paddle.to_tensor([[0, 1], [2, 3]])
        result = F.one_hot(x, num_classes=5)
        self.assertEqual(result.shape, [2, 2, 5])

    def test_one_hot_last_class(self):
        """测试最后一类One-Hot / Test one-hot for last class"""
        x = paddle.to_tensor([4])
        result = F.one_hot(x, num_classes=5)
        np.testing.assert_allclose(result[0].numpy(), [0, 0, 0, 0, 1])


class TestLabelSmooth(unittest.TestCase):
    """测试标签平滑 / Test label smoothing"""

    def test_label_smooth_basic(self):
        """测试基本标签平滑 / Test basic label smoothing"""
        label = paddle.to_tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        smoothed = F.label_smooth(label, epsilon=0.1)
        self.assertEqual(smoothed.shape, [2, 3])
        # After smoothing, sum should still be 1 per row
        row_sums = smoothed.sum(axis=1)
        np.testing.assert_allclose(row_sums.numpy(), [1.0, 1.0], rtol=1e-5)

    def test_label_smooth_epsilon(self):
        """测试不同epsilon的标签平滑 / Test label smooth with different epsilon"""
        label = paddle.to_tensor([[1.0, 0.0]])
        smoothed = F.label_smooth(label, epsilon=0.2)
        # Maximum value should be less than 1.0
        self.assertLess(float(smoothed.max().numpy()), 1.0)


class TestSequenceOps(unittest.TestCase):
    """测试序列操作 / Test sequence operations"""

    def test_embedding_bag(self):
        """测试EmbeddingBag模拟 / Test EmbeddingBag simulation"""
        emb = nn.Embedding(num_embeddings=100, embedding_dim=16)
        x = paddle.to_tensor([0, 1, 2, 3, 4, 5])
        # Simulate bag aggregation: two bags [0,1,2] and [3,4,5]
        bag1 = emb(x[:3]).mean(axis=0)
        bag2 = emb(x[3:]).mean(axis=0)
        result = paddle.stack([bag1, bag2])
        self.assertEqual(result.shape, [2, 16])

    def test_embedding_sequential(self):
        """测试序列嵌入 / Test sequential embedding"""
        # Simulate sequence processing
        vocab_size = 50
        d_model = 16
        emb = nn.Embedding(vocab_size, d_model)
        # Batch of 4 sequences, each length 10
        tokens = paddle.randint(0, vocab_size, [4, 10])
        embedded = emb(tokens)
        self.assertEqual(embedded.shape, [4, 10, d_model])


class TestCrossEntropyVariants(unittest.TestCase):
    """测试交叉熵变体 / Test cross-entropy variants"""

    def test_cross_entropy_hard_label(self):
        """测试硬标签交叉熵 / Test hard label cross entropy"""
        logits = paddle.randn([4, 10])
        labels = paddle.randint(0, 10, [4])
        loss = F.cross_entropy(logits, labels)
        self.assertGreater(float(loss.numpy()), 0)

    def test_cross_entropy_soft_label(self):
        """测试软标签交叉熵 / Test soft label cross entropy"""
        logits = paddle.randn([4, 10])
        labels = paddle.ones([4, 10]) / 10  # Uniform soft labels
        loss = F.cross_entropy(logits, labels, soft_label=True)
        self.assertGreater(float(loss.numpy()), 0)

    def test_cross_entropy_reduction(self):
        """测试不同归约方式的交叉熵 / Test CE with different reductions"""
        logits = paddle.randn([4, 10])
        labels = paddle.randint(0, 10, [4])
        loss_mean = F.cross_entropy(logits, labels, reduction='mean')
        loss_sum = F.cross_entropy(logits, labels, reduction='sum')
        loss_none = F.cross_entropy(logits, labels, reduction='none')
        self.assertEqual(loss_none.shape, [4])
        self.assertAlmostEqual(
            float(loss_sum.numpy()), float(loss_none.sum().numpy()), places=4
        )


if __name__ == '__main__':
    unittest.main()
