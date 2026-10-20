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
Embedding层高级测试 / Advanced Embedding Layer Tests

测试目标 / Test Target:
  paddle.nn.Embedding 高级用法

覆盖的模块 / Covered Modules:
  - paddle.nn.Embedding: 基本嵌入
  - 稀疏更新嵌入
  - 带padding的嵌入
  - 嵌入层梯度

作用 / Purpose:
  补充Embedding层API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestEmbeddingBasic(unittest.TestCase):
    """测试基本Embedding / Test basic Embedding"""

    def test_embedding_basic(self):
        """测试基本嵌入 / Test basic embedding"""
        emb = nn.Embedding(100, 16)
        x = paddle.to_tensor([0, 1, 2, 3])
        result = emb(x)
        self.assertEqual(result.shape, [4, 16])

    def test_embedding_2d_input(self):
        """测试2D输入嵌入 / Test embedding with 2D input"""
        emb = nn.Embedding(100, 16)
        x = paddle.to_tensor([[0, 1, 2], [3, 4, 5]])
        result = emb(x)
        self.assertEqual(result.shape, [2, 3, 16])

    def test_embedding_padding_idx(self):
        """测试padding_idx嵌入 / Test embedding with padding_idx"""
        emb = nn.Embedding(100, 16, padding_idx=0)
        x = paddle.to_tensor([0, 1, 2])
        result = emb(x)
        self.assertEqual(result.shape, [3, 16])
        # Padding index should produce zero vector
        np.testing.assert_allclose(result[0].numpy(), np.zeros(16), atol=1e-7)

    def test_embedding_sparse(self):
        """测试稀疏嵌入 / Test sparse embedding"""
        emb = nn.Embedding(100, 16, sparse=True)
        x = paddle.to_tensor([5, 10, 15])
        result = emb(x)
        self.assertEqual(result.shape, [3, 16])

    def test_embedding_dtype(self):
        """测试嵌入数据类型 / Test embedding dtype"""
        emb = nn.Embedding(
            100,
            16,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal()
            ),
        )
        x = paddle.to_tensor([1, 2, 3])
        result = emb(x)
        self.assertEqual(result.dtype, paddle.float32)


class TestEmbeddingGradient(unittest.TestCase):
    """测试嵌入梯度 / Test embedding gradient"""

    def test_embedding_gradient(self):
        """测试嵌入梯度更新 / Test embedding gradient update"""
        emb = nn.Embedding(10, 4)
        x = paddle.to_tensor([0, 2, 4])
        output = emb(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(emb.weight.grad)

    def test_embedding_with_linear(self):
        """测试嵌入与线性层组合 / Test embedding combined with linear layer"""
        emb = nn.Embedding(100, 16)
        linear = nn.Linear(16, 8)
        x = paddle.to_tensor([1, 2, 3, 4])
        embedded = emb(x)
        output = linear(embedded)
        self.assertEqual(output.shape, [4, 8])


class TestMultipleEmbeddings(unittest.TestCase):
    """测试多嵌入组合 / Test multiple embeddings combination"""

    def test_category_embeddings(self):
        """测试类别嵌入组合 / Test category embeddings combination"""
        # Common in recommendation systems
        item_emb = nn.Embedding(1000, 32)
        user_emb = nn.Embedding(500, 32)
        items = paddle.to_tensor([1, 5, 10])
        users = paddle.to_tensor([2, 3, 4])
        item_vecs = item_emb(items)
        user_vecs = user_emb(users)
        # Dot product similarity
        scores = (item_vecs * user_vecs).sum(axis=1)
        self.assertEqual(scores.shape, [3])

    def test_position_embedding(self):
        """测试位置嵌入 / Test position embedding"""
        max_seq_len = 128
        d_model = 64
        pos_emb = nn.Embedding(max_seq_len, d_model)
        positions = paddle.arange(10)
        pos_vecs = pos_emb(positions)
        self.assertEqual(pos_vecs.shape, [10, d_model])


class TestEmbeddingWeight(unittest.TestCase):
    """测试嵌入权重操作 / Test embedding weight operations"""

    def test_embedding_weight_init(self):
        """测试嵌入权重初始化 / Test embedding weight initialization"""
        # Initialize with specific weights
        weight = paddle.randn([100, 16])
        emb = nn.Embedding(100, 16)
        emb.weight.set_value(weight)
        x = paddle.to_tensor([0, 1, 2])
        result = emb(x)
        np.testing.assert_allclose(
            result.numpy(), weight[:3].numpy(), rtol=1e-5
        )

    def test_embedding_num_embeddings(self):
        """测试嵌入数量属性 / Test embedding num_embeddings attribute"""
        emb = nn.Embedding(200, 32)
        self.assertEqual(emb._num_embeddings, 200)
        self.assertEqual(emb._embedding_dim, 32)


if __name__ == '__main__':
    unittest.main()
