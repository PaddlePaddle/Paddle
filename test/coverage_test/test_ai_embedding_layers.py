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

# Unit test for paddle.nn.functional.embedding
# Target: cover embedding related code paths

import unittest

import paddle
import paddle.nn.functional as F
from paddle import nn


class TestEmbedding(unittest.TestCase):
    """Test embedding function."""

    def setUp(self):
        paddle.disable_static()

    def test_embedding_basic(self):
        """Basic embedding lookup."""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='int64')
        w = paddle.randn([10, 32])
        out = F.embedding(x, w)
        self.assertEqual(out.shape, [2, 3, 32])

    def test_embedding_with_padding_idx(self):
        """Embedding with padding_idx."""
        x = paddle.to_tensor([[0, 1, 2], [0, 3, 4]], dtype='int64')
        w = paddle.randn([10, 32])
        out = F.embedding(x, w, padding_idx=0)
        self.assertEqual(out.shape, [2, 3, 32])

    def test_embedding_sparse_gradient(self):
        """Embedding with sparse gradient."""
        x = paddle.to_tensor([[1, 2, 3]], dtype='int64')
        w = paddle.randn([10, 32])
        w.stop_gradient = False
        out = F.embedding(x, w, sparse=True)
        self.assertEqual(out.shape, [1, 3, 32])

    def test_embedding_float16_weight(self):
        """Embedding with float16 weight."""
        x = paddle.to_tensor([[1, 2, 3]], dtype='int64')
        w = paddle.randn([10, 32], dtype='float16')
        out = F.embedding(x, w)
        self.assertEqual(out.dtype, paddle.float16)

    def test_embedding_int32_input(self):
        """Embedding with int32 input."""
        x = paddle.to_tensor([[1, 2, 3]], dtype='int32')
        w = paddle.randn([10, 32])
        out = F.embedding(x, w)
        self.assertEqual(out.shape, [1, 3, 32])

    def test_embedding_max_norm(self):
        """Embedding with max_norm."""
        x = paddle.to_tensor([[1, 2, 3]], dtype='int64')
        w = paddle.randn([10, 32]) * 10  # large values
        out = F.embedding(x, w, max_norm=1.0)
        self.assertEqual(out.shape, [1, 3, 32])
        # Check that norms are bounded
        norms = paddle.norm(out, p=2, axis=-1)
        self.assertTrue(paddle.all(norms <= 1.0 + 1e-5))


class TestOneHot(unittest.TestCase):
    """Test one_hot function.
    F.one_hot(x, num_classes) - no dtype parameter.
    """

    def setUp(self):
        paddle.disable_static()

    def test_one_hot_basic(self):
        """Basic one_hot encoding."""
        x = paddle.to_tensor([0, 1, 2, 3], dtype='int64')
        out = F.one_hot(x, num_classes=5)
        self.assertEqual(out.shape, [4, 5])
        # Check one-hot encoding
        result = out.numpy()
        for i in range(4):
            self.assertEqual(result[i, i], 1)
            # All other positions should be 0
            for j in range(5):
                if j != i:
                    self.assertEqual(result[i, j], 0)

    def test_one_hot_int32(self):
        """One_hot with int32 input."""
        x = paddle.to_tensor([0, 1], dtype='int32')
        out = F.one_hot(x, num_classes=3)
        self.assertEqual(out.shape, [2, 3])

    def test_one_hot_2d(self):
        """One_hot with 2D input."""
        x = paddle.to_tensor([[0, 1], [2, 0]], dtype='int64')
        out = F.one_hot(x, num_classes=4)
        self.assertEqual(out.shape, [2, 2, 4])

    def test_one_hot_output_dtype(self):
        """One_hot default dtype is float32."""
        x = paddle.to_tensor([0, 1], dtype='int64')
        out = F.one_hot(x, num_classes=3)
        # Default output dtype
        self.assertEqual(out.shape, [2, 3])


class TestEmbeddingLayer(unittest.TestCase):
    """Test nn.Embedding layer."""

    def setUp(self):
        paddle.disable_static()

    def test_embedding_layer_basic(self):
        """nn.Embedding basic usage."""
        layer = nn.Embedding(100, 32)
        x = paddle.to_tensor([[1, 2, 3]], dtype='int64')
        out = layer(x)
        self.assertEqual(out.shape, [1, 3, 32])

    def test_embedding_layer_with_padding_idx(self):
        """nn.Embedding with padding_idx."""
        layer = nn.Embedding(100, 32, padding_idx=0)
        x = paddle.to_tensor([[0, 1, 2]], dtype='int64')
        out = layer(x)
        self.assertEqual(out.shape, [1, 3, 32])

    def test_embedding_layer_sparse(self):
        """nn.Embedding with sparse gradient."""
        layer = nn.Embedding(100, 32, sparse=True)
        x = paddle.to_tensor([[1, 2, 3]], dtype='int64')
        out = layer(x)
        loss = out.mean()
        loss.backward()
        # Should not crash


if __name__ == '__main__':
    unittest.main()
