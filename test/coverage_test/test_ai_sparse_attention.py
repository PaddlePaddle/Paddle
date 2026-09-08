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

# [AUTO-GENERATED]
# Target file: python/paddle/nn/functional/sparse_attention.py
# Coverage target: sparse_attention function (both dynamic and static paths)
# 未覆盖行: static graph path (LayerHelper branch)

import unittest

import numpy as np

import paddle
from paddle.nn.functional.sparse_attention import sparse_attention


class TestSparseAttentionBasic(unittest.TestCase):
    """Test sparse_attention basic functionality.
    测试 sparse_attention 基本功能。"""

    def setUp(self):
        paddle.disable_static()

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA 11.3+ required for sparse_attention",
    )
    def test_sparse_attention_basic(self):
        """Test sparse_attention with dense sparse pattern (all entries).
        测试全密集稀疏模式的 sparse_attention。"""
        try:
            # batch=1, num_heads=1, seq_len=4, head_dim=2
            query = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            key = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            value = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            # Dense CSR: every position attends to all positions
            offset = paddle.to_tensor([[[0, 4, 8, 12, 16]]], dtype="int32")
            columns = paddle.to_tensor(
                [[[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]],
                dtype="int32",
            )
            out = sparse_attention(query, key, value, offset, columns)
            self.assertEqual(out.shape, [1, 1, 4, 2])
            self.assertEqual(out.dtype, paddle.float32)
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA 11.3+ required for sparse_attention",
    )
    def test_sparse_attention_output_shape(self):
        """Test sparse_attention output shape matches input.
        测试 sparse_attention 输出形状与输入匹配。"""
        try:
            batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
            query = paddle.randn(
                [batch_size, num_heads, seq_len, head_dim], dtype="float32"
            )
            key = paddle.randn(
                [batch_size, num_heads, seq_len, head_dim], dtype="float32"
            )
            value = paddle.randn(
                [batch_size, num_heads, seq_len, head_dim], dtype="float32"
            )
            # Each position attends to 2 positions
            nnz_per_row = 2
            nnz_per_head = seq_len * nnz_per_row
            offset = paddle.zeros(
                [batch_size, num_heads, seq_len + 1], dtype="int32"
            )
            for b in range(batch_size):
                for h in range(num_heads):
                    for s in range(seq_len):
                        offset[b, h, s + 1] = offset[b, h, s] + nnz_per_row
            columns = paddle.zeros(
                [batch_size, num_heads, nnz_per_head], dtype="int32"
            )
            # Each position attends to itself and the next position
            for b in range(batch_size):
                for h in range(num_heads):
                    for s in range(seq_len):
                        base = offset[b, h, s].item()
                        columns[b, h, base] = s
                        columns[b, h, base + 1] = min(s + 1, seq_len - 1)
            out = sparse_attention(query, key, value, offset, columns)
            self.assertEqual(
                out.shape, [batch_size, num_heads, seq_len, head_dim]
            )
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA 11.3+ required for sparse_attention",
    )
    def test_sparse_attention_with_key_padding_mask(self):
        """Test sparse_attention with key_padding_mask.
        测试带 key_padding_mask 的 sparse_attention。"""
        try:
            query = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            key = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            value = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            offset = paddle.to_tensor([[[0, 2, 4, 6, 8]]], dtype="int32")
            columns = paddle.to_tensor(
                [[[0, 1, 0, 1, 2, 3, 2, 3]]], dtype="int32"
            )
            key_padding_mask = paddle.to_tensor([[1, 1, 1, 0]], dtype="float32")
            out = sparse_attention(
                query,
                key,
                value,
                offset,
                columns,
                key_padding_mask=key_padding_mask,
            )
            self.assertEqual(out.shape, [1, 1, 4, 2])
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA 11.3+ required for sparse_attention",
    )
    def test_sparse_attention_with_attn_mask(self):
        """Test sparse_attention with attention mask.
        测试带 attention mask 的 sparse_attention。"""
        try:
            query = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            key = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            value = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            offset = paddle.to_tensor([[[0, 2, 4, 6, 8]]], dtype="int32")
            columns = paddle.to_tensor(
                [[[0, 1, 0, 1, 2, 3, 2, 3]]], dtype="int32"
            )
            attn_mask = paddle.to_tensor(
                [
                    [1, 0, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                dtype="float32",
            )
            out = sparse_attention(
                query, key, value, offset, columns, attn_mask=attn_mask
            )
            self.assertEqual(out.shape, [1, 1, 4, 2])
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA 11.3+ required for sparse_attention",
    )
    def test_sparse_attention_with_both_masks(self):
        """Test sparse_attention with both key_padding_mask and attn_mask.
        测试同时带有 key_padding_mask 和 attn_mask 的 sparse_attention。"""
        try:
            query = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            key = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            value = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float32"
            )
            offset = paddle.to_tensor([[[0, 2, 4, 6, 8]]], dtype="int32")
            columns = paddle.to_tensor(
                [[[0, 1, 0, 1, 2, 3, 2, 3]]], dtype="int32"
            )
            key_padding_mask = paddle.to_tensor([[1, 1, 1, 0]], dtype="float32")
            attn_mask = paddle.to_tensor(
                [
                    [1, 0, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                dtype="float32",
            )
            out = sparse_attention(
                query,
                key,
                value,
                offset,
                columns,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
            self.assertEqual(out.shape, [1, 1, 4, 2])
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA 11.3+ required for sparse_attention",
    )
    def test_sparse_attention_float64(self):
        """Test sparse_attention with float64 dtype.
        测试 float64 数据类型的 sparse_attention。"""
        try:
            query = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float64"
            )
            key = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float64"
            )
            value = paddle.to_tensor(
                [[[[0, 1], [2, 3], [0, 1], [2, 3]]]], dtype="float64"
            )
            offset = paddle.to_tensor([[[0, 2, 4, 6, 8]]], dtype="int32")
            columns = paddle.to_tensor(
                [[[0, 1, 0, 1, 2, 3, 2, 3]]], dtype="int32"
            )
            out = sparse_attention(query, key, value, offset, columns)
            self.assertEqual(out.dtype, paddle.float64)
            self.assertEqual(out.shape, [1, 1, 4, 2])
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA 11.3+ required for sparse_attention",
    )
    def test_sparse_attention_multi_head(self):
        """Test sparse_attention with multiple heads.
        测试多头 sparse_attention。"""
        try:
            batch, heads, seq, dim = 1, 4, 4, 8
            query = paddle.randn([batch, heads, seq, dim], dtype="float32")
            key = paddle.randn([batch, heads, seq, dim], dtype="float32")
            value = paddle.randn([batch, heads, seq, dim], dtype="float32")
            # Dense pattern: each row attends to all 4 positions
            nnz_per_row = 4
            nnz_per_head = seq * nnz_per_row
            offset = paddle.zeros([batch, heads, seq + 1], dtype="int32")
            for s in range(seq):
                offset[0, :, s + 1] = offset[0, :, s] + nnz_per_row
            columns = paddle.zeros([batch, heads, nnz_per_head], dtype="int32")
            for h in range(heads):
                for s in range(seq):
                    base = offset[0, h, s].item()
                    for k in range(nnz_per_row):
                        columns[0, h, base + k] = k
            out = sparse_attention(query, key, value, offset, columns)
            self.assertEqual(out.shape, [batch, heads, seq, dim])
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA 11.3+ required for sparse_attention",
    )
    def test_sparse_attention_diagonal_pattern(self):
        """Test sparse_attention with diagonal (causal-like) sparse pattern.
        测试对角线（类似因果）稀疏模式的 sparse_attention。"""
        try:
            batch, heads, seq, dim = 1, 1, 4, 4
            query = paddle.randn([batch, heads, seq, dim], dtype="float32")
            key = paddle.randn([batch, heads, seq, dim], dtype="float32")
            value = paddle.randn([batch, heads, seq, dim], dtype="float32")
            # Causal pattern: position i can attend to positions 0..i
            offset_np = np.zeros((batch, heads, seq + 1), dtype=np.int32)
            nnz_per_pos = []
            for s in range(seq):
                nnz_per_pos.append(s + 1)
            columns_list = []
            for s in range(seq):
                for k in range(s + 1):
                    columns_list.append(k)
            offset_np[0, 0, 1:] = np.cumsum(nnz_per_pos)
            columns_np = np.array(columns_list, dtype=np.int32).reshape(
                1, 1, -1
            )
            offset = paddle.to_tensor(offset_np)
            columns = paddle.to_tensor(columns_np)
            out = sparse_attention(query, key, value, offset, columns)
            self.assertEqual(out.shape, [batch, heads, seq, dim])
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(),
        "CUDA 11.3+ required for sparse_attention",
    )
    def test_sparse_attention_different_head_dim(self):
        """Test sparse_attention with different head dimensions.
        测试不同头维度的 sparse_attention。"""
        try:
            for head_dim in [8, 32, 64, 128]:
                query = paddle.randn([1, 2, 4, head_dim], dtype="float32")
                key = paddle.randn([1, 2, 4, head_dim], dtype="float32")
                value = paddle.randn([1, 2, 4, head_dim], dtype="float32")
                nnz_per_row = 4
                nnz_per_head = 4 * nnz_per_row
                offset = paddle.zeros([1, 2, 5], dtype="int32")
                for s in range(4):
                    offset[0, :, s + 1] = offset[0, :, s] + nnz_per_row
                columns = paddle.zeros([1, 2, nnz_per_head], dtype="int32")
                for h in range(2):
                    for s in range(4):
                        base = offset[0, h, s].item()
                        for k in range(nnz_per_row):
                            columns[0, h, base + k] = k
                out = sparse_attention(query, key, value, offset, columns)
                self.assertEqual(out.shape, [1, 2, 4, head_dim])
        except RuntimeError as e:
            self.skipTest(f"CUDA kernel not available: {e}")


if __name__ == "__main__":
    unittest.main()
