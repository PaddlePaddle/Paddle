# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import paddle


class TestEmptyTensorIndexing(unittest.TestCase):
    """Test empty tensor indexing operations."""

    def test_consecutive_empty_tensor_indexing(self):
        """Test consecutive indexing on empty tensors.

        This test case reproduces the bug reported in issue #76194.
        Before the fix, the second consecutive empty tensor indexing
        operation would fail with ValueError.
        """
        cum_nodes = paddle.zeros([2])
        batch = paddle.empty([0], dtype=paddle.int64)
        edge_index = paddle.empty([2, 0], dtype=paddle.int64)

        # Both operations should succeed
        result1 = cum_nodes[batch][edge_index[0]]
        result2 = cum_nodes[batch][edge_index[1]]

        # Verify results
        self.assertEqual(result1.shape, [0])
        self.assertEqual(result2.shape, [0])
        self.assertEqual(result1.dtype, cum_nodes.dtype)
        self.assertEqual(result2.dtype, cum_nodes.dtype)

    def test_multiple_consecutive_empty_indexing(self):
        """Test multiple consecutive empty tensor indexing operations."""
        tensor = paddle.zeros([10])
        empty_idx = paddle.empty([0], dtype=paddle.int64)

        # Perform multiple consecutive indexing operations
        for i in range(10):
            result = tensor[empty_idx]
            self.assertEqual(result.shape, [0])
            self.assertEqual(result.dtype, tensor.dtype)

    def test_empty_2d_tensor_indexing(self):
        """Test 2D empty tensor indexing."""
        tensor = paddle.zeros([3, 4])
        empty_idx = paddle.empty([0], dtype=paddle.int64)

        result = tensor[empty_idx]
        self.assertEqual(result.shape, [0, 4])
        self.assertEqual(result.dtype, tensor.dtype)

    def test_empty_3d_tensor_indexing(self):
        """Test 3D empty tensor indexing."""
        tensor = paddle.zeros([5, 5, 5])
        empty_idx = paddle.empty([0], dtype=paddle.int64)

        result1 = tensor[empty_idx]
        self.assertEqual(result1.shape, [0, 5, 5])

        # Test chained empty indexing
        result2 = tensor[empty_idx][empty_idx]
        self.assertEqual(result2.shape[0], 0)

    def test_mixed_empty_nonempty_indexing(self):
        """Test alternating empty and non-empty tensor indexing."""
        tensor = paddle.zeros([10])
        empty_idx = paddle.empty([0], dtype=paddle.int64)
        normal_idx = paddle.to_tensor([0, 1, 2], dtype=paddle.int64)

        # Alternate between empty and non-empty indexing
        for i in range(5):
            if i % 2 == 0:
                result = tensor[empty_idx]
                self.assertEqual(result.shape, [0])
            else:
                result = tensor[normal_idx]
                self.assertEqual(result.shape, [3])

    def test_empty_tensor_with_different_dtypes(self):
        """Test empty tensor indexing with different data types."""
        dtypes = [
            paddle.float32,
            paddle.float64,
            paddle.int32,
            paddle.int64,
        ]

        for dtype in dtypes:
            tensor = paddle.zeros([5], dtype=dtype)
            empty_idx = paddle.empty([0], dtype=paddle.int64)

            result = tensor[empty_idx]
            self.assertEqual(result.shape, [0])
            self.assertEqual(result.dtype, dtype)

    def test_chained_triple_empty_indexing(self):
        """Test three consecutive empty indexing operations."""
        tensor = paddle.zeros([10, 10, 10])
        idx1 = paddle.empty([0], dtype=paddle.int64)
        idx2 = paddle.empty([0], dtype=paddle.int64)

        try:
            result = tensor[idx1]
            result = result[idx2]
            self.assertEqual(result.shape[0], 0)
        except Exception as e:
            self.fail(f"Chained empty indexing failed with: {e}")


class TestEmptyTensorIndexingEdgeCases(unittest.TestCase):
    """Test edge cases for empty tensor indexing."""

    def test_empty_index_on_empty_tensor(self):
        """Test indexing an empty tensor with an empty index."""
        empty_tensor = paddle.empty([0], dtype=paddle.float32)
        empty_idx = paddle.empty([0], dtype=paddle.int64)

        result = empty_tensor[empty_idx]
        self.assertEqual(result.shape, [0])

    def test_empty_tensor_holder_size(self):
        """Test that empty tensor holder is correctly initialized."""
        tensor = paddle.zeros([5])
        empty_idx = paddle.empty([0], dtype=paddle.int64)

        # Multiple indexing operations should not corrupt holder
        for _ in range(20):
            result = tensor[empty_idx]
            # Verify the tensor is still valid
            self.assertEqual(result.numel(), 0)
            self.assertTrue(result.is_tensor())


if __name__ == '__main__':
    unittest.main()
