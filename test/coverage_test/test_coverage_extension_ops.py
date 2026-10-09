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
# Target file: python/paddle/nn/functional/extension.py
# Coverage target: sequence_mask, gather_tree, temporal_shift, diag_embed
# 未覆盖行: static graph paths for some functions

import unittest

import numpy as np

import paddle
from paddle.nn.functional.extension import (
    diag_embed,
    gather_tree,
    sequence_mask,
    temporal_shift,
)


class TestSequenceMask(unittest.TestCase):
    """Test sequence_mask function.
    测试 sequence_mask 函数。"""

    def setUp(self):
        paddle.disable_static()

    def test_sequence_mask_basic(self):
        """Test sequence_mask with basic 1D input.
        测试基本一维输入的 sequence_mask。"""
        x = paddle.to_tensor([3, 1, 1, 0])
        mask = sequence_mask(x, maxlen=4)
        expected = np.array(
            [[1, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
        )
        np.testing.assert_array_equal(mask.numpy(), expected)

    def test_sequence_mask_2d_input(self):
        """Test sequence_mask with 2D input.
        测试二维输入的 sequence_mask。"""
        x = paddle.to_tensor([[3, 2], [1, 4]])
        mask = sequence_mask(x, maxlen=5)
        self.assertEqual(mask.shape, [2, 2, 5])
        # First row, first column: length 3
        np.testing.assert_array_equal(mask.numpy()[0, 0], [1, 1, 1, 0, 0])
        # Second row, second column: length 4
        np.testing.assert_array_equal(mask.numpy()[1, 1], [1, 1, 1, 1, 0])

    def test_sequence_mask_no_maxlen(self):
        """Test sequence_mask with maxlen=None (auto-computed).
        测试 maxlen=None（自动计算）的 sequence_mask。"""
        x = paddle.to_tensor([3, 1, 5])
        mask = sequence_mask(x)
        # maxlen should be max(x) = 5
        self.assertEqual(mask.shape, [3, 5])
        np.testing.assert_array_equal(mask.numpy()[2], [1, 1, 1, 1, 1])

    def test_sequence_mask_custom_dtype(self):
        """Test sequence_mask with custom dtype.
        测试自定义数据类型的 sequence_mask。"""
        x = paddle.to_tensor([2, 1, 3])
        mask = sequence_mask(x, maxlen=4, dtype="float32")
        self.assertEqual(mask.dtype, paddle.float32)
        np.testing.assert_array_equal(mask.numpy()[0], [1.0, 1.0, 0.0, 0.0])

    def test_sequence_mask_int32_dtype(self):
        """Test sequence_mask with int32 dtype.
        测试 int32 数据类型的 sequence_mask。"""
        x = paddle.to_tensor([1, 3, 2])
        mask = sequence_mask(x, maxlen=3, dtype="int32")
        self.assertEqual(mask.dtype, paddle.int32)
        np.testing.assert_array_equal(mask.numpy()[1], [1, 1, 1])

    def test_sequence_mask_float64_dtype(self):
        """Test sequence_mask with float64 dtype.
        测试 float64 数据类型的 sequence_mask。"""
        x = paddle.to_tensor([2, 0, 1])
        mask = sequence_mask(x, maxlen=3, dtype="float64")
        self.assertEqual(mask.dtype, paddle.float64)

    def test_sequence_mask_all_zeros(self):
        """Test sequence_mask with all zeros input.
        测试全零输入的 sequence_mask。"""
        x = paddle.to_tensor([0, 0, 0])
        mask = sequence_mask(x, maxlen=4)
        expected = np.zeros((3, 4), dtype=np.int64)
        np.testing.assert_array_equal(mask.numpy(), expected)

    def test_sequence_mask_all_max(self):
        """Test sequence_mask where all lengths equal maxlen.
        测试所有长度等于 maxlen 的 sequence_mask。"""
        x = paddle.to_tensor([4, 4, 4])
        mask = sequence_mask(x, maxlen=4)
        expected = np.ones((3, 4), dtype=np.int64)
        np.testing.assert_array_equal(mask.numpy(), expected)

    def test_sequence_mask_maxlen_zero_raises(self):
        """Test sequence_mask with maxlen=0 raises error.
        测试 maxlen=0 时 sequence_mask 抛出异常。"""
        x = paddle.to_tensor([0, 0])
        with self.assertRaises(RuntimeError):
            sequence_mask(x, maxlen=0)

    def test_sequence_mask_single_element(self):
        """Test sequence_mask with single element.
        测试单元素的 sequence_mask。"""
        x = paddle.to_tensor([2])
        mask = sequence_mask(x, maxlen=3)
        np.testing.assert_array_equal(mask.numpy(), [[1, 1, 0]])

    def test_sequence_mask_3d_input(self):
        """Test sequence_mask with 3D input.
        测试三维输入的 sequence_mask。"""
        x = paddle.to_tensor([[[2], [1]], [[3], [0]]])
        mask = sequence_mask(x, maxlen=4)
        self.assertEqual(mask.shape, [2, 2, 1, 4])


class TestGatherTree(unittest.TestCase):
    """Test gather_tree function.
    测试 gather_tree 函数。"""

    def setUp(self):
        paddle.disable_static()

    def test_gather_tree_basic(self):
        """Test gather_tree with basic beam search data from docstring example.
        测试文档字符串示例中基本束搜索数据的 gather_tree。"""
        ids = paddle.to_tensor(
            [[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]]
        )
        parents = paddle.to_tensor(
            [[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]]
        )
        result = gather_tree(ids, parents)
        expected = np.array(
            [[[2, 2], [1, 6]], [[3, 3], [6, 1]], [[0, 1], [9, 0]]]
        )
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_gather_tree_output_shape(self):
        """Test gather_tree output shape matches input.
        测试 gather_tree 输出形状与输入匹配。"""
        ids = paddle.randint(0, 10, [5, 3, 4])
        parents = paddle.randint(0, 4, [5, 3, 4])
        result = gather_tree(ids, parents)
        self.assertEqual(result.shape, [5, 3, 4])

    def test_gather_tree_int32(self):
        """Test gather_tree with int32 dtype.
        测试 int32 数据类型的 gather_tree。"""
        ids = paddle.to_tensor(
            [[[1, 0], [2, 1]], [[3, 2], [4, 3]]], dtype="int32"
        )
        parents = paddle.to_tensor(
            [[[0, 0], [0, 1]], [[0, 0], [1, 1]]], dtype="int32"
        )
        result = gather_tree(ids, parents)
        self.assertEqual(result.dtype, paddle.int32)
        self.assertEqual(result.shape, [2, 2, 2])

    def test_gather_tree_int64(self):
        """Test gather_tree with int64 dtype.
        测试 int64 数据类型的 gather_tree。"""
        ids = paddle.to_tensor(
            [[[1, 0], [2, 1]], [[3, 2], [4, 3]]], dtype="int64"
        )
        parents = paddle.to_tensor(
            [[[0, 0], [0, 1]], [[0, 0], [1, 1]]], dtype="int64"
        )
        result = gather_tree(ids, parents)
        self.assertEqual(result.dtype, paddle.int64)

    def test_gather_tree_identity_parents(self):
        """Test gather_tree where all parents point to themselves (identity).
        测试所有 parent 指向自身的 gather_tree（恒等映射）。"""
        ids = paddle.to_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        parents = paddle.to_tensor([[[0, 1], [0, 1]], [[0, 1], [0, 1]]])
        result = gather_tree(ids, parents)
        # With identity parents, the last step stays the same
        np.testing.assert_array_equal(
            result.numpy()[-1].tolist(), ids.numpy()[-1].tolist()
        )

    def test_gather_tree_single_beam(self):
        """Test gather_tree with beam_size=1.
        测试 beam_size=1 的 gather_tree。"""
        ids = paddle.to_tensor([[[1], [2]], [[3], [4]], [[5], [6]]])
        parents = paddle.to_tensor([[[0], [0]], [[0], [0]], [[0], [0]]])
        result = gather_tree(ids, parents)
        self.assertEqual(result.shape, [3, 2, 1])

    def test_gather_tree_invalid_ndim(self):
        """Test gather_tree raises ValueError for non-3D input.
        测试 gather_tree 对非三维输入引发 ValueError。"""
        ids = paddle.to_tensor([[1, 2], [3, 4]])
        parents = paddle.to_tensor([[0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            gather_tree(ids, parents)


class TestTemporalShift(unittest.TestCase):
    """Test temporal_shift function.
    测试 temporal_shift 函数。"""

    def setUp(self):
        paddle.disable_static()

    def test_temporal_shift_nchw(self):
        """Test temporal_shift with NCHW format.
        测试 NCHW 格式的 temporal_shift。"""
        x = paddle.randn([6, 4, 2, 2], dtype="float32")
        out = temporal_shift(x, seg_num=2, shift_ratio=0.25)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_temporal_shift_nhwc(self):
        """Test temporal_shift with NHWC format.
        测试 NHWC 格式的 temporal_shift。"""
        x = paddle.randn([6, 2, 2, 4], dtype="float32")
        out = temporal_shift(x, seg_num=2, shift_ratio=0.25, data_format="NHWC")
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_temporal_shift_custom_ratio(self):
        """Test temporal_shift with custom shift_ratio.
        测试自定义 shift_ratio 的 temporal_shift。"""
        x = paddle.randn([6, 8, 2, 2], dtype="float32")
        out = temporal_shift(x, seg_num=2, shift_ratio=0.2)
        self.assertEqual(out.shape, x.shape)

    def test_temporal_shift_larger_segments(self):
        """Test temporal_shift with more segments.
        测试更多分段的 temporal_shift。"""
        x = paddle.randn([12, 4, 2, 2], dtype="float32")
        out = temporal_shift(x, seg_num=4, shift_ratio=0.25)
        self.assertEqual(out.shape, x.shape)

    def test_temporal_shift_float16(self):
        """Test temporal_shift with float16 input.
        测试 float16 输入的 temporal_shift。"""
        x = paddle.randn([6, 4, 2, 2], dtype="float16")
        out = temporal_shift(x, seg_num=2, shift_ratio=0.25)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_temporal_shift_float64(self):
        """Test temporal_shift with float64 input.
        测试 float64 输入的 temporal_shift。"""
        x = paddle.randn([6, 4, 2, 2], dtype="float64")
        out = temporal_shift(x, seg_num=2, shift_ratio=0.25)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)

    def test_temporal_shift_invalid_format(self):
        """Test temporal_shift raises ValueError for invalid data_format.
        测试 temporal_shift 对无效 data_format 引发 ValueError。"""
        x = paddle.randn([6, 4, 2, 2], dtype="float32")
        with self.assertRaises(ValueError):
            temporal_shift(
                x, seg_num=2, shift_ratio=0.25, data_format="invalid"
            )

    def test_temporal_shift_nhwc_larger(self):
        """Test temporal_shift with NHWC and larger input.
        测试 NHWC 格式和较大输入的 temporal_shift。"""
        x = paddle.randn([12, 4, 4, 8], dtype="float32")
        out = temporal_shift(x, seg_num=3, shift_ratio=0.25, data_format="NHWC")
        self.assertEqual(out.shape, x.shape)

    def test_temporal_shift_seg_num_1(self):
        """Test temporal_shift with seg_num=1.
        测试 seg_num=1 的 temporal_shift。"""
        x = paddle.randn([3, 4, 2, 2], dtype="float32")
        out = temporal_shift(x, seg_num=1, shift_ratio=0.25)
        self.assertEqual(out.shape, x.shape)


class TestDiagEmbed(unittest.TestCase):
    """Test diag_embed function.
    测试 diag_embed 函数。"""

    def setUp(self):
        paddle.disable_static()

    def test_diag_embed_1d(self):
        """Test diag_embed with 1D input (produces 2D diagonal matrix).
        测试一维输入的 diag_embed（生成二维对角矩阵）。"""
        import warnings

        x = paddle.to_tensor([1, 2, 3])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out = diag_embed(x)
        self.assertEqual(out.shape, [3, 3])
        # Diagonal should be [1, 2, 3]
        for i in range(3):
            self.assertEqual(out.numpy()[i, i], i + 1)

    def test_diag_embed_2d(self):
        """Test diag_embed with 2D input.
        测试二维输入的 diag_embed。"""
        import warnings

        x = paddle.to_tensor([[1, 2], [3, 4]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out = diag_embed(x)
        # Should produce 3D output with diagonal 2D slices
        self.assertEqual(out.shape, [2, 2, 2])

    def test_diag_embed_with_offset(self):
        """Test diag_embed with offset parameter.
        测试带有 offset 参数的 diag_embed。
        offset shifts the diagonal, so output size expands."""
        import warnings

        x = paddle.to_tensor([1, 2, 3])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out = diag_embed(x, offset=1)
        # offset=1 with 1D input of size 3 -> output is 4x4 (max_dim + abs(offset))
        self.assertEqual(list(out.shape), [4, 4])

    def test_diag_embed_negative_offset(self):
        """Test diag_embed with negative offset.
        测试负 offset 的 diag_embed。
        Negative offset shifts diagonal down, expanding output size."""
        import warnings

        x = paddle.to_tensor([1, 2])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            out = diag_embed(x, offset=-1)
        # offset=-1 with 1D input of size 2 -> output is 3x3
        self.assertEqual(list(out.shape), [3, 3])


if __name__ == "__main__":
    unittest.main()
