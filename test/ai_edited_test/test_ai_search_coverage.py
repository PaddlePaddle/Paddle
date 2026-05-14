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

# [AUTO-GENERATED] Tests for paddle/tensor/search.py (coverage: 66.7% -> higher)
# Target file: python/paddle/tensor/search.py
# Functions: argsort, sort, msort, mode, where, where_, nonzero, index_select,
#            masked_select, topk, bucketize, searchsorted, index_sample, argwhere, _restrict_nonzero

import unittest

import paddle


class TestArgsortBasic(unittest.TestCase):
    """测试 argsort 基本功能 / Test basic argsort functionality."""

    def setUp(self):
        paddle.seed(42)
        self.x = paddle.to_tensor(
            [[5, 8, 9, 5], [0, 0, 1, 7], [6, 9, 2, 4]],
            dtype='float32',
        )

    def tearDown(self):
        pass

    def test_argsort_descending(self):
        """测试降序排序 / Test descending argsort."""
        out = paddle.argsort(self.x, axis=-1, descending=True)
        self.assertEqual(out.shape, self.x.shape)
        self.assertEqual(out.dtype, paddle.int64)

    def test_argsort_stable(self):
        """测试稳定排序 / Test stable argsort."""
        x = paddle.to_tensor([1, 0] * 40, dtype='float32')
        out = paddle.argsort(x, stable=True)
        self.assertEqual(out.shape, [80])

    def test_argsort_axis(self):
        """测试不同轴排序 / Test argsort along different axes."""
        out0 = paddle.argsort(self.x, axis=0)
        out1 = paddle.argsort(self.x, axis=1)
        self.assertEqual(out0.shape, self.x.shape)
        self.assertEqual(out1.shape, self.x.shape)

    def test_argsort_int_dtype(self):
        """测试整数类型排序 / Test argsort with int dtype."""
        x = paddle.to_tensor([[3, 1, 4], [1, 5, 9]], dtype='int64')
        out = paddle.argsort(x, axis=1)
        self.assertEqual(out.shape, x.shape)


class TestSortBasic(unittest.TestCase):
    """测试 sort 基本功能 / Test basic sort functionality."""

    def setUp(self):
        paddle.seed(42)
        self.x = paddle.to_tensor(
            [[[5, 8, 9, 5], [0, 0, 1, 7]], [[5, 2, 4, 2], [4, 7, 7, 9]]],
            dtype='float32',
        )

    def tearDown(self):
        pass

    def test_sort_default(self):
        """测试默认排序（升序）/ Test default ascending sort."""
        out = paddle.sort(self.x, axis=-1)
        self.assertEqual(out.shape, self.x.shape)
        self.assertEqual(out.dtype, paddle.float32)

    def test_sort_descending(self):
        """测试降序排序 / Test descending sort."""
        out = paddle.sort(self.x, axis=-1, descending=True)
        self.assertEqual(out.shape, self.x.shape)

    def test_sort_axis0(self):
        """测试沿 axis=0 排序 / Test sort along axis 0."""
        out = paddle.sort(self.x, axis=0)
        self.assertEqual(out.shape, self.x.shape)

    def test_sort_stable(self):
        """测试稳定排序 / Test stable sort."""
        x = paddle.to_tensor([1, 0] * 20, dtype='float32')
        out = paddle.sort(x, stable=True)
        self.assertEqual(out.shape, [40])


class TestMsort(unittest.TestCase):
    """测试 msort 功能 / Test msort functionality."""

    def test_msort_basic(self):
        """测试 msort 基本功能 / Test basic msort."""
        x = paddle.to_tensor([[5, 8, 9], [0, 0, 1]], dtype='float32')
        out = paddle.msort(x)
        self.assertEqual(out.shape, x.shape)

    def test_msort_with_out(self):
        """测试 msort 带 out 参数 / Test msort with out parameter."""
        x = paddle.to_tensor([[5, 8, 9], [0, 0, 1]], dtype='float32')
        out = paddle.empty_like(x)
        paddle.msort(x, out=out)
        self.assertEqual(out.shape, x.shape)


class TestMode(unittest.TestCase):
    """测试 mode 功能 / Test mode functionality."""

    def test_mode_default(self):
        """测试 mode 默认参数 / Test mode with default parameters."""
        x = paddle.to_tensor(
            [[[1, 2, 2], [2, 3, 3]], [[0, 5, 5], [9, 9, 0]]],
            dtype=paddle.float32,
        )
        values, indices = paddle.mode(x, 2)
        self.assertEqual(values.shape, [2, 2])

    def test_mode_keepdim(self):
        """测试 mode 保留维度 / Test mode with keepdim."""
        x = paddle.to_tensor([1, 2, 2, 3, 3, 3], dtype='float32')
        values, indices = paddle.mode(x, axis=0, keepdim=True)
        self.assertEqual(values.shape, [1])

    def test_mode_axis0(self):
        """测试 mode 沿 axis=0 / Test mode along axis 0."""
        x = paddle.to_tensor([[1, 2, 2], [2, 3, 3], [0, 5, 5]], dtype='float32')
        values, indices = paddle.mode(x, axis=0)
        self.assertEqual(values.shape, [3])


class TestWhere(unittest.TestCase):
    """测试 where 功能 / Test where functionality."""

    def test_where_scalar_xy(self):
        """测试 where 标量 x, y / Test where with scalar x and y."""
        cond = paddle.to_tensor([True, False, True, False])
        out = paddle.where(cond, 1.0, 0.0)
        self.assertEqual(out.shape, [4])

    def test_where_no_xy(self):
        """测试 where 无 x, y（等同 nonzero）/ Test where without x, y (same as nonzero)."""
        x = paddle.to_tensor([1, 0, 3, 0], dtype='float32')
        out = paddle.where(x > 0)
        self.assertIsInstance(out, tuple)

    def test_where_broadcast(self):
        """测试 where 广播 / Test where with broadcasting."""
        cond = paddle.to_tensor([[True, False], [False, True]])
        x = paddle.to_tensor([1.0, 2.0])
        y = paddle.to_tensor([[3.0], [4.0]])
        out = paddle.where(cond, x, y)
        self.assertEqual(out.shape, [2, 2])

    def test_where_none_error(self):
        """测试 where 仅传一个 None 报错 / Test where with only one None raises error."""
        cond = paddle.to_tensor([True, False])
        with self.assertRaises(ValueError):
            paddle.where(cond, x=1.0, y=None)


class TestNonzero(unittest.TestCase):
    """测试 nonzero 功能 / Test nonzero functionality."""

    def test_nonzero_2d(self):
        """测试二维 nonzero / Test 2D nonzero."""
        x = paddle.to_tensor([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype='float32')
        out = paddle.nonzero(x)
        self.assertEqual(out.shape[1], 2)

    def test_nonzero_as_tuple(self):
        """测试 nonzero as_tuple=True / Test nonzero with as_tuple=True."""
        x = paddle.to_tensor([[1, 0], [0, 2]], dtype='float32')
        result = paddle.nonzero(x, as_tuple=True)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_nonzero_1d(self):
        """测试一维 nonzero / Test 1D nonzero."""
        x = paddle.to_tensor([0, 1, 0, 3], dtype='float32')
        out = paddle.nonzero(x)
        self.assertEqual(out.shape[1], 1)


class TestArgwhere(unittest.TestCase):
    """测试 argwhere 功能 / Test argwhere functionality."""

    def test_argwhere_basic(self):
        """测试 argwhere 基本功能 / Test basic argwhere."""
        x = paddle.to_tensor([[1, 0, 0], [0, 2, 0]], dtype='float32')
        out = paddle.tensor.search.argwhere(x)
        self.assertEqual(out.shape[1], 2)


class TestIndexSelect(unittest.TestCase):
    """测试 index_select 功能 / Test index_select functionality."""

    def test_index_select_basic(self):
        """测试 index_select 基本功能 / Test basic index_select."""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        index = paddle.to_tensor([0, 1, 1], dtype='int32')
        out = paddle.index_select(x, index)
        self.assertEqual(out.shape, [3, 3])

    def test_index_select_axis1(self):
        """测试 index_select axis=1 / Test index_select along axis 1."""
        x = paddle.to_tensor([[1, 2, 3, 4]], dtype='float32')
        index = paddle.to_tensor([0, 2], dtype='int64')
        out = paddle.index_select(x, index, axis=1)
        self.assertEqual(out.shape, [1, 2])

    def test_index_select_pytorch_order(self):
        """测试 index_select PyTorch 参数顺序 / Test index_select with PyTorch argument order."""
        x = paddle.to_tensor([[1, 2, 3]], dtype='float32')
        index = paddle.to_tensor([0, 2], dtype='int64')
        out = paddle.index_select(x, 1, index)
        self.assertEqual(out.shape, [1, 2])


class TestMaskedSelect(unittest.TestCase):
    """测试 masked_select 功能 / Test masked_select functionality."""

    def test_masked_select_basic(self):
        """测试 masked_select 基本功能 / Test basic masked_select."""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32')
        mask = paddle.to_tensor([[True, False, False], [True, True, False]])
        out = paddle.masked_select(x, mask)
        self.assertEqual(out.shape, [3])

    def test_masked_select_all_true(self):
        """测试 masked_select 全选 / Test masked_select with all True mask."""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        mask = paddle.to_tensor([True, True, True])
        out = paddle.masked_select(x, mask)
        self.assertEqual(out.shape, [3])


class TestTopk(unittest.TestCase):
    """测试 topk 功能 / Test topk functionality."""

    def test_topk_basic(self):
        """测试 topk 基本功能 / Test basic topk."""
        x = paddle.to_tensor([1, 4, 5, 7], dtype='int64')
        values, indices = paddle.topk(x, k=2)
        self.assertEqual(values.shape, [2])

    def test_topk_smallest(self):
        """测试 topk 取最小值 / Test topk with largest=False."""
        x = paddle.to_tensor([1, 4, 5, 7], dtype='int64')
        values, indices = paddle.topk(x, k=2, largest=False)
        self.assertEqual(values.shape, [2])
        self.assertTrue(values[0] < values[1])

    def test_topk_axis0(self):
        """测试 topk 沿 axis=0 / Test topk along axis 0."""
        x = paddle.to_tensor([[1, 4, 5, 7], [2, 6, 2, 5]], dtype='int64')
        values, indices = paddle.topk(x, k=1, axis=0)
        self.assertEqual(values.shape, [1, 4])

    def test_topk_unsorted(self):
        """测试 topk 不排序 / Test topk with sorted=False."""
        x = paddle.to_tensor([3, 1, 4, 1, 5], dtype='int64')
        values, indices = paddle.topk(x, k=3, sorted=False)
        self.assertEqual(values.shape, [3])


class TestBucketize(unittest.TestCase):
    """测试 bucketize 功能 / Test bucketize functionality."""

    def test_bucketize_basic(self):
        """测试 bucketize 基本功能 / Test basic bucketize."""
        sorted_seq = paddle.to_tensor([2, 4, 8, 16], dtype='int32')
        x = paddle.to_tensor([[0, 8, 4, 16], [-1, 2, 8, 4]], dtype='int32')
        out = paddle.bucketize(x, sorted_seq)
        self.assertEqual(out.shape, x.shape)

    def test_bucketize_right(self):
        """测试 bucketize 右边界 / Test bucketize with right=True."""
        sorted_seq = paddle.to_tensor([2, 4, 8, 16], dtype='int32')
        x = paddle.to_tensor([0, 8, 4, 16], dtype='int32')
        out = paddle.bucketize(x, sorted_seq, right=True)
        self.assertEqual(out.shape, x.shape)

    def test_bucketize_out_int32(self):
        """测试 bucketize 输出 int32 / Test bucketize with out_int32=True."""
        sorted_seq = paddle.to_tensor([2, 4, 8, 16], dtype='int32')
        x = paddle.to_tensor([0, 8, 4], dtype='int32')
        out = paddle.bucketize(x, sorted_seq, out_int32=True)
        self.assertEqual(out.dtype, paddle.int32)

    def test_bucketize_dim_error(self):
        """测试 bucketize 非1D sorted_sequence 报错 / Test bucketize with non-1D sorted_sequence."""
        sorted_seq = paddle.to_tensor([[2, 4], [8, 16]], dtype='int32')
        x = paddle.to_tensor([0, 8], dtype='int32')
        with self.assertRaises(ValueError):
            paddle.bucketize(x, sorted_seq)


class TestSearchsorted(unittest.TestCase):
    """测试 searchsorted 功能 / Test searchsorted functionality."""

    def test_searchsorted_basic(self):
        """测试 searchsorted 基本功能 / Test basic searchsorted."""
        sorted_seq = paddle.to_tensor([1, 3, 5, 7, 9], dtype='int32')
        values = paddle.to_tensor([3, 6, 9, 10], dtype='int32')
        out = paddle.searchsorted(sorted_seq, values)
        self.assertEqual(out.shape, values.shape)

    def test_searchsorted_right(self):
        """测试 searchsorted 右边界 / Test searchsorted with right=True."""
        sorted_seq = paddle.to_tensor([1, 3, 5, 7, 9], dtype='int32')
        values = paddle.to_tensor([3, 6], dtype='int32')
        out = paddle.searchsorted(sorted_seq, values, right=True)
        self.assertEqual(out.shape, values.shape)

    def test_searchsorted_side(self):
        """测试 searchsorted 使用 side 参数 / Test searchsorted with side parameter."""
        sorted_seq = paddle.to_tensor([1, 3, 5, 7, 9], dtype='int32')
        values = paddle.to_tensor([3, 6], dtype='int32')
        out = paddle.searchsorted(sorted_seq, values, side='right')
        self.assertEqual(out.shape, values.shape)

    def test_searchsorted_2d(self):
        """测试 searchsorted 2D 序列 / Test searchsorted with 2D sorted_sequence."""
        sorted_seq = paddle.to_tensor([[1, 3, 5], [2, 4, 6]], dtype='int32')
        values = paddle.to_tensor([[3, 4], [3, 4]], dtype='int32')
        out = paddle.searchsorted(sorted_seq, values)
        self.assertEqual(out.shape, values.shape)


class TestIndexSample(unittest.TestCase):
    """测试 index_sample 功能 / Test index_sample functionality."""

    def test_index_sample_basic(self):
        """测试 index_sample 基本功能 / Test basic index_sample."""
        x = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype='float32')
        index = paddle.to_tensor([[0, 2], [1, 3]], dtype='int32')
        out = paddle.index_sample(x, index)
        self.assertEqual(out.shape, index.shape)


class TestTopPSampling(unittest.TestCase):
    """测试 top_p_sampling 功能 / Test top_p_sampling functionality."""

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "top_p_sampling requires CUDA"
    )
    def test_top_p_sampling_basic(self):
        """测试 top_p_sampling 基本功能 / Test basic top_p_sampling."""
        paddle.set_device('gpu')
        paddle.seed(2023)
        x = paddle.randn([2, 3])
        paddle.seed(2023)
        ps = paddle.randn([2])
        value, index = paddle.tensor.search.top_p_sampling(x, ps)
        self.assertEqual(value.shape[0], 2)

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "top_p_sampling requires CUDA"
    )
    def test_top_p_sampling_return_top(self):
        """测试 top_p_sampling 返回 top / Test top_p_sampling with return_top."""
        paddle.set_device('gpu')
        paddle.seed(2023)
        x = paddle.randn([2, 3])
        ps = paddle.randn([2])
        result = paddle.tensor.search.top_p_sampling(x, ps, return_top=True)
        self.assertEqual(len(result), 4)


class TestRestrictNonzero(unittest.TestCase):
    """测试 _restrict_nonzero 功能 / Test _restrict_nonzero functionality."""

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "_restrict_nonzero requires CUDA"
    )
    def test_restrict_nonzero(self):
        """测试 _restrict_nonzero 基本功能 / Test basic _restrict_nonzero."""
        x = paddle.to_tensor([False, True, False, True], dtype='bool')
        out = paddle.tensor.search._restrict_nonzero(x, 2)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 1)


if __name__ == '__main__':
    unittest.main()
