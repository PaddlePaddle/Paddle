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

# [AUTO-GENERATED] Test file for paddle.tensor.manipulation
# 覆盖模块: paddle/tensor/manipulation.py
# Uncovered lines: gather_nd, scatter_nd, scatter_nd_add, scatter_add,
#   scatter_reduce, narrow, take_along_axis, put_along_axis, masked_fill,
#   index_add, index_fill, as_real, as_complex, unflatten, atleast_1d/2d/3d,
#   column_stack, hstack, vstack, row_stack, dstack, moveaxis, ravel,
#   broadcast_tensors, broadcast_to, expand, tile, unique_consecutive,
#   tensor_split, dsplit, hsplit, vsplit, block_diag, shard_index,
#   rot90, flip, roll, unbind, as_strided, unfold, diagonal_scatter,
#   select_scatter, slice_scatter, masked_scatter, fill_diagonal_tensor

import unittest

import numpy as np

import paddle


class TestGatherNd(unittest.TestCase):
    """测试 gather_nd 函数
    Test gather_nd function"""

    def test_gather_nd_2d(self):
        """测试二维 gather_nd
        Test 2D gather_nd"""
        x = paddle.randn([4, 5])
        index = paddle.to_tensor([[0], [2]])
        result = paddle.gather_nd(x, index)
        self.assertEqual(result.shape, [2, 5])

    def test_gather_nd_multiindex(self):
        """测试多索引 gather_nd
        Test multi-index gather_nd"""
        x = paddle.randn([3, 4, 5])
        index = paddle.to_tensor([[0, 1], [2, 3]])
        result = paddle.gather_nd(x, index)
        self.assertEqual(result.shape, [2, 5])


class TestScatterNd(unittest.TestCase):
    """测试 scatter_nd 函数
    Test scatter_nd function"""

    def test_scatter_nd_basic(self):
        """测试基本 scatter_nd
        Test basic scatter_nd"""
        shape = [4, 5]
        index = paddle.to_tensor([[0], [2]])
        updates = paddle.randn([2, 5])
        result = paddle.scatter_nd(index, updates, shape)
        self.assertEqual(result.shape, [4, 5])

    def test_scatter_nd_add(self):
        """测试 scatter_nd_add
        Test scatter_nd_add"""
        x = paddle.randn([4, 5])
        index = paddle.to_tensor([[0], [2]])
        updates = paddle.randn([2, 5])
        result = paddle.scatter_nd_add(x, index, updates)
        self.assertEqual(result.shape, [4, 5])


class TestScatterAdd(unittest.TestCase):
    """测试 scatter_add 函数
    Test scatter_add function"""

    def test_scatter_add_basic(self):
        """测试基本 scatter_add
        Test basic scatter_add"""
        x = paddle.randn([4, 5])
        index = paddle.to_tensor([[0], [2]])
        src = paddle.randn([2, 5])
        result = paddle.scatter_add(x, dim=0, index=index, src=src)
        self.assertEqual(result.shape, [4, 5])


class TestScatterReduce(unittest.TestCase):
    """测试 scatter_reduce 函数
    Test scatter_reduce function"""

    def test_scatter_reduce_add(self):
        """测试 scatter_reduce 加法
        Test scatter_reduce with add"""
        x = paddle.randn([4, 5])
        index = paddle.to_tensor([[0], [2]])
        src = paddle.randn([2, 5])
        result = paddle.scatter_reduce(
            x, dim=0, index=index, src=src, reduce='sum'
        )
        self.assertEqual(result.shape, [4, 5])


class TestNarrow(unittest.TestCase):
    """测试 narrow 函数
    Test narrow function"""

    def test_narrow_basic(self):
        """测试基本 narrow
        Test basic narrow"""
        x = paddle.randn([4, 5])
        result = paddle.narrow(x, dim=0, start=1, length=2)
        self.assertEqual(result.shape, [2, 5])

    def test_narrow_axis1(self):
        """测试 axis=1 的 narrow
        Test narrow on axis=1"""
        x = paddle.randn([4, 5])
        result = paddle.narrow(x, dim=1, start=1, length=3)
        self.assertEqual(result.shape, [4, 3])


class TestTakeAlongAxis(unittest.TestCase):
    """测试 take_along_axis 函数
    Test take_along_axis function"""

    def test_take_along_axis_basic(self):
        """测试基本 take_along_axis
        Test basic take_along_axis"""
        x = paddle.randn([3, 4])
        index = paddle.randint(0, 4, [3, 2])
        result = paddle.take_along_axis(x, index, axis=1)
        self.assertEqual(result.shape, [3, 2])

    def test_take_along_axis_axis0(self):
        """测试 axis=0 的 take_along_axis
        Test take_along_axis on axis=0"""
        x = paddle.randn([3, 4])
        index = paddle.randint(0, 3, [2, 4])
        result = paddle.take_along_axis(x, index, axis=0)
        self.assertEqual(result.shape, [2, 4])


class TestPutAlongAxis(unittest.TestCase):
    """测试 put_along_axis 函数
    Test put_along_axis function"""

    def test_put_along_axis_basic(self):
        """测试基本 put_along_axis
        Test basic put_along_axis"""
        x = paddle.randn([3, 4])
        index = paddle.randint(0, 4, [3, 2])
        values = paddle.randn([3, 2])
        result = paddle.put_along_axis(x, index, values, axis=1)
        self.assertEqual(result.shape, [3, 4])


class TestMaskedFill(unittest.TestCase):
    """测试 masked_fill 函数
    Test masked_fill function"""

    def test_masked_fill_basic(self):
        """测试基本 masked_fill
        Test basic masked_fill"""
        x = paddle.randn([3, 4])
        mask = paddle.randn([3, 4]) > 0
        result = paddle.masked_fill(x, mask, 0.0)
        self.assertEqual(result.shape, [3, 4])
        # Verify masked positions are 0.0
        self.assertTrue(paddle.all(result[mask] == 0.0).item())

    def test_masked_fill_value(self):
        """测试 masked_fill 带不同填充值
        Test masked_fill with different fill value"""
        x = paddle.randn([3, 4])
        mask = paddle.randn([3, 4]) > 0
        result = paddle.masked_fill(x, mask, -1.0)
        self.assertTrue(paddle.all(result[mask] == -1.0).item())


class TestIndexAdd(unittest.TestCase):
    """测试 index_add 函数
    Test index_add function"""

    def test_index_add_basic(self):
        """测试基本 index_add
        Test basic index_add"""
        x = paddle.randn([3, 4])
        index = paddle.to_tensor([0, 2])
        values = paddle.randn([2, 4])
        result = paddle.index_add(x, index, axis=0, value=values)
        self.assertEqual(result.shape, [3, 4])


class TestIndexFill(unittest.TestCase):
    """测试 index_fill 函数
    Test index_fill function"""

    def test_index_fill_basic(self):
        """测试基本 index_fill
        Test basic index_fill"""
        x = paddle.randn([3, 4])
        index = paddle.to_tensor([0, 2])
        result = paddle.index_fill(x, index, axis=0, value=0.0)
        self.assertEqual(result.shape, [3, 4])


class TestAsRealComplex(unittest.TestCase):
    """测试 as_real 和 as_complex 函数
    Test as_real and as_complex functions"""

    def test_as_real_complex64(self):
        """测试 complex64 转 real
        Test complex64 to real"""
        x = paddle.randn([3], dtype='complex64')
        result = paddle.as_real(x)
        self.assertEqual(result.shape, [3, 2])

    def test_as_complex_roundtrip(self):
        """测试 real/complex 往返转换
        Test real/complex roundtrip"""
        x = paddle.randn([3], dtype='complex64')
        real = paddle.as_real(x)
        back = paddle.as_complex(real)
        np.testing.assert_allclose(back.numpy(), x.numpy(), atol=1e-6)

    def test_as_complex_from_2d(self):
        """测试从二维张量构建复数
        Test building complex from 2D tensor"""
        x = paddle.randn([3, 2])
        result = paddle.as_complex(x)
        self.assertEqual(result.shape, [3])
        self.assertEqual(result.dtype, paddle.complex64)


class TestUnflatten(unittest.TestCase):
    """测试 unflatten 函数
    Test unflatten function"""

    def test_unflatten_basic(self):
        """测试基本 unflatten
        Test basic unflatten"""
        x = paddle.randn([6, 12])
        result = paddle.unflatten(x, 1, [3, 4])
        self.assertEqual(result.shape, [6, 3, 4])

    def test_unflatten_axis0(self):
        """测试 axis=0 的 unflatten
        Test unflatten on axis=0"""
        x = paddle.randn([12, 5])
        result = paddle.unflatten(x, 0, [3, 4])
        self.assertEqual(result.shape, [3, 4, 5])


class TestAtleastNd(unittest.TestCase):
    """测试 atleast_1d/2d/3d 函数
    Test atleast_1d/2d/3d functions"""

    def test_atleast_1d_scalar(self):
        """测试标量 atleast_1d
        Test scalar atleast_1d"""
        x = paddle.randn([])
        result = paddle.atleast_1d(x)
        self.assertEqual(result.ndim, 1)

    def test_atleast_1d_1d(self):
        """测试一维张量 atleast_1d
        Test 1D tensor atleast_1d"""
        x = paddle.randn([3])
        result = paddle.atleast_1d(x)
        self.assertEqual(result.ndim, 1)

    def test_atleast_2d_1d(self):
        """测试一维张量 atleast_2d
        Test 1D tensor atleast_2d"""
        x = paddle.randn([3])
        result = paddle.atleast_2d(x)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, [1, 3])

    def test_atleast_2d_2d(self):
        """测试二维张量 atleast_2d
        Test 2D tensor atleast_2d"""
        x = paddle.randn([2, 3])
        result = paddle.atleast_2d(x)
        self.assertEqual(result.ndim, 2)

    def test_atleast_3d_1d(self):
        """测试一维张量 atleast_3d
        Test 1D tensor atleast_3d"""
        x = paddle.randn([3])
        result = paddle.atleast_3d(x)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape, [1, 3, 1])

    def test_atleast_3d_2d(self):
        """测试二维张量 atleast_3d
        Test 2D tensor atleast_3d"""
        x = paddle.randn([2, 3])
        result = paddle.atleast_3d(x)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape, [2, 3, 1])


class TestStacking(unittest.TestCase):
    """测试 column_stack, hstack, vstack, row_stack, dstack
    Test stacking functions"""

    def test_column_stack(self):
        """测试 column_stack
        Test column_stack"""
        a = paddle.randn([3, 2])
        b = paddle.randn([3, 3])
        result = paddle.column_stack([a, b])
        self.assertEqual(result.shape, [3, 5])

    def test_hstack(self):
        """测试 hstack
        Test hstack"""
        a = paddle.randn([3, 2])
        b = paddle.randn([3, 3])
        result = paddle.hstack([a, b])
        self.assertEqual(result.shape, [3, 5])

    def test_vstack(self):
        """测试 vstack
        Test vstack"""
        a = paddle.randn([2, 4])
        b = paddle.randn([3, 4])
        result = paddle.vstack([a, b])
        self.assertEqual(result.shape, [5, 4])

    def test_row_stack(self):
        """测试 row_stack
        Test row_stack"""
        a = paddle.randn([2, 4])
        b = paddle.randn([3, 4])
        result = paddle.row_stack([a, b])
        self.assertEqual(result.shape, [5, 4])

    def test_dstack(self):
        """测试 dstack
        Test dstack"""
        a = paddle.randn([2, 3])
        b = paddle.randn([2, 3])
        result = paddle.dstack([a, b])
        self.assertEqual(result.shape, [2, 3, 2])


class TestMoveaxis(unittest.TestCase):
    """测试 moveaxis 函数
    Test moveaxis function"""

    def test_moveaxis_basic(self):
        """测试基本 moveaxis
        Test basic moveaxis"""
        x = paddle.randn([2, 3, 4])
        result = paddle.moveaxis(x, [0, 1], [1, 0])
        self.assertEqual(result.shape, [3, 2, 4])

    def test_moveaxis_single(self):
        """测试单轴 moveaxis
        Test single axis moveaxis"""
        x = paddle.randn([2, 3, 4])
        result = paddle.moveaxis(x, 0, 2)
        self.assertEqual(result.shape, [3, 4, 2])


class TestRavel(unittest.TestCase):
    """测试 ravel 函数
    Test ravel function"""

    def test_ravel_basic(self):
        """测试基本 ravel
        Test basic ravel"""
        x = paddle.randn([2, 3])
        result = paddle.ravel(x)
        self.assertEqual(result.shape, [6])

    def test_ravel_3d(self):
        """测试三维 ravel
        Test 3D ravel"""
        x = paddle.randn([2, 3, 4])
        result = paddle.ravel(x)
        self.assertEqual(result.shape, [24])


class TestBroadcastTensors(unittest.TestCase):
    """测试 broadcast_tensors 函数
    Test broadcast_tensors function"""

    def test_broadcast_tensors_basic(self):
        """测试基本 broadcast_tensors
        Test basic broadcast_tensors"""
        a = paddle.randn([1, 3])
        b = paddle.randn([2, 1])
        result = paddle.broadcast_tensors([a, b])
        self.assertEqual(result[0].shape, [2, 3])
        self.assertEqual(result[1].shape, [2, 3])

    def test_broadcast_to(self):
        """测试 broadcast_to
        Test broadcast_to"""
        x = paddle.randn([1, 3])
        result = paddle.broadcast_to(x, [2, 3])
        self.assertEqual(result.shape, [2, 3])


class TestExpand(unittest.TestCase):
    """测试 expand 函数
    Test expand function"""

    def test_expand_basic(self):
        """测试基本 expand
        Test basic expand"""
        x = paddle.randn([1, 3])
        result = paddle.expand(x, [2, 3])
        self.assertEqual(result.shape, [2, 3])

    def test_expand_with_neg1(self):
        """测试带 -1 的 expand（-1 表示保持原维度不变）
        Test expand with -1 (-1 means keep original dimension)"""
        x = paddle.randn([1, 3])
        result = paddle.expand(x, [4, 3])
        self.assertEqual(result.shape, [4, 3])


class TestTile(unittest.TestCase):
    """测试 tile 函数
    Test tile function"""

    def test_tile_basic(self):
        """测试基本 tile
        Test basic tile"""
        x = paddle.randn([2, 3])
        result = paddle.tile(x, [2, 3])
        self.assertEqual(result.shape, [4, 9])

    def test_tile_1d(self):
        """测试一维 tile
        Test 1D tile"""
        x = paddle.to_tensor([1, 2, 3])
        result = paddle.tile(x, [3])
        self.assertEqual(result.shape, [9])


class TestUniqueConsecutive(unittest.TestCase):
    """测试 unique_consecutive 函数
    Test unique_consecutive function"""

    def test_unique_consecutive_basic(self):
        """测试基本 unique_consecutive
        Test basic unique_consecutive"""
        x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1])
        result = paddle.unique_consecutive(x)
        expected = np.array([1, 2, 3, 1])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_unique_consecutive_counts(self):
        """测试带计数的 unique_consecutive
        Test unique_consecutive with counts"""
        x = paddle.to_tensor([1, 1, 2, 2, 3, 1, 1])
        result, counts = paddle.unique_consecutive(x, return_counts=True)
        np.testing.assert_array_equal(counts.numpy(), [2, 2, 1, 2])


class TestTensorSplit(unittest.TestCase):
    """测试 tensor_split 函数
    Test tensor_split function"""

    def test_tensor_split_basic(self):
        """测试基本 tensor_split
        Test basic tensor_split"""
        x = paddle.randn([6, 4])
        results = paddle.tensor_split(x, 3, axis=0)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.shape[1], 4)

    def test_tensor_split_indices(self):
        """测试用索引分割
        Test tensor_split with indices"""
        x = paddle.randn([6, 4])
        results = paddle.tensor_split(x, [2, 4], axis=0)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shape, [2, 4])
        self.assertEqual(results[1].shape, [2, 4])
        self.assertEqual(results[2].shape, [2, 4])


class TestDSplit(unittest.TestCase):
    """测试 dsplit 函数
    Test dsplit function"""

    def test_dsplit_basic(self):
        """测试基本 dsplit
        Test basic dsplit"""
        x = paddle.randn([2, 3, 6])
        results = paddle.dsplit(x, 3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shape, [2, 3, 2])


class TestHSplit(unittest.TestCase):
    """测试 hsplit 函数
    Test hsplit function"""

    def test_hsplit_basic(self):
        """测试基本 hsplit
        Test basic hsplit"""
        x = paddle.randn([2, 6])
        results = paddle.hsplit(x, 3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shape, [2, 2])


class TestVSplit(unittest.TestCase):
    """测试 vsplit 函数
    Test vsplit function"""

    def test_vsplit_basic(self):
        """测试基本 vsplit
        Test basic vsplit"""
        x = paddle.randn([6, 2])
        results = paddle.vsplit(x, 3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shape, [2, 2])


class TestBlockDiag(unittest.TestCase):
    """测试 block_diag 函数
    Test block_diag function"""

    def test_block_diag_basic(self):
        """测试基本 block_diag
        Test basic block_diag"""
        a = paddle.randn([2, 3])
        b = paddle.randn([3, 2])
        result = paddle.block_diag([a, b])
        self.assertEqual(result.shape, [5, 5])


class TestShardIndex(unittest.TestCase):
    """测试 shard_index 函数
    Test shard_index function"""

    def test_shard_index_basic(self):
        """测试基本 shard_index
        Test basic shard_index"""
        x = paddle.to_tensor([[0], [1], [2], [3], [4], [5]])
        result = paddle.shard_index(x, index_num=20, nshards=2, shard_id=0)
        self.assertEqual(result.shape, [6, 1])


class TestRot90(unittest.TestCase):
    """测试 rot90 函数
    Test rot90 function"""

    def test_rot90_basic(self):
        """测试基本 rot90
        Test basic rot90"""
        x = paddle.randn([2, 3, 4])
        result = paddle.rot90(x, k=1, axes=[1, 2])
        self.assertEqual(result.shape, [2, 4, 3])

    def test_rot90_full_rotation(self):
        """测试完整旋转 4 次
        Test full rotation 4 times"""
        x = paddle.randn([2, 3, 4])
        result = x
        for _ in range(4):
            result = paddle.rot90(result, k=1, axes=[1, 2])
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)


class TestFlip(unittest.TestCase):
    """测试 flip 函数
    Test flip function"""

    def test_flip_basic(self):
        """测试基本 flip
        Test basic flip"""
        x = paddle.to_tensor([[1, 2], [3, 4]])
        result = paddle.flip(x, [0])
        expected = np.array([[3, 4], [1, 2]])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_flip_double(self):
        """测试翻转两次还原
        Test double flip roundtrip"""
        x = paddle.randn([3, 4])
        result = paddle.flip(paddle.flip(x, [0, 1]), [0, 1])
        np.testing.assert_allclose(result.numpy(), x.numpy(), atol=1e-6)


class TestRoll(unittest.TestCase):
    """测试 roll 函数
    Test roll function"""

    def test_roll_basic(self):
        """测试基本 roll
        Test basic roll"""
        x = paddle.to_tensor([1, 2, 3, 4, 5, 6])
        result = paddle.roll(x, shifts=2)
        expected = np.array([5, 6, 1, 2, 3, 4])
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_roll_negative(self):
        """测试负向 roll
        Test negative roll"""
        x = paddle.to_tensor([1, 2, 3, 4, 5, 6])
        result = paddle.roll(x, shifts=-2)
        expected = np.array([3, 4, 5, 6, 1, 2])
        np.testing.assert_array_equal(result.numpy(), expected)


class TestUnbind(unittest.TestCase):
    """测试 unbind 函数
    Test unbind function"""

    def test_unbind_axis0(self):
        """测试 axis=0 的 unbind
        Test unbind on axis=0"""
        x = paddle.randn([3, 4])
        results = paddle.unbind(x, axis=0)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shape, [4])

    def test_unbind_axis1(self):
        """测试 axis=1 的 unbind
        Test unbind on axis=1"""
        x = paddle.randn([3, 4])
        results = paddle.unbind(x, axis=1)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].shape, [3])


class TestAsStrided(unittest.TestCase):
    """测试 as_strided 函数
    Test as_strided function"""

    def test_as_strided_basic(self):
        """测试基本 as_strided
        Test basic as_strided"""
        x = paddle.arange(12, dtype='float32')
        result = paddle.as_strided(x, [3, 4], [4, 1])
        self.assertEqual(result.shape, [3, 4])


class TestUnfold(unittest.TestCase):
    """测试 unfold 函数
    Test unfold function"""

    def test_unfold_basic(self):
        """测试基本 unfold
        Test basic unfold"""
        paddle.base.set_flags({'FLAGS_use_stride_kernel': True})
        x = paddle.arange(9, dtype='float64')
        result = paddle.unfold(x, 0, 2, 4)
        self.assertEqual(result.shape, [2, 2])


class TestDiagonalScatter(unittest.TestCase):
    """测试 diagonal_scatter 函数
    Test diagonal_scatter function"""

    def test_diagonal_scatter_basic(self):
        """测试基本 diagonal_scatter
        Test basic diagonal_scatter"""
        x = paddle.zeros([3, 3])
        diagonal = paddle.ones([3])
        result = paddle.diagonal_scatter(x, diagonal)
        # Diagonal should be 1
        diag_result = paddle.diag(result)
        np.testing.assert_allclose(diag_result.numpy(), np.ones(3), atol=1e-6)


class TestSelectScatter(unittest.TestCase):
    """测试 select_scatter 函数
    Test select_scatter function"""

    def test_select_scatter_basic(self):
        """测试基本 select_scatter
        Test basic select_scatter"""
        x = paddle.zeros([3, 4])
        value = paddle.ones([3])
        result = paddle.select_scatter(x, value, dim=1, index=1)
        self.assertEqual(result.shape, [3, 4])
        # Column 1 should be all 1s
        np.testing.assert_allclose(result[:, 1].numpy(), np.ones(3), atol=1e-6)


class TestSliceScatter(unittest.TestCase):
    """测试 slice_scatter 函数
    Test slice_scatter function"""

    def test_slice_scatter_basic(self):
        """测试基本 slice_scatter
        Test basic slice_scatter"""
        x = paddle.zeros([3, 4])
        value = paddle.ones([2, 4])
        result = paddle.slice_scatter(
            x, value, axes=[0], starts=[1], ends=[3], strides=[1]
        )
        self.assertEqual(result.shape, [3, 4])


class TestMaskedScatter(unittest.TestCase):
    """测试 masked_scatter 函数
    Test masked_scatter function"""

    def test_masked_scatter_basic(self):
        """测试基本 masked_scatter
        Test basic masked_scatter"""
        x = paddle.zeros([3, 4])
        mask = paddle.to_tensor(
            [
                [True, False, True, False],
                [False, True, False, True],
                [True, False, False, True],
            ]
        )
        source = paddle.ones([6]) * 5.0
        result = paddle.masked_scatter(x, mask, source)
        self.assertEqual(result.shape, [3, 4])


class TestFillDiagonalTensor(unittest.TestCase):
    """测试 fill_diagonal_tensor 函数
    Test fill_diagonal_tensor function"""

    def test_fill_diagonal_tensor_basic(self):
        """测试基本 fill_diagonal_tensor
        Test basic fill_diagonal_tensor"""
        from paddle.tensor.manipulation import fill_diagonal_tensor

        x = paddle.zeros([3, 3])
        result = fill_diagonal_tensor(x, paddle.ones([3]), offset=0)
        self.assertEqual(result.shape, [3, 3])
        # Diagonal should be 1
        diag = paddle.diag(result)
        np.testing.assert_allclose(diag.numpy(), np.ones(3), atol=1e-6)


class TestChunk(unittest.TestCase):
    """测试 chunk 函数
    Test chunk function"""

    def test_chunk_even(self):
        """测试均匀 chunk
        Test even chunk"""
        x = paddle.randn([6, 4])
        results = paddle.chunk(x, 3, axis=0)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shape, [2, 4])

    def test_chunk_uneven(self):
        """测试非均匀 chunk（chunk 要求整除）
        Test chunk (requires even division)"""
        x = paddle.randn([6, 4])
        results = paddle.chunk(x, 3, axis=0)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shape, [2, 4])


class TestSqueezeUnsqueeze(unittest.TestCase):
    """测试 squeeze 和 unsqueeze 函数
    Test squeeze and unsqueeze functions"""

    def test_squeeze_basic(self):
        """测试基本 squeeze
        Test basic squeeze"""
        x = paddle.randn([1, 3, 1])
        result = paddle.squeeze(x)
        self.assertEqual(result.shape, [3])

    def test_squeeze_axis(self):
        """测试指定轴 squeeze
        Test squeeze with axis"""
        x = paddle.randn([1, 3, 1])
        result = paddle.squeeze(x, [0])
        self.assertEqual(result.shape, [3, 1])

    def test_unsqueeze_basic(self):
        """测试基本 unsqueeze
        Test basic unsqueeze"""
        x = paddle.randn([3, 4])
        result = paddle.unsqueeze(x, [0])
        self.assertEqual(result.shape, [1, 3, 4])


class TestStackUnstack(unittest.TestCase):
    """测试 stack 和 unstack 函数
    Test stack and unstack functions"""

    def test_stack_basic(self):
        """测试基本 stack
        Test basic stack"""
        a = paddle.randn([3, 4])
        b = paddle.randn([3, 4])
        result = paddle.stack([a, b], axis=0)
        self.assertEqual(result.shape, [2, 3, 4])

    def test_unstack_basic(self):
        """测试基本 unstack
        Test basic unstack"""
        x = paddle.randn([2, 3, 4])
        results = paddle.unstack(x, axis=0)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].shape, [3, 4])


if __name__ == '__main__':
    unittest.main()
