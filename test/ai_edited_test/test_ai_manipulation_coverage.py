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

# [AUTO-GENERATED] Tests for paddle/tensor/manipulation.py (coverage: 73.1% -> higher)
# Target file: python/paddle/tensor/manipulation.py
# Functions: cast, slice, narrow, unstack, rot90, flatten, ravel, flatten_,
#            stack, hstack, vstack, dstack, column_stack, split, tensor_split,
#            hsplit, dsplit, vsplit, squeeze, unsqueeze, gather, unbind,
#            chunk, tile, repeat, broadcast_to, expand, reshape, roll,
#            flip, flip_, unique_consecutive, unique, index_add, unflatten,
#            as_strided, view, view_as, unfold, index_fill, block_diag

import unittest

import numpy as np

import paddle


class TestCast(unittest.TestCase):
    """测试 cast 功能 / Test cast functionality."""

    def test_cast_float32_to_int32(self):
        """测试 float32 转 int32 / Test float32 to int32 cast."""
        x = paddle.to_tensor([1.5, 2.7, 3.3])
        out = paddle.cast(x, 'int32')
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3])

    def test_cast_int32_to_float64(self):
        """测试 int32 转 float64 / Test int32 to float64 cast."""
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        out = paddle.cast(x, 'float64')
        self.assertEqual(out.dtype, paddle.float64)

    def test_cast_to_bool(self):
        """测试转为 bool / Test cast to bool."""
        x = paddle.to_tensor([0, 1, 2], dtype='int32')
        out = paddle.cast(x, 'bool')
        np.testing.assert_array_equal(out.numpy(), [False, True, True])


class TestSlice(unittest.TestCase):
    """测试 slice 功能 / Test slice functionality."""

    def test_slice_basic(self):
        """测试 slice 基本功能 / Test basic slice."""
        x = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype='int32')
        out = paddle.slice(x, [0, 1], [0, 0], [2, 2])
        np.testing.assert_array_equal(out.numpy(), [[1, 2], [5, 6]])


class TestNarrow(unittest.TestCase):
    """测试 narrow 功能 / Test narrow functionality."""

    def test_narrow_basic(self):
        """测试 narrow 基本功能 / Test basic narrow."""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='int32')
        # narrow is a tensor method
        out = x.narrow(1, 1, 2)
        np.testing.assert_array_equal(out.numpy(), [[2, 3], [5, 6]])


class TestUnstack(unittest.TestCase):
    """测试 unstack 功能 / Test unstack functionality."""

    def test_unstack_axis0(self):
        """测试 unstack 沿 axis=0 / Test unstack along axis 0."""
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int32')
        out = paddle.unstack(x, axis=0)
        self.assertEqual(len(out), 2)
        np.testing.assert_array_equal(out[0].numpy(), [1, 2])

    def test_unstack_axis1(self):
        """测试 unstack 沿 axis=1 / Test unstack along axis 1."""
        x = paddle.to_tensor([[1, 2, 3]], dtype='int32')
        out = paddle.unstack(x, axis=1)
        self.assertEqual(len(out), 3)


class TestRot90(unittest.TestCase):
    """测试 rot90 功能 / Test rot90 functionality."""

    def test_rot90_k1(self):
        """测试 rot90 k=1 / Test rot90 with k=1."""
        x = paddle.arange(4).reshape([2, 2]).astype('float32')
        out = paddle.rot90(x, k=1)
        expected = np.rot90(np.arange(4).reshape(2, 2), k=1)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_rot90_k2(self):
        """测试 rot90 k=2 / Test rot90 with k=2."""
        x = paddle.arange(4).reshape([2, 2]).astype('float32')
        out = paddle.rot90(x, k=2)
        expected = np.rot90(np.arange(4).reshape(2, 2), k=2)
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_rot90_axes(self):
        """测试 rot90 指定轴 / Test rot90 with specified axes."""
        x = paddle.arange(8).reshape([2, 2, 2]).astype('float32')
        out = paddle.rot90(x, k=1, axes=[1, 2])
        self.assertEqual(out.shape, [2, 2, 2])


class TestFlatten(unittest.TestCase):
    """测试 flatten 功能 / Test flatten functionality."""

    def test_flatten_basic(self):
        """测试 flatten 基本功能 / Test basic flatten."""
        x = paddle.randn([2, 3, 4])
        out = paddle.flatten(x)
        self.assertEqual(out.shape, [24])

    def test_flatten_axis(self):
        """测试 flatten 指定轴 / Test flatten with specified axis."""
        x = paddle.randn([2, 3, 4])
        out = paddle.flatten(x, start_axis=1, stop_axis=2)
        self.assertEqual(out.shape, [2, 12])

    def test_flatten_start_only(self):
        """测试 flatten 仅指定 start / Test flatten with only start_axis."""
        x = paddle.randn([2, 3, 4])
        out = paddle.flatten(x, start_axis=1)
        self.assertEqual(out.shape, [2, 12])


class TestRavel(unittest.TestCase):
    """测试 ravel 功能 / Test ravel functionality."""

    def test_ravel_basic(self):
        """测试 ravel 基本功能 / Test basic ravel."""
        x = paddle.randn([2, 3, 4])
        out = paddle.ravel(x)
        self.assertEqual(out.shape, [24])


class TestFlattenInplace(unittest.TestCase):
    """测试 flatten_ 功能 / Test flatten_ functionality."""

    def test_flatten_inplace_basic(self):
        """测试 flatten_ 基本功能 / Test basic flatten_."""
        x = paddle.randn([2, 3, 4])
        out = x.flatten_(1)
        self.assertEqual(out.shape, [2, 12])


class TestStack(unittest.TestCase):
    """测试 stack 功能 / Test stack functionality."""

    def test_stack_basic(self):
        """测试 stack 基本功能 / Test basic stack."""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([4, 5, 6])
        out = paddle.stack([x, y])
        self.assertEqual(out.shape, [2, 3])

    def test_stack_axis(self):
        """测试 stack 指定轴 / Test stack with specified axis."""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([4, 5, 6])
        out = paddle.stack([x, y], axis=1)
        self.assertEqual(out.shape, [3, 2])


class TestHstack(unittest.TestCase):
    """测试 hstack 功能 / Test hstack functionality."""

    def test_hstack_1d(self):
        """测试 hstack 一维 / Test hstack with 1D tensors."""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([4, 5])
        out = paddle.hstack([x, y])
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3, 4, 5])

    def test_hstack_2d(self):
        """测试 hstack 二维 / Test hstack with 2D tensors."""
        x = paddle.to_tensor([[1], [2]])
        y = paddle.to_tensor([[3], [4]])
        out = paddle.hstack([x, y])
        np.testing.assert_array_equal(out.numpy(), [[1, 3], [2, 4]])


class TestVstack(unittest.TestCase):
    """测试 vstack 功能 / Test vstack functionality."""

    def test_vstack_1d(self):
        """测试 vstack 一维 / Test vstack with 1D tensors."""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([4, 5, 6])
        out = paddle.vstack([x, y])
        self.assertEqual(out.shape, [2, 3])

    def test_vstack_2d(self):
        """测试 vstack 二维 / Test vstack with 2D tensors."""
        x = paddle.to_tensor([[1, 2]])
        y = paddle.to_tensor([[3, 4]])
        out = paddle.vstack([x, y])
        np.testing.assert_array_equal(out.numpy(), [[1, 2], [3, 4]])


class TestDstack(unittest.TestCase):
    """测试 dstack 功能 / Test dstack functionality."""

    def test_dstack_basic(self):
        """测试 dstack 基本功能 / Test basic dstack."""
        x = paddle.to_tensor([[1, 2], [3, 4]])
        y = paddle.to_tensor([[5, 6], [7, 8]])
        out = paddle.dstack([x, y])
        self.assertEqual(out.shape, [2, 2, 2])


class TestColumnStack(unittest.TestCase):
    """测试 column_stack 功能 / Test column_stack functionality."""

    def test_column_stack_1d(self):
        """测试 column_stack 一维 / Test column_stack with 1D tensors."""
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([4, 5, 6])
        out = paddle.column_stack([x, y])
        self.assertEqual(out.shape, [3, 2])

    def test_column_stack_2d(self):
        """测试 column_stack 二维 / Test column_stack with 2D tensors."""
        x = paddle.to_tensor([[1], [2]])
        y = paddle.to_tensor([[3], [4]])
        out = paddle.column_stack([x, y])
        self.assertEqual(out.shape, [2, 2])


class TestSplit(unittest.TestCase):
    """测试 split 功能 / Test split functionality."""

    def test_split_sections(self):
        """测试 split 指定段数 / Test split with num_or_sections."""
        x = paddle.to_tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype='int32')
        out = paddle.split(x, num_or_sections=2, axis=1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, [2, 2])

    def test_split_list(self):
        """测试 split 指定列表 / Test split with list of sections."""
        x = paddle.to_tensor([[1, 2, 3, 4, 5]], dtype='int32')
        out = paddle.split(x, num_or_sections=[2, 3], axis=1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, [1, 2])


class TestTensorSplit(unittest.TestCase):
    """测试 tensor_split 功能 / Test tensor_split functionality."""

    def test_tensor_split_sections(self):
        """测试 tensor_split 指定段数 / Test tensor_split with sections."""
        x = paddle.arange(6).reshape([2, 3]).astype('int32')
        out = paddle.tensor_split(x, sections=3, axis=1)
        self.assertEqual(len(out), 3)

    def test_tensor_split_indices(self):
        """测试 tensor_split 指定索引 / Test tensor_split with indices."""
        x = paddle.arange(6).reshape([2, 3]).astype('int32')
        out = paddle.tensor_split(x, indices_or_sections=[1, 3], axis=1)
        self.assertEqual(len(out), 3)


class TestHsplit(unittest.TestCase):
    """测试 hsplit 功能 / Test hsplit functionality."""

    def test_hsplit_basic(self):
        """测试 hsplit 基本功能 / Test basic hsplit."""
        x = paddle.arange(6).reshape([2, 3]).astype('int32')
        out = paddle.hsplit(x, 3)
        self.assertEqual(len(out), 3)


class TestVsplit(unittest.TestCase):
    """测试 vsplit 功能 / Test vsplit functionality."""

    def test_vsplit_basic(self):
        """测试 vsplit 基本功能 / Test basic vsplit."""
        x = paddle.arange(6).reshape([3, 2]).astype('int32')
        out = paddle.vsplit(x, 3)
        self.assertEqual(len(out), 3)


class TestDsplit(unittest.TestCase):
    """测试 dsplit 功能 / Test dsplit functionality."""

    def test_dsplit_basic(self):
        """测试 dsplit 基本功能 / Test basic dsplit."""
        x = paddle.arange(6).reshape([1, 2, 3]).astype('int32')
        out = paddle.dsplit(x, 3)
        self.assertEqual(len(out), 3)


class TestSqueeze(unittest.TestCase):
    """测试 squeeze 功能 / Test squeeze functionality."""

    def test_squeeze_basic(self):
        """测试 squeeze 基本功能 / Test basic squeeze."""
        x = paddle.randn([1, 3, 1, 5])
        out = paddle.squeeze(x)
        self.assertEqual(out.shape, [3, 5])

    def test_squeeze_axis(self):
        """测试 squeeze 指定轴 / Test squeeze with specified axis."""
        x = paddle.randn([1, 3, 1, 5])
        out = paddle.squeeze(x, axis=0)
        self.assertEqual(out.shape, [3, 1, 5])

    def test_squeeze_multi_axis(self):
        """测试 squeeze 多轴 / Test squeeze with multiple axes."""
        x = paddle.randn([1, 3, 1, 5])
        out = paddle.squeeze(x, axis=[0, 2])
        self.assertEqual(out.shape, [3, 5])


class TestUnsqueeze(unittest.TestCase):
    """测试 unsqueeze 功能 / Test unsqueeze functionality."""

    def test_unsqueeze_basic(self):
        """测试 unsqueeze 基本功能 / Test basic unsqueeze."""
        x = paddle.randn([3, 4])
        out = paddle.unsqueeze(x, axis=0)
        self.assertEqual(out.shape, [1, 3, 4])

    def test_unsqueeze_negative(self):
        """测试 unsqueeze 负轴 / Test unsqueeze with negative axis."""
        x = paddle.randn([3, 4])
        out = paddle.unsqueeze(x, axis=-1)
        self.assertEqual(out.shape, [3, 4, 1])


class TestGather(unittest.TestCase):
    """测试 gather 功能 / Test gather functionality."""

    def test_gather_basic(self):
        """测试 gather 基本功能 / Test basic gather."""
        x = paddle.to_tensor([[1, 2], [3, 4]], dtype='int32')
        index = paddle.to_tensor([0, 1], dtype='int64')
        out = paddle.gather(x, index)
        self.assertEqual(out.shape, [2, 2])


class TestUnbind(unittest.TestCase):
    """测试 unbind 功能 / Test unbind functionality."""

    def test_unbind_basic(self):
        """测试 unbind 基本功能 / Test basic unbind."""
        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='int32')
        out = paddle.unbind(x, axis=0)
        self.assertEqual(len(out), 2)


class TestChunk(unittest.TestCase):
    """测试 chunk 功能 / Test chunk functionality."""

    def test_chunk_basic(self):
        """测试 chunk 基本功能 / Test basic chunk."""
        x = paddle.arange(12).reshape([3, 4]).astype('int32')
        out = paddle.chunk(x, chunks=2, axis=1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].shape, [3, 2])


class TestTile(unittest.TestCase):
    """测试 tile 功能 / Test tile functionality."""

    def test_tile_basic(self):
        """测试 tile 基本功能 / Test basic tile."""
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        out = paddle.tile(x, repeat_times=[3])
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3, 1, 2, 3, 1, 2, 3])

    def test_tile_2d(self):
        """测试 tile 二维 / Test tile with 2D repeat."""
        x = paddle.to_tensor([[1, 2]], dtype='int32')
        out = paddle.tile(x, repeat_times=[2, 3])
        self.assertEqual(out.shape, [2, 6])


class TestRepeatInterleave(unittest.TestCase):
    """测试 repeat_interleave 功能 / Test repeat_interleave functionality (via repeat)."""

    def test_repeat_basic(self):
        """测试 repeat 基本功能 / Test basic repeat."""
        x = paddle.to_tensor([1, 2, 3], dtype='int32')
        out = paddle.tile(x, repeat_times=[3])
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3, 1, 2, 3, 1, 2, 3])


class TestBroadcastTo(unittest.TestCase):
    """测试 broadcast_to 功能 / Test broadcast_to functionality."""

    def test_broadcast_to_basic(self):
        """测试 broadcast_to 基本功能 / Test basic broadcast_to."""
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        out = paddle.broadcast_to(x, shape=[3, 3])
        self.assertEqual(out.shape, [3, 3])


class TestExpand(unittest.TestCase):
    """测试 expand 功能 / Test expand functionality."""

    def test_expand_basic(self):
        """测试 expand 基本功能 / Test basic expand."""
        x = paddle.to_tensor([[1], [2]], dtype='int32')
        out = x.expand([2, 3])
        np.testing.assert_array_equal(out.numpy(), [[1, 1, 1], [2, 2, 2]])


class TestReshape(unittest.TestCase):
    """测试 reshape 功能 / Test reshape functionality."""

    def test_reshape_basic(self):
        """测试 reshape 基本功能 / Test basic reshape."""
        x = paddle.randn([2, 3, 4])
        out = paddle.reshape(x, [6, 4])
        self.assertEqual(out.shape, [6, 4])

    def test_reshape_negative(self):
        """测试 reshape 负数推断 / Test reshape with negative dimension."""
        x = paddle.randn([2, 3, 4])
        out = paddle.reshape(x, [6, -1])
        self.assertEqual(out.shape, [6, 4])


class TestRoll(unittest.TestCase):
    """测试 roll 功能 / Test roll functionality."""

    def test_roll_basic(self):
        """测试 roll 基本功能 / Test basic roll."""
        x = paddle.arange(10).astype('float32')
        out = paddle.roll(x, shifts=2)
        np.testing.assert_array_almost_equal(
            out.numpy(), [8.0, 9.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        )

    def test_roll_axis(self):
        """测试 roll 指定轴 / Test roll with specified axis."""
        x = paddle.arange(12).reshape([3, 4]).astype('float32')
        out = paddle.roll(x, shifts=1, axis=0)
        self.assertEqual(out.shape, [3, 4])


class TestFlip(unittest.TestCase):
    """测试 flip 功能 / Test flip functionality."""

    def test_flip_basic(self):
        """测试 flip 基本功能 / Test basic flip."""
        x = paddle.arange(6).reshape([2, 3]).astype('float32')
        out = paddle.flip(x, axis=[0])
        np.testing.assert_array_almost_equal(
            out.numpy(), [[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]
        )

    def test_flip_all_axes(self):
        """测试 flip 全轴 / Test flip along all axes."""
        x = paddle.arange(6).reshape([2, 3]).astype('float32')
        out = paddle.flip(x, axis=[0, 1])
        np.testing.assert_array_almost_equal(
            out.numpy(), [[5.0, 4.0, 3.0], [2.0, 1.0, 0.0]]
        )


class TestUniqueConsecutive(unittest.TestCase):
    """测试 unique_consecutive 功能 / Test unique_consecutive functionality."""

    def test_unique_consecutive_basic(self):
        """测试 unique_consecutive 基本功能 / Test basic unique_consecutive."""
        x = paddle.to_tensor([1, 1, 2, 2, 3])
        out = paddle.unique_consecutive(x)
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3])


class TestUnique(unittest.TestCase):
    """测试 unique 功能 / Test unique functionality."""

    def test_unique_basic(self):
        """测试 unique 基本功能 / Test basic unique."""
        x = paddle.to_tensor([1, 2, 3, 2, 1])
        out, indices, inverse, counts = paddle.unique(
            x, return_index=True, return_inverse=True, return_counts=True
        )
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3])

    def test_unique_sorted(self):
        """测试 unique 排序 / Test unique sorted."""
        x = paddle.to_tensor([3, 1, 2, 1])
        out = paddle.unique(x, sorted=True)
        np.testing.assert_array_equal(out.numpy(), [1, 2, 3])


class TestIndexAdd(unittest.TestCase):
    """测试 index_add 功能 / Test index_add functionality."""

    def test_index_add_basic(self):
        """测试 index_add 张量方法 / Test index_add tensor method."""
        x = paddle.zeros([5, 3], dtype='float32')
        value = paddle.ones([2, 3], dtype='float32')
        index = paddle.to_tensor([1, 3])
        out = x.index_add_(index, axis=0, value=value)
        self.assertEqual(out.shape, [5, 3])
        self.assertAlmostEqual(float(out[1, 0].numpy()), 1.0)

    def test_index_add_alias(self):
        """测试 index_add 张量方法 / Test index_add tensor method."""
        x = paddle.zeros([5, 3], dtype='float32')
        value = paddle.ones([2, 3], dtype='float32')
        index = paddle.to_tensor([0, 2])
        out = x.index_add_(index, axis=0, value=value)
        self.assertEqual(out.shape, [5, 3])


class TestUnflatten(unittest.TestCase):
    """测试 unflatten 功能 / Test unflatten functionality."""

    def test_unflatten_basic(self):
        """测试 unflatten 基本功能 / Test basic unflatten."""
        x = paddle.randn([4, 6])
        out = paddle.unflatten(x, axis=1, shape=[2, 3])
        self.assertEqual(out.shape, [4, 2, 3])


class TestViewAs(unittest.TestCase):
    """测试 view_as 功能 / Test view_as functionality."""

    def test_view_as_basic(self):
        """测试 view_as 基本功能 / Test basic view_as."""
        x = paddle.randn([4, 6])
        other = paddle.randn([2, 12])
        out = paddle.view_as(x, other)
        self.assertEqual(out.shape, [2, 12])


class TestUnfold(unittest.TestCase):
    """测试 unfold 功能 / Test unfold functionality."""

    def test_unfold_basic(self):
        """测试 unfold 基本功能 / Test basic unfold."""
        x = paddle.arange(9, dtype='float64').reshape([1, 9])
        out = paddle.unfold(x, axis=1, size=3, step=2)
        self.assertIsNotNone(out)


class TestIndexFill(unittest.TestCase):
    """测试 index_fill 功能 / Test index_fill functionality."""

    def test_index_fill_basic(self):
        """测试 index_fill 基本功能 / Test basic index_fill."""
        x = paddle.zeros([5], dtype='float32')
        index = paddle.to_tensor([1, 3])
        out = paddle.index_fill(x, axis=0, index=index, value=7.0)
        np.testing.assert_array_almost_equal(
            out.numpy(), [0.0, 7.0, 0.0, 7.0, 0.0]
        )


class TestBlockDiag(unittest.TestCase):
    """测试 block_diag 功能 / Test block_diag functionality."""

    def test_block_diag_basic(self):
        """测试 block_diag 基本功能 / Test basic block_diag."""
        a = paddle.to_tensor([[1, 2], [3, 4]], dtype='float32')
        b = paddle.to_tensor([[5]], dtype='float32')
        out = paddle.block_diag([a, b])
        self.assertEqual(out.shape, [3, 3])


if __name__ == '__main__':
    unittest.main()
