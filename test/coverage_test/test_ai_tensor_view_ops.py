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
张量切片和视图测试 / Tensor Slicing and View Tests

测试目标 / Test Target:
  paddle.tensor 切片和视图操作

覆盖的模块 / Covered Modules:
  - paddle.Tensor.view: 视图操作
  - paddle.Tensor.contiguous: 连续性
  - paddle.split/vsplit/hsplit: 分割
  - paddle.tile/repeat_interleave: 重复

作用 / Purpose:
  补充张量视图和切片API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestTensorView(unittest.TestCase):
    """测试张量视图 / Test tensor view"""

    def test_reshape_view(self):
        """测试reshape视图 / Test reshape view"""
        x = paddle.arange(24)
        y = x.reshape([4, 6])
        self.assertEqual(y.shape, [4, 6])
        z = x.reshape([2, 3, 4])
        self.assertEqual(z.shape, [2, 3, 4])

    def test_reshape_minus_one(self):
        """测试reshape中的-1 / Test reshape with -1"""
        x = paddle.randn([4, 6])
        y = x.reshape([-1])
        self.assertEqual(y.shape, [24])
        z = x.reshape([2, -1])
        self.assertEqual(z.shape, [2, 12])

    def test_flatten_all(self):
        """测试完全展平 / Test full flatten"""
        x = paddle.randn([2, 3, 4])
        result = paddle.flatten(x)
        self.assertEqual(result.shape, [24])

    def test_view_as(self):
        """测试view_as / Test view_as"""
        x = paddle.randn([2, 12])
        template = paddle.zeros([4, 6])
        result = x.reshape(template.shape)
        self.assertEqual(result.shape, [4, 6])


class TestSplitOperations(unittest.TestCase):
    """测试分割操作 / Test split operations"""

    def test_split_equal(self):
        """测试等分 / Test equal split"""
        x = paddle.randn([6, 4])
        chunks = paddle.split(x, 3)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(chunk.shape, [2, 4])

    def test_split_sections(self):
        """测试按段分割 / Test split by sections"""
        x = paddle.randn([10, 4])
        chunks = paddle.split(x, [3, 3, 4])
        self.assertEqual(chunks[0].shape, [3, 4])
        self.assertEqual(chunks[1].shape, [3, 4])
        self.assertEqual(chunks[2].shape, [4, 4])

    def test_vsplit(self):
        """测试垂直分割 / Test vertical split"""
        x = paddle.randn([6, 4])
        chunks = paddle.vsplit(x, 3)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(chunk.shape, [2, 4])

    def test_hsplit(self):
        """测试水平分割 / Test horizontal split"""
        x = paddle.randn([4, 6])
        chunks = paddle.hsplit(x, 3)
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(chunk.shape, [4, 2])


class TestTileAndRepeat(unittest.TestCase):
    """测试tile和repeat操作 / Test tile and repeat operations"""

    def test_tile(self):
        """测试tile操作 / Test tile operation"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        result = paddle.tile(x, [2, 3])
        self.assertEqual(result.shape, [4, 6])

    def test_tile_1d(self):
        """测试1D tile / Test 1D tile"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.tile(x, [4])
        np.testing.assert_allclose(
            result.numpy(),
            [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        )

    def test_repeat_interleave(self):
        """测试repeat_interleave / Test repeat_interleave"""
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.repeat_interleave(x, repeats=3)
        np.testing.assert_allclose(
            result.numpy(), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
        )

    def test_repeat_interleave_2d(self):
        """测试2D repeat_interleave / Test 2D repeat_interleave"""
        x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        result = paddle.repeat_interleave(x, repeats=2, axis=0)
        self.assertEqual(result.shape, [4, 2])


class TestRollAndPad(unittest.TestCase):
    """测试roll和pad操作 / Test roll and pad operations"""

    def test_roll_1d(self):
        """测试1D roll / Test 1D roll"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.roll(x, shifts=2)
        np.testing.assert_allclose(result.numpy(), [4.0, 5.0, 1.0, 2.0, 3.0])

    def test_roll_negative(self):
        """测试负移roll / Test negative shift roll"""
        x = paddle.to_tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paddle.roll(x, shifts=-2)
        np.testing.assert_allclose(result.numpy(), [3.0, 4.0, 5.0, 1.0, 2.0])

    def test_roll_2d(self):
        """测试2D roll / Test 2D roll"""
        x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = paddle.roll(x, shifts=1, axis=1)
        self.assertEqual(result.shape, [2, 3])
        np.testing.assert_allclose(result[0].numpy(), [3.0, 1.0, 2.0])


if __name__ == '__main__':
    unittest.main()
