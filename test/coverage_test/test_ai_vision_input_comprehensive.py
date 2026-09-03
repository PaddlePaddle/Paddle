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

# [AUTO-GENERATED] Test file for paddle.nn.functional.vision
# 覆盖模块: paddle/nn/functional/vision.py
# Uncovered lines: affine_grid, grid_sample, pixel_shuffle, pixel_unshuffle

import unittest

import numpy as np

import paddle


class TestAffineGrid(unittest.TestCase):
    """测试 affine_grid 函数
    Test affine_grid function"""

    def test_affine_grid_identity(self):
        """测试恒等变换的 affine_grid
        Test identity affine_grid"""
        theta = paddle.to_tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        grid = paddle.nn.functional.affine_grid(theta, [1, 1, 3, 3])
        self.assertEqual(grid.shape, [1, 3, 3, 2])

    def test_affine_grid_shape(self):
        """测试 affine_grid 输出形状
        Test affine_grid output shape"""
        # Batch size comes from theta, not from output_shape
        theta = paddle.to_tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ]
        )
        grid = paddle.nn.functional.affine_grid(theta, [2, 1, 4, 4])
        self.assertEqual(grid.shape, [2, 4, 4, 2])

    def test_affine_grid_values(self):
        """测试 affine_grid 输出值范围
        Test affine_grid output value range"""
        theta = paddle.to_tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        grid = paddle.nn.functional.affine_grid(theta, [1, 1, 2, 2])
        # Grid values should be in [-1, 1]
        self.assertTrue(paddle.all(grid >= -1.0).item())
        self.assertTrue(paddle.all(grid <= 1.0).item())


class TestGridSample(unittest.TestCase):
    """测试 grid_sample 函数
    Test grid_sample function"""

    def test_grid_sample_basic(self):
        """测试基本 grid_sample
        Test basic grid_sample"""
        x = paddle.randn([1, 1, 4, 4])
        theta = paddle.to_tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        grid = paddle.nn.functional.affine_grid(theta, [1, 1, 4, 4])
        result = paddle.nn.functional.grid_sample(x, grid)
        self.assertEqual(result.shape, [1, 1, 4, 4])

    def test_grid_sample_bilinear(self):
        """测试双线性插值的 grid_sample
        Test grid_sample with bilinear interpolation"""
        x = paddle.randn([1, 3, 8, 8])
        theta = paddle.to_tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        grid = paddle.nn.functional.affine_grid(theta, [1, 3, 4, 4])
        result = paddle.nn.functional.grid_sample(x, grid, mode='bilinear')
        self.assertEqual(result.shape, [1, 3, 4, 4])

    def test_grid_sample_nearest(self):
        """测试最近邻插值的 grid_sample
        Test grid_sample with nearest interpolation"""
        x = paddle.randn([1, 1, 4, 4])
        theta = paddle.to_tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        grid = paddle.nn.functional.affine_grid(theta, [1, 1, 4, 4])
        result = paddle.nn.functional.grid_sample(x, grid, mode='nearest')
        self.assertEqual(result.shape, [1, 1, 4, 4])

    def test_grid_sample_padding_zeros(self):
        """测试零填充的 grid_sample
        Test grid_sample with zeros padding"""
        x = paddle.randn([1, 1, 4, 4])
        theta = paddle.to_tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        grid = paddle.nn.functional.affine_grid(theta, [1, 1, 4, 4])
        result = paddle.nn.functional.grid_sample(x, grid, padding_mode='zeros')
        self.assertEqual(result.shape, [1, 1, 4, 4])


class TestPixelShuffle(unittest.TestCase):
    """测试 pixel_shuffle 函数
    Test pixel_shuffle function"""

    def test_pixel_shuffle_basic(self):
        """测试基本 pixel_shuffle
        Test basic pixel_shuffle"""
        x = paddle.randn([1, 9, 2, 2])
        result = paddle.nn.functional.pixel_shuffle(x, upscale_factor=3)
        self.assertEqual(result.shape, [1, 1, 6, 6])

    def test_pixel_shuffle_factor2(self):
        """测试 upscale_factor=2 的 pixel_shuffle
        Test pixel_shuffle with upscale_factor=2"""
        x = paddle.randn([1, 4, 3, 3])
        result = paddle.nn.functional.pixel_shuffle(x, upscale_factor=2)
        self.assertEqual(result.shape, [1, 1, 6, 6])

    def test_pixel_shuffle_batch(self):
        """测试批量 pixel_shuffle
        Test batched pixel_shuffle"""
        x = paddle.randn([2, 16, 4, 4])
        result = paddle.nn.functional.pixel_shuffle(x, upscale_factor=4)
        self.assertEqual(result.shape, [2, 1, 16, 16])


class TestPixelUnshuffle(unittest.TestCase):
    """测试 pixel_unshuffle 函数
    Test pixel_unshuffle function"""

    def test_pixel_unshuffle_basic(self):
        """测试基本 pixel_unshuffle
        Test basic pixel_unshuffle"""
        x = paddle.randn([1, 1, 6, 6])
        result = paddle.nn.functional.pixel_unshuffle(x, downscale_factor=3)
        self.assertEqual(result.shape, [1, 9, 2, 2])

    def test_pixel_unshuffle_roundtrip(self):
        """测试 pixel_shuffle 和 pixel_unshuffle 往返
        Test pixel_shuffle and pixel_unshuffle roundtrip"""
        x = paddle.randn([1, 4, 3, 3])
        shuffled = paddle.nn.functional.pixel_shuffle(x, upscale_factor=2)
        unshuffled = paddle.nn.functional.pixel_unshuffle(
            shuffled, downscale_factor=2
        )
        np.testing.assert_allclose(unshuffled.numpy(), x.numpy(), atol=1e-6)


class TestOneHot(unittest.TestCase):
    """测试 one_hot 函数
    Test one_hot function"""

    def test_one_hot_basic(self):
        """测试基本 one_hot
        Test basic one_hot"""
        x = paddle.to_tensor([0, 1, 2, 3])
        result = paddle.nn.functional.one_hot(x, num_classes=4)
        self.assertEqual(result.shape, [4, 4])
        # Check that diagonal is 1
        for i in range(4):
            self.assertEqual(result[i, i].item(), 1)

    def test_one_hot_2d(self):
        """测试二维输入 one_hot
        Test 2D input one_hot"""
        x = paddle.to_tensor([[0, 1], [2, 3]])
        result = paddle.nn.functional.one_hot(x, num_classes=4)
        self.assertEqual(result.shape, [2, 2, 4])


class TestEmbedding(unittest.TestCase):
    """测试 embedding 函数
    Test embedding function"""

    def test_embedding_basic(self):
        """测试基本 embedding
        Test basic embedding"""
        x = paddle.to_tensor([0, 1, 2, 3])
        weight = paddle.randn([4, 5])
        result = paddle.nn.functional.embedding(x, weight)
        self.assertEqual(result.shape, [4, 5])

    def test_embedding_2d_input(self):
        """测试二维输入 embedding
        Test 2D input embedding"""
        x = paddle.to_tensor([[0, 1], [2, 3]])
        weight = paddle.randn([4, 5])
        result = paddle.nn.functional.embedding(x, weight)
        self.assertEqual(result.shape, [2, 2, 5])

    def test_embedding_padding_idx(self):
        """测试带 padding_idx 的 embedding
        Test embedding with padding_idx"""
        x = paddle.to_tensor([0, 1, 2])
        weight = paddle.randn([4, 5])
        result = paddle.nn.functional.embedding(x, weight, padding_idx=0)
        self.assertEqual(result.shape, [3, 5])


if __name__ == '__main__':
    unittest.main()
