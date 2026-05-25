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
# 未覆盖行: 111,125,135,136,139,140,141,142,143,144,148,150,156,210,211,214,215,224,291,292,293,294,300
# Covered module: paddle/nn/functional/vision.py
# Uncovered lines: 111,125,135,136,139,140,141,142,143,144,148,150,156,210,211,214,215,224,291,292,293,294,300

import unittest

import paddle


class TestAffineGrid(unittest.TestCase):
    """测试 affine_grid 函数
    Test affine_grid function"""

    def test_affine_grid_2d(self):
        """测试2D affine_grid
        Test 2D affine_grid"""
        theta = paddle.to_tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype='float32'
        )
        out_shape = paddle.to_tensor([1, 1, 3, 3], dtype='int32')
        grid = paddle.nn.functional.affine_grid(theta, out_shape)
        self.assertEqual(grid.shape, [1, 3, 3, 2])

    def test_affine_grid_2d_list(self):
        """测试使用 list 作为 out_shape 的2D affine_grid
        Test 2D affine_grid with list out_shape"""
        theta = paddle.to_tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype='float32'
        )
        grid = paddle.nn.functional.affine_grid(theta, [1, 1, 3, 3])
        self.assertEqual(grid.shape, [1, 3, 3, 2])

    def test_affine_grid_align_corners(self):
        """测试 align_corners 参数的 affine_grid
        Test affine_grid with align_corners parameter"""
        theta = paddle.to_tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype='float32'
        )
        grid = paddle.nn.functional.affine_grid(
            theta, [1, 1, 3, 3], align_corners=False
        )
        self.assertEqual(grid.shape, [1, 3, 3, 2])

    def test_affine_grid_float64(self):
        """测试 float64 类型的 affine_grid
        Test affine_grid with float64 dtype"""
        theta = paddle.to_tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype='float64'
        )
        grid = paddle.nn.functional.affine_grid(theta, [1, 1, 3, 3])
        self.assertEqual(grid.dtype, paddle.float64)


class TestPixelShuffle(unittest.TestCase):
    """测试 pixel_shuffle 函数
    Test pixel_shuffle function"""

    def test_pixel_shuffle_basic(self):
        """测试基本的 pixel_shuffle
        Test basic pixel_shuffle"""
        x = paddle.randn([1, 4, 2, 2])
        result = paddle.nn.functional.pixel_shuffle(x, upscale_factor=2)
        self.assertEqual(result.shape, [1, 1, 4, 4])

    def test_pixel_shuffle_3d(self):
        """测试3D输入的 pixel_shuffle (需要4D输入)
        Test pixel_shuffle requires 4D input"""
        x = paddle.randn([1, 4, 2, 2])
        result = paddle.nn.functional.pixel_shuffle(x, upscale_factor=2)
        self.assertEqual(result.shape, [1, 1, 4, 4])

    def test_pixel_shuffle_larger_factor(self):
        """测试更大 upscale_factor 的 pixel_shuffle
        Test pixel_shuffle with larger upscale_factor"""
        x = paddle.randn([1, 9, 3, 3])
        result = paddle.nn.functional.pixel_shuffle(x, upscale_factor=3)
        self.assertEqual(result.shape, [1, 1, 9, 9])


class TestGridSample(unittest.TestCase):
    """测试 grid_sample 函数
    Test grid_sample function"""

    def test_grid_sample_basic(self):
        """测试基本的 grid_sample
        Test basic grid_sample"""
        x = paddle.randn([1, 1, 3, 3])
        grid = paddle.to_tensor(
            [[[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]]], dtype='float32'
        )
        result = paddle.nn.functional.grid_sample(x, grid)
        self.assertEqual(result.shape, [1, 1, 1, 3])

    def test_grid_sample_bilinear(self):
        """测试 bilinear 插值的 grid_sample
        Test grid_sample with bilinear interpolation"""
        x = paddle.randn([1, 1, 4, 4])
        grid = paddle.to_tensor(
            [
                [
                    [
                        [0.0, 0.0],
                        [0.5, 0.0],
                        [1.0, 0.0],
                        [-1.0, 0.0],
                    ]
                ]
            ],
            dtype='float32',
        )
        result = paddle.nn.functional.grid_sample(
            x, grid, mode='bilinear', padding_mode='zeros'
        )
        self.assertEqual(result.shape, [1, 1, 1, 4])

    def test_grid_sample_nearest(self):
        """测试 nearest 插值的 grid_sample
        Test grid_sample with nearest interpolation"""
        x = paddle.randn([1, 1, 3, 3])
        grid = paddle.to_tensor([[[[0.0, 0.0], [0.5, 0.5]]]], dtype='float32')
        result = paddle.nn.functional.grid_sample(x, grid, mode='nearest')
        self.assertEqual(result.shape, [1, 1, 1, 2])

    def test_grid_sample_align_corners(self):
        """测试 align_corners 参数的 grid_sample
        Test grid_sample with align_corners parameter"""
        x = paddle.randn([1, 1, 4, 4])
        grid = paddle.to_tensor([[[[0.0, 0.0], [1.0, 1.0]]]], dtype='float32')
        result = paddle.nn.functional.grid_sample(x, grid, align_corners=True)
        self.assertEqual(result.shape, [1, 1, 1, 2])


if __name__ == '__main__':
    unittest.main()
