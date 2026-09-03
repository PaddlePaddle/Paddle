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
图像插值和几何变换测试 / Image Interpolation and Geometric Transform Tests

测试目标 / Test Target:
  paddle.nn.functional 图像操作

覆盖的模块 / Covered Modules:
  - F.interpolate: 插值操作
  - F.affine_grid: 仿射网格
  - F.grid_sample: 网格采样
  - F.pixel_shuffle/unshuffle: 像素重排

作用 / Purpose:
  补充图像几何变换API的测试，提升覆盖率。
"""

import unittest

import paddle
import paddle.nn.functional as F

paddle.disable_static()


class TestInterpolate(unittest.TestCase):
    """测试插值操作 / Test interpolation"""

    def test_bilinear_interpolate(self):
        """测试双线性插值 / Test bilinear interpolation"""
        x = paddle.randn([2, 3, 8, 8])
        result = F.interpolate(
            x, size=(16, 16), mode='bilinear', align_corners=True
        )
        self.assertEqual(result.shape, [2, 3, 16, 16])

    def test_nearest_interpolate(self):
        """测试最近邻插值 / Test nearest neighbor interpolation"""
        x = paddle.randn([2, 3, 8, 8])
        result = F.interpolate(x, size=(16, 16), mode='nearest')
        self.assertEqual(result.shape, [2, 3, 16, 16])

    def test_bicubic_interpolate(self):
        """测试双三次插值 / Test bicubic interpolation"""
        x = paddle.randn([2, 3, 8, 8])
        result = F.interpolate(
            x, size=(16, 16), mode='bicubic', align_corners=True
        )
        self.assertEqual(result.shape, [2, 3, 16, 16])

    def test_scale_factor_interpolate(self):
        """测试缩放因子插值 / Test interpolation with scale factor"""
        x = paddle.randn([2, 3, 8, 8])
        result = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=False
        )
        self.assertEqual(result.shape, [2, 3, 16, 16])

    def test_downscale_interpolate(self):
        """测试下采样插值 / Test downscale interpolation"""
        x = paddle.randn([2, 3, 16, 16])
        result = F.interpolate(
            x, size=(8, 8), mode='bilinear', align_corners=False
        )
        self.assertEqual(result.shape, [2, 3, 8, 8])

    def test_1d_interpolate(self):
        """测试1D插值 / Test 1D interpolation"""
        x = paddle.randn([2, 3, 8])
        result = F.interpolate(x, size=[16], mode='linear', align_corners=True)
        self.assertEqual(result.shape, [2, 3, 16])

    def test_3d_interpolate(self):
        """测试3D插值 / Test 3D interpolation"""
        x = paddle.randn([2, 3, 4, 4, 4])
        result = F.interpolate(
            x, size=(8, 8, 8), mode='trilinear', align_corners=True
        )
        self.assertEqual(result.shape, [2, 3, 8, 8, 8])


class TestAffineGrid(unittest.TestCase):
    """测试仿射网格 / Test affine grid"""

    def test_affine_grid_basic(self):
        """测试基本仿射网格 / Test basic affine grid"""
        # Identity transform
        theta = paddle.eye(2, 3).unsqueeze(0).expand([2, 2, 3])
        size = [2, 3, 8, 8]
        grid = F.affine_grid(theta, size)
        self.assertEqual(grid.shape, [2, 8, 8, 2])

    def test_grid_sample(self):
        """测试网格采样 / Test grid sample"""
        x = paddle.randn([2, 3, 8, 8])
        # Identity grid
        theta = paddle.eye(2, 3).unsqueeze(0).expand([2, 2, 3])
        grid = F.affine_grid(theta, x.shape)
        result = F.grid_sample(x, grid, mode='bilinear', align_corners=True)
        self.assertEqual(result.shape, [2, 3, 8, 8])


class TestPixelShuffle(unittest.TestCase):
    """测试像素重排 / Test pixel shuffle"""

    def test_pixel_shuffle(self):
        """测试像素重排 / Test pixel shuffle (sub-pixel convolution)"""
        # upscale_factor=2: C*r^2 -> C, H -> H*r, W -> W*r
        upscale_factor = 2
        x = paddle.randn([2, 4 * upscale_factor**2, 8, 8])
        result = F.pixel_shuffle(x, upscale_factor=upscale_factor)
        self.assertEqual(result.shape, [2, 4, 16, 16])

    def test_pixel_unshuffle(self):
        """测试像素反重排 / Test pixel unshuffle"""
        # Inverse of pixel_shuffle
        downscale_factor = 2
        x = paddle.randn([2, 4, 16, 16])
        result = F.pixel_unshuffle(x, downscale_factor=downscale_factor)
        self.assertEqual(result.shape, [2, 4 * downscale_factor**2, 8, 8])


if __name__ == '__main__':
    unittest.main()
