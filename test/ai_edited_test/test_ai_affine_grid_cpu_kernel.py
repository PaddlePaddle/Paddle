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

# [AUTO-GENERATED] Do not edit manually.
# Target source: paddle/phi/kernels/cpu/affine_grid_kernel.cc
# Generated for exercising C++ CPU kernel: AffineGridKernel, AffineGrid4DKernel, AffineGrid5DKernel
#
# 测试 Affine Grid 生成 CPU 内核
# Tests for Affine Grid generation CPU kernel

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestAffineGrid4DBasic(unittest.TestCase):
    """4D 仿射网格基本测试 / Basic 4D affine grid tests"""

    def setUp(self):
        """初始化测试环境 / Setup test environment"""
        paddle.disable_static()
        paddle.set_device("cpu")

    def tearDown(self):
        """清理测试环境 / Teardown test environment"""
        paddle.enable_static()

    def test_affine_grid_identity_transform(self):
        """测试单位变换生成正确的网格坐标
        Test identity transform generates correct grid coordinates
        """
        # Identity transform: theta = [[1, 0, 0], [0, 1, 0]]
        theta = paddle.zeros([1, 2, 3], dtype="float32")
        theta[0, 0, 0] = 1.0
        theta[0, 1, 1] = 1.0

        out_shape = [1, 1, 2, 2]
        grid = F.affine_grid(theta, out_shape, align_corners=True)

        # For identity transform with align_corners=True, H=2, W=2:
        # Grid should be [[-1, -1], [1, -1], [-1, 1], [1, 1]]
        self.assertEqual(grid.shape, [1, 2, 2, 2])
        expected = np.array(
            [[[-1, -1], [1, -1]], [[-1, 1], [1, 1]]], dtype="float32"
        )
        np.testing.assert_allclose(grid[0].numpy(), expected, atol=1e-5)

    def test_affine_grid_output_shape(self):
        """测试仿射网格的输出形状
        Test output shape of affine grid
        """
        theta = paddle.randn([2, 2, 3], dtype="float32")
        grid = F.affine_grid(theta, [2, 3, 4, 5], align_corners=True)
        self.assertEqual(grid.shape, [2, 4, 5, 2])

    def test_affine_grid_translation(self):
        """测试平移变换
        Test translation transform
        """
        # Translation: shift right by 0.5
        theta = paddle.zeros([1, 2, 3], dtype="float32")
        theta[0, 0, 0] = 1.0
        theta[0, 0, 2] = 0.5  # tx
        theta[0, 1, 1] = 1.0

        out_shape = [1, 1, 2, 2]
        grid = F.affine_grid(theta, out_shape, align_corners=True)

        self.assertEqual(grid.shape, [1, 2, 2, 2])
        # First coordinate column should be shifted by 0.5
        expected_x = np.array(
            [[-1 + 0.5, 1 + 0.5], [-1 + 0.5, 1 + 0.5]], dtype="float32"
        )
        np.testing.assert_allclose(
            grid[0, :, :, 0].numpy(), expected_x, atol=1e-5
        )
        # Second coordinate column should be identity
        expected_y = np.array([[-1, -1], [1, 1]], dtype="float32")
        np.testing.assert_allclose(
            grid[0, :, :, 1].numpy(), expected_y, atol=1e-5
        )

    def test_affine_grid_scale(self):
        """测试缩放变换
        Test scaling transform
        """
        # Scale by 0.5
        theta = paddle.zeros([1, 2, 3], dtype="float32")
        theta[0, 0, 0] = 0.5
        theta[0, 1, 1] = 0.5

        out_shape = [1, 1, 2, 2]
        grid = F.affine_grid(theta, out_shape, align_corners=True)

        self.assertEqual(grid.shape, [1, 2, 2, 2])
        # With scale 0.5, coordinates should be halved
        expected = np.array(
            [[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]],
            dtype="float32",
        )
        np.testing.assert_allclose(grid[0].numpy(), expected, atol=1e-5)

    def test_affine_grid_align_corners_vs_no_align(self):
        """测试 align_corners=True 和 align_corners=False 的区别
        Test difference between align_corners=True and align_corners=False
        """
        theta = paddle.zeros([1, 2, 3], dtype="float32")
        theta[0, 0, 0] = 1.0
        theta[0, 1, 1] = 1.0

        grid_ac = F.affine_grid(theta, [1, 1, 3, 3], align_corners=True)
        grid_noac = F.affine_grid(theta, [1, 1, 3, 3], align_corners=False)

        # align_corners=True: endpoints at -1 and 1
        self.assertAlmostEqual(grid_ac[0, 0, 0, 0].item(), -1.0, places=4)
        self.assertAlmostEqual(grid_ac[0, 0, 2, 0].item(), 1.0, places=4)

        # align_corners=False: endpoints offset by half pixel
        self.assertAlmostEqual(
            grid_noac[0, 0, 0, 0].item(), -1.0 + 1.0 / 3, places=4
        )
        self.assertAlmostEqual(
            grid_noac[0, 0, 2, 0].item(), 1.0 - 1.0 / 3, places=4
        )

    def test_affine_grid_batch(self):
        """测试批量仿射网格
        Test batched affine grid
        """
        theta = paddle.zeros([3, 2, 3], dtype="float32")
        for i in range(3):
            theta[i, 0, 0] = 1.0
            theta[i, 1, 1] = 1.0

        grid = F.affine_grid(theta, [3, 1, 4, 4], align_corners=True)
        self.assertEqual(grid.shape, [3, 4, 4, 2])

    def test_affine_grid_float64(self):
        """测试 float64 类型的仿射网格
        Test affine grid with float64
        """
        theta = paddle.zeros([1, 2, 3], dtype="float64")
        theta[0, 0, 0] = 1.0
        theta[0, 1, 1] = 1.0

        grid = F.affine_grid(theta, [1, 1, 2, 2], align_corners=True)
        self.assertEqual(grid.dtype, paddle.float64)
        self.assertEqual(grid.shape, [1, 2, 2, 2])

    def test_affine_grid_5d(self):
        """测试 5D 仿射网格（3D 变换）
        Test 5D affine grid (3D transform)
        """
        # Identity transform for 3D: theta = [[1,0,0,0],[0,1,0,0],[0,0,1,0]]
        theta = paddle.zeros([1, 3, 4], dtype="float32")
        theta[0, 0, 0] = 1.0
        theta[0, 1, 1] = 1.0
        theta[0, 2, 2] = 1.0

        out_shape = [1, 1, 2, 2, 2]
        grid = F.affine_grid(theta, out_shape, align_corners=True)

        self.assertEqual(grid.shape, [1, 2, 2, 2, 3])
        # Check corner values
        self.assertAlmostEqual(grid[0, 0, 0, 0, 0].item(), -1.0, places=4)
        self.assertAlmostEqual(grid[0, 0, 0, 0, 1].item(), -1.0, places=4)
        self.assertAlmostEqual(grid[0, 0, 0, 0, 2].item(), -1.0, places=4)
        self.assertAlmostEqual(grid[0, 1, 1, 1, 0].item(), 1.0, places=4)
        self.assertAlmostEqual(grid[0, 1, 1, 1, 1].item(), 1.0, places=4)
        self.assertAlmostEqual(grid[0, 1, 1, 1, 2].item(), 1.0, places=4)

    def test_affine_grid_5d_batch(self):
        """测试批量 5D 仿射网格
        Test batched 5D affine grid
        """
        theta = paddle.zeros([2, 3, 4], dtype="float32")
        for i in range(2):
            theta[i, 0, 0] = 1.0
            theta[i, 1, 1] = 1.0
            theta[i, 2, 2] = 1.0

        grid = F.affine_grid(theta, [2, 1, 3, 4, 5], align_corners=True)
        self.assertEqual(grid.shape, [2, 3, 4, 5, 3])

    def test_affine_grid_rotation(self):
        """测试旋转变换
        Test rotation transform (45 degrees)
        """
        angle = np.pi / 4  # 45 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        theta = paddle.zeros([1, 2, 3], dtype="float32")
        theta[0, 0, 0] = cos_a
        theta[0, 0, 1] = -sin_a
        theta[0, 1, 0] = sin_a
        theta[0, 1, 1] = cos_a

        out_shape = [1, 1, 2, 2]
        grid = F.affine_grid(theta, out_shape, align_corners=True)

        self.assertEqual(grid.shape, [1, 2, 2, 2])
        # Verify rotation applied: (-1,-1) should map to rotated coords
        result = grid[0].numpy()
        # (-1,-1) -> (cos(-1)-sin(-1), sin(-1)+cos(-1)) = (-cos+sin, -sin-cos)
        expected_00 = np.array(
            [cos_a * (-1) - sin_a * (-1), sin_a * (-1) + cos_a * (-1)]
        )
        np.testing.assert_allclose(result[0, 0], expected_00, atol=1e-5)

    def test_affine_grid_large_output(self):
        """测试大输出尺寸的仿射网格
        Test affine grid with large output size
        """
        theta = paddle.zeros([1, 2, 3], dtype="float32")
        theta[0, 0, 0] = 1.0
        theta[0, 1, 1] = 1.0

        grid = F.affine_grid(theta, [1, 1, 64, 64], align_corners=True)
        self.assertEqual(grid.shape, [1, 64, 64, 2])


if __name__ == "__main__":
    unittest.main()
