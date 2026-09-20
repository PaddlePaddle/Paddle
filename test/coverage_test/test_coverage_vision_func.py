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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.vision
# 自动生成的单测，覆盖 paddle.nn.functional.vision 模块中未覆盖的代码
# Target: cover uncovered lines in python/paddle/nn/functional/vision.py
# NOTE: test_ai_vision_ops.py already covers yolo_loss and yolo_box from paddle.vision.ops.
#       This test covers affine_grid, pixel_unshuffle, and channel_shuffle.

"""
测试模块：paddle.nn.functional.vision
Test Module: paddle.nn.functional.vision

本测试覆盖以下功能：
This test covers the following functions:
1. affine_grid - 仿射变换坐标网格生成 / Affine transformation coordinate grid generation
   - 2D 仿射网格 / 2D affine grid
   - align_corners=True/False / align_corners option
   - 3D 仿射网格 / 3D affine grid
   - 输出形状验证 / Output shape verification
2. pixel_unshuffle - 像素反混洗 / Pixel unshuffle operation
   - downscale_factor=2, 3 / Different downscale factors
   - NCHW 和 NHWC 格式 / NCHW and NHWC formats
   - 输出形状验证 / Output shape verification
3. channel_shuffle - 通道混洗 / Channel shuffle operation
   - 不同 groups 参数 / Different groups
   - NCHW 和 NHWC 格式 / NCHW and NHWC formats
   - 输出形状等于输入形状 / Output shape equals input shape
"""

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestAffineGrid2D(unittest.TestCase):
    """测试 2D affine_grid
    Test 2D affine_grid"""

    def setUp(self):
        """设置测试环境 / Set up test environment"""
        paddle.disable_static()

    def test_2d_affine_grid_batch1(self):
        """测试 batch=1 的 2D 仿射网格
        Test 2D affine grid with batch=1"""
        theta = paddle.to_tensor(
            [[[-0.7, -0.4, 0.3], [0.6, 0.5, 1.5]]],
            dtype="float32",
        )
        grid = F.affine_grid(theta, [1, 2, 3, 3])
        # 输出形状应为 [1, 3, 3, 2] / Output shape should be [1, 3, 3, 2]
        self.assertEqual(list(grid.shape), [1, 3, 3, 2])

    def test_2d_affine_grid_align_corners_false(self):
        """测试 align_corners=False 的 2D 仿射网格
        Test 2D affine grid with align_corners=False"""
        theta = paddle.to_tensor(
            [[[-0.7, -0.4, 0.3], [0.6, 0.5, 1.5]]],
            dtype="float32",
        )
        grid = F.affine_grid(theta, [1, 2, 3, 3], align_corners=False)
        self.assertEqual(list(grid.shape), [1, 3, 3, 2])
        # 验证网格值范围合理 / Verify grid values are in reasonable range
        grid_np = grid.numpy()
        self.assertTrue(np.all(np.isfinite(grid_np)))

    def test_2d_affine_grid_align_corners_true(self):
        """测试 align_corners=True 的 2D 仿射网格
        Test 2D affine grid with align_corners=True"""
        theta = paddle.to_tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype="float32",
        )
        # 单位变换应该生成标准坐标网格 / Identity should generate standard grid
        grid = F.affine_grid(theta, [1, 1, 4, 4], align_corners=True)
        self.assertEqual(list(grid.shape), [1, 4, 4, 2])
        # align_corners=True 时，角点应为 -1 和 1
        # With align_corners=True, corners should be -1 and 1
        grid_np = grid.numpy()
        np.testing.assert_allclose(grid_np[0, 0, 0, :], [-1.0, -1.0], atol=1e-5)
        np.testing.assert_allclose(grid_np[0, 3, 3, :], [1.0, 1.0], atol=1e-5)

    def test_2d_affine_grid_batch2(self):
        """测试 batch=2 的 2D 仿射网格
        Test 2D affine grid with batch=2"""
        theta = paddle.randn([2, 2, 3], dtype='float32')
        grid = F.affine_grid(theta, [2, 3, 5, 5])
        self.assertEqual(list(grid.shape), [2, 5, 5, 2])

    def test_2d_affine_grid_float64(self):
        """测试 float64 类型的 2D 仿射网格
        Test 2D affine grid with float64 dtype"""
        theta = paddle.to_tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype="float64",
        )
        grid = F.affine_grid(theta, [1, 1, 2, 2])
        self.assertEqual(grid.dtype, paddle.float64)
        self.assertEqual(list(grid.shape), [1, 2, 2, 2])

    def test_2d_affine_grid_identity_transform(self):
        """测试单位变换生成标准坐标网格
        Test identity transform generates standard coordinate grid"""
        theta = paddle.to_tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype="float32",
        )
        grid = F.affine_grid(theta, [1, 1, 2, 2], align_corners=True)
        grid_np = grid.numpy()
        # 四个角点 / Four corners
        expected = [[[-1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [1.0, 1.0]]]
        np.testing.assert_allclose(grid_np[0], expected, atol=1e-5)


class TestAffineGrid3D(unittest.TestCase):
    """测试 3D affine_grid (体积仿射变换)
    Test 3D affine_grid (volumetric affine transformation)"""

    def setUp(self):
        paddle.disable_static()

    def test_3d_affine_grid(self):
        """测试 3D 仿射网格 (batch, depth, height, width)
        Test 3D affine grid"""
        theta = paddle.randn([1, 3, 4], dtype='float32')
        grid = F.affine_grid(theta, [1, 2, 4, 4, 4])
        # 输出形状应为 [batch, D, H, W, 3] / Output shape should be [batch, D, H, W, 3]
        self.assertEqual(list(grid.shape), [1, 4, 4, 4, 3])

    def test_3d_affine_grid_identity(self):
        """测试 3D 单位变换
        Test 3D identity transform"""
        theta = paddle.to_tensor(
            [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]],
            dtype='float32',
        )
        grid = F.affine_grid(theta, [1, 1, 2, 2, 2], align_corners=True)
        self.assertEqual(list(grid.shape), [1, 2, 2, 2, 3])
        # 验证角点坐标 / Verify corner coordinates
        grid_np = grid.numpy()
        np.testing.assert_allclose(
            grid_np[0, 0, 0, 0, :], [-1.0, -1.0, -1.0], atol=1e-5
        )
        np.testing.assert_allclose(
            grid_np[0, 1, 1, 1, :], [1.0, 1.0, 1.0], atol=1e-5
        )

    def test_3d_affine_grid_different_sizes(self):
        """测试 3D 仿射网格不同尺寸
        Test 3D affine grid with different sizes"""
        theta = paddle.randn([2, 3, 4], dtype='float32')
        grid = F.affine_grid(theta, [2, 3, 8, 6, 4])
        self.assertEqual(list(grid.shape), [2, 8, 6, 4, 3])


class TestPixelUnshuffle(unittest.TestCase):
    """测试 pixel_unshuffle 操作
    Test pixel_unshuffle operation"""

    def setUp(self):
        paddle.disable_static()

    def test_downscale_factor_2_nchw(self):
        """测试 downscale_factor=2 在 NCHW 格式下
        Test downscale_factor=2 with NCHW format"""
        x = paddle.randn([2, 4, 6, 6], dtype='float32')
        out = F.pixel_unshuffle(x, 2, data_format='NCHW')
        # 输出形状应为 [N, C*r*r, H/r, W/r]
        # Output shape should be [N, C*r*r, H/r, W/r]
        self.assertEqual(list(out.shape), [2, 16, 3, 3])

    def test_downscale_factor_3_nchw(self):
        """测试 downscale_factor=3 在 NCHW 格式下
        Test downscale_factor=3 with NCHW format"""
        x = paddle.randn([1, 9, 6, 6], dtype='float32')
        out = F.pixel_unshuffle(x, 3, data_format='NCHW')
        # [1, 9*3*3, 6/3, 6/3] = [1, 81, 2, 2]
        self.assertEqual(list(out.shape), [1, 81, 2, 2])

    def test_downscale_factor_2_nhwc(self):
        """测试 downscale_factor=2 在 NHWC 格式下
        Test downscale_factor=2 with NHWC format"""
        x = paddle.randn([2, 6, 6, 4], dtype='float32')
        out = F.pixel_unshuffle(x, 2, data_format='NHWC')
        # NHWC: [N, H/r, W/r, C*r*r]
        self.assertEqual(list(out.shape), [2, 3, 3, 16])

    def test_downscale_factor_4_nchw(self):
        """测试 downscale_factor=4 在 NCHW 格式下
        Test downscale_factor=4 with NCHW format"""
        x = paddle.randn([1, 16, 8, 8], dtype='float32')
        out = F.pixel_unshuffle(x, 4, data_format='NCHW')
        self.assertEqual(list(out.shape), [1, 256, 2, 2])

    def test_pixel_unshuffle_float64(self):
        """测试 float64 类型的 pixel_unshuffle
        Test pixel_unshuffle with float64 dtype"""
        x = paddle.randn([1, 4, 4, 4], dtype='float64')
        out = F.pixel_unshuffle(x, 2, data_format='NCHW')
        self.assertEqual(out.dtype, paddle.float64)
        self.assertEqual(list(out.shape), [1, 16, 2, 2])

    def test_pixel_unshuffle_values_preserved(self):
        """测试 pixel_unshuffle 是否正确重新排列像素值
        Test that pixel_unshuffle correctly rearranges pixel values"""
        # 创建简单的输入来验证数据正确性
        # Create simple input to verify data correctness
        x = paddle.arange(0, 36, dtype='float32').reshape([1, 1, 6, 6])
        out = F.pixel_unshuffle(x, 2, data_format='NCHW')
        self.assertEqual(list(out.shape), [1, 4, 3, 3])
        # 所有原始值应保留（仅重新排列）
        # All original values should be preserved (just rearranged)
        x_flat = paddle.sort(x.flatten())
        out_flat = paddle.sort(out.flatten())
        np.testing.assert_array_equal(x_flat.numpy(), out_flat.numpy())


class TestPixelUnshuffleErrors(unittest.TestCase):
    """测试 pixel_unshuffle 的错误处理
    Test pixel_unshuffle error handling"""

    def setUp(self):
        paddle.disable_static()

    def test_invalid_downscale_factor_type(self):
        """测试非整数 downscale_factor 应报错
        Test that non-integer downscale_factor raises error"""
        x = paddle.randn([1, 4, 4, 4], dtype='float32')
        with self.assertRaises(TypeError):
            F.pixel_unshuffle(x, 2.0, data_format='NCHW')

    def test_zero_downscale_factor(self):
        """测试 downscale_factor=0 应报错
        Test that downscale_factor=0 raises error"""
        x = paddle.randn([1, 4, 4, 4], dtype='float32')
        with self.assertRaises(ValueError):
            F.pixel_unshuffle(x, 0, data_format='NCHW')

    def test_negative_downscale_factor(self):
        """测试负数 downscale_factor 应报错
        Test that negative downscale_factor raises error"""
        x = paddle.randn([1, 4, 4, 4], dtype='float32')
        with self.assertRaises(ValueError):
            F.pixel_unshuffle(x, -2, data_format='NCHW')

    def test_invalid_data_format(self):
        """测试无效的 data_format 应报错
        Test that invalid data_format raises error"""
        x = paddle.randn([1, 4, 4, 4], dtype='float32')
        with self.assertRaises(ValueError):
            F.pixel_unshuffle(x, 2, data_format='NCHWD')

    def test_3d_input_raises_error(self):
        """测试 3D 输入应报错
        Test that 3D input raises error"""
        x = paddle.randn([1, 4, 4], dtype='float32')
        with self.assertRaises(ValueError):
            F.pixel_unshuffle(x, 2, data_format='NCHW')


class TestChannelShuffle(unittest.TestCase):
    """测试 channel_shuffle 操作
    Test channel_shuffle operation"""

    def setUp(self):
        paddle.disable_static()

    def test_groups_2_4channel_nchw(self):
        """测试 groups=2 在 4 通道 NCHW 输入
        Test groups=2 on 4-channel NCHW input"""
        x = paddle.arange(0, 4, dtype='float32').reshape([1, 4, 1, 1])
        out = F.channel_shuffle(x, 2, data_format='NCHW')
        # 输出形状应与输入相同 / Output shape should equal input shape
        self.assertEqual(list(out.shape), [1, 4, 1, 1])
        # 验证值被正确重新排列
        # Verify values are correctly rearranged
        # groups=2: 原始 [0,1,2,3] -> 分组 [[0,1],[2,3]] -> 交错 [0,2,1,3]
        expected = paddle.to_tensor([0, 2, 1, 3], dtype='float32').reshape(
            [1, 4, 1, 1]
        )
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_groups_4_8channel_nchw(self):
        """测试 groups=4 在 8 通道 NCHW 输入
        Test groups=4 on 8-channel NCHW input"""
        x = paddle.arange(0, 8, dtype='float32').reshape([1, 8, 1, 1])
        out = F.channel_shuffle(x, 4, data_format='NCHW')
        self.assertEqual(list(out.shape), [1, 8, 1, 1])
        # groups=4: 分组 [[0,1],[2,3],[4,5],[6,7]] -> 交错 [0,2,4,6,1,3,5,7]
        expected = paddle.to_tensor(
            [0, 2, 4, 6, 1, 3, 5, 7], dtype='float32'
        ).reshape([1, 8, 1, 1])
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_groups_1(self):
        """测试 groups=1 (无混洗)
        Test groups=1 (no shuffle)"""
        x = paddle.arange(0, 6, dtype='float32').reshape([1, 6, 1, 1])
        out = F.channel_shuffle(x, 1, data_format='NCHW')
        # groups=1: 无变化 / groups=1: no change
        np.testing.assert_array_equal(out.numpy(), x.numpy())

    def test_groups_2_nhwc(self):
        """测试 groups=2 在 NHWC 格式下
        Test groups=2 in NHWC format"""
        x = paddle.arange(0, 4, dtype='float32').reshape([1, 1, 1, 4])
        out = F.channel_shuffle(x, 2, data_format='NHWC')
        self.assertEqual(list(out.shape), [1, 1, 1, 4])
        # groups=2 on NHWC with channels [0,1,2,3] -> [0,2,1,3]
        expected = paddle.to_tensor([0, 2, 1, 3], dtype='float32').reshape(
            [1, 1, 1, 4]
        )
        np.testing.assert_array_equal(out.numpy(), expected.numpy())

    def test_groups_2_spatial_input(self):
        """测试 groups=2 在有空间维度的输入上
        Test groups=2 on input with spatial dimensions"""
        x = paddle.arange(0, 24, dtype='float32').reshape([1, 4, 2, 3])
        out = F.channel_shuffle(x, 2, data_format='NCHW')
        self.assertEqual(list(out.shape), [1, 4, 2, 3])
        # 所有原始值应保留 / All original values should be preserved
        np.testing.assert_array_equal(
            np.sort(x.flatten().numpy()),
            np.sort(out.flatten().numpy()),
        )

    def test_channel_shuffle_float64(self):
        """测试 float64 类型的 channel_shuffle
        Test channel_shuffle with float64 dtype"""
        x = paddle.arange(0, 8, dtype='float64').reshape([1, 8, 1, 1])
        out = F.channel_shuffle(x, 4, data_format='NCHW')
        self.assertEqual(out.dtype, paddle.float64)
        self.assertEqual(list(out.shape), [1, 8, 1, 1])


class TestChannelShuffleErrors(unittest.TestCase):
    """测试 channel_shuffle 的错误处理
    Test channel_shuffle error handling"""

    def setUp(self):
        paddle.disable_static()

    def test_invalid_groups_type(self):
        """测试非整数 groups 应报错
        Test that non-integer groups raises error"""
        x = paddle.randn([1, 4, 2, 2], dtype='float32')
        with self.assertRaises(TypeError):
            F.channel_shuffle(x, 2.0, data_format='NCHW')

    def test_zero_groups(self):
        """测试 groups=0 应报错
        Test that groups=0 raises error"""
        x = paddle.randn([1, 4, 2, 2], dtype='float32')
        with self.assertRaises(ValueError):
            F.channel_shuffle(x, 0, data_format='NCHW')

    def test_negative_groups(self):
        """测试负数 groups 应报错
        Test that negative groups raises error"""
        x = paddle.randn([1, 4, 2, 2], dtype='float32')
        with self.assertRaises(ValueError):
            F.channel_shuffle(x, -1, data_format='NCHW')

    def test_invalid_data_format(self):
        """测试无效的 data_format 应报错
        Test that invalid data_format raises error"""
        x = paddle.randn([1, 4, 2, 2], dtype='float32')
        with self.assertRaises(ValueError):
            F.channel_shuffle(x, 2, data_format='invalid')

    def test_3d_input_raises_error(self):
        """测试 3D 输入应报错
        Test that 3D input raises error"""
        x = paddle.randn([1, 4, 2], dtype='float32')
        with self.assertRaises(ValueError):
            F.channel_shuffle(x, 2, data_format='NCHW')


class TestAffineGridErrors(unittest.TestCase):
    """测试 affine_grid 的错误处理
    Test affine_grid error handling"""

    def setUp(self):
        paddle.disable_static()

    def test_non_tensor_theta(self):
        """测试非 Tensor 的 theta 应报错
        Test that non-Tensor theta raises error"""
        with self.assertRaises(TypeError):
            F.affine_grid([[1, 0, 0], [0, 1, 0]], [1, 1, 2, 2])


if __name__ == '__main__':
    unittest.main()
