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

# [AUTO-GENERATED] test for paddle/nn/functional/vision.py
# Target file: python/paddle/nn/functional/vision.py
# Coverage: 70.1% (54/77) - Uncovered lines: 111,125,135-156,210-224,291-300
# 本文件为 nn/functional/vision.py 的单元测试 / Unit tests for nn/functional/vision.py
#
# 测试目标：
# - affine_grid 仿射网格生成（含错误处理）
# - pixel_unshuffle 像素反混洗（含错误处理）
# - channel_shuffle 通道混洗（含错误处理）
# - grid_sample 网格采样

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestAffineGrid(unittest.TestCase):
    """affine_grid 测试 / affine_grid tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_affine_grid_2d(self):
        """测试 2D 仿射网格 / Test 2D affine grid"""
        theta = paddle.to_tensor(
            [[[-0.7, -0.4, 0.3], [0.6, 0.5, 1.5]]], dtype="float32"
        )
        out = F.affine_grid(theta, [1, 2, 3, 3], align_corners=False)
        self.assertEqual(out.shape, [1, 3, 3, 2])

    def test_affine_grid_2d_align_corners(self):
        """测试 2D 仿射网格 align_corners=True / Test 2D affine grid align_corners=True"""
        theta = paddle.to_tensor([[[1, 0, 0], [0, 1, 0]]], dtype="float32")
        out = F.affine_grid(theta, [1, 1, 3, 3], align_corners=True)
        self.assertEqual(out.shape, [1, 3, 3, 2])

    def test_affine_grid_2d_float64(self):
        """测试 float64 仿射网格 / Test float64 affine grid"""
        theta = paddle.to_tensor([[[1, 0, 0], [0, 1, 0]]], dtype="float64")
        out = F.affine_grid(theta, [1, 1, 2, 2], align_corners=False)
        self.assertEqual(out.shape, [1, 2, 2, 2])
        self.assertEqual(out.dtype, paddle.float64)

    def test_affine_grid_not_tensor_raises(self):
        """测试非张量输入报错 / Test non-tensor input raises TypeError"""
        with self.assertRaises(TypeError):
            F.affine_grid("not_tensor", [1, 2, 3, 3])

    def test_affine_grid_empty_shape_raises(self):
        """测试空输出形状报错 / Test empty out_shape raises ValueError"""
        theta = paddle.to_tensor([[[1, 0, 0], [0, 1, 0]]], dtype="float32")
        empty_shape = paddle.to_tensor([], dtype='int32')
        with self.assertRaises(ValueError):
            F.affine_grid(theta, empty_shape, align_corners=False)

    def test_affine_grid_3d(self):
        """测试 3D 仿射网格 / Test 3D affine grid"""
        theta = paddle.randn([1, 3, 4], dtype='float32')
        out = F.affine_grid(theta, [1, 2, 3, 3, 3], align_corners=False)
        self.assertEqual(out.shape, [1, 3, 3, 3, 3])

    def test_affine_grid_tensor_out_shape(self):
        """测试张量输出形状 / Test tensor out_shape"""
        theta = paddle.to_tensor([[[1, 0, 0], [0, 1, 0]]], dtype="float32")
        out_shape = paddle.to_tensor([1, 1, 3, 3], dtype='int32')
        out = F.affine_grid(theta, out_shape, align_corners=False)
        self.assertEqual(out.shape, [1, 3, 3, 2])


class TestPixelUnshuffle(unittest.TestCase):
    """pixel_unshuffle 测试 / pixel_unshuffle tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_pixel_unshuffle_basic(self):
        """测试基本 pixel_unshuffle / Test basic pixel_unshuffle"""
        x = paddle.randn([2, 1, 12, 12], dtype='float32')
        out = F.pixel_unshuffle(x, 3)
        self.assertEqual(out.shape, [2, 9, 4, 4])

    def test_pixel_unshuffle_downscale_2(self):
        """测试 downscale_factor=2 / Test downscale_factor=2"""
        x = paddle.randn([1, 4, 8, 8], dtype='float32')
        out = F.pixel_unshuffle(x, 2)
        self.assertEqual(out.shape, [1, 16, 4, 4])

    def test_pixel_unshuffle_nhwc(self):
        """测试 NHWC 格式 / Test NHWC format"""
        x = paddle.randn([2, 12, 12, 1], dtype='float32')
        out = F.pixel_unshuffle(x, 3, data_format='NHWC')
        self.assertEqual(out.shape, [2, 4, 4, 9])

    def test_pixel_unshuffle_not_4d_raises(self):
        """测试非 4D 输入报错 / Test non-4D input raises ValueError"""
        x = paddle.randn([2, 12, 12], dtype='float32')
        with self.assertRaises(ValueError):
            F.pixel_unshuffle(x, 3)

    def test_pixel_unshuffle_non_int_factor_raises(self):
        """测试非整数 downscale_factor 报错 / Test non-int downscale_factor raises TypeError"""
        x = paddle.randn([2, 1, 12, 12], dtype='float32')
        with self.assertRaises(TypeError):
            F.pixel_unshuffle(x, 2.5)

    def test_pixel_unshuffle_negative_factor_raises(self):
        """测试负 downscale_factor 报错 / Test negative downscale_factor raises ValueError"""
        x = paddle.randn([2, 1, 12, 12], dtype='float32')
        with self.assertRaises(ValueError):
            F.pixel_unshuffle(x, -2)

    def test_pixel_unshuffle_zero_factor_raises(self):
        """测试零 downscale_factor 报错 / Test zero downscale_factor raises ValueError"""
        x = paddle.randn([2, 1, 12, 12], dtype='float32')
        with self.assertRaises(ValueError):
            F.pixel_unshuffle(x, 0)

    def test_pixel_unshuffle_invalid_format_raises(self):
        """测试无效 data_format 报错 / Test invalid data_format raises ValueError"""
        x = paddle.randn([2, 1, 12, 12], dtype='float32')
        with self.assertRaises(ValueError):
            F.pixel_unshuffle(x, 3, data_format='invalid')

    def test_pixel_unshuffle_input_alias(self):
        """测试 input 别名参数 / Test input alias parameter"""
        x = paddle.randn([2, 1, 12, 12], dtype='float32')
        out = F.pixel_unshuffle(input=x, downscale_factor=3)
        self.assertEqual(out.shape, [2, 9, 4, 4])

    def test_pixel_unshuffle_float64(self):
        """测试 float64 输入 / Test float64 input"""
        x = paddle.randn([1, 4, 8, 8], dtype='float64')
        out = F.pixel_unshuffle(x, 2)
        self.assertEqual(out.shape, [1, 16, 4, 4])
        self.assertEqual(out.dtype, paddle.float64)


class TestChannelShuffle(unittest.TestCase):
    """channel_shuffle 测试 / channel_shuffle tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_channel_shuffle_basic(self):
        """测试基本 channel_shuffle / Test basic channel_shuffle"""
        x = paddle.arange(0, 0.6, 0.1, 'float32')
        x = paddle.reshape(x, [1, 6, 1, 1])
        y = F.channel_shuffle(x, 3)
        self.assertEqual(y.shape, [1, 6, 1, 1])
        # 验证所有元素仍然存在 / Verify all elements still present
        sorted_x = sorted(x.flatten().numpy().tolist())
        sorted_y = sorted(y.flatten().numpy().tolist())
        np.testing.assert_allclose(sorted_x, sorted_y, rtol=1e-5)

    def test_channel_shuffle_groups_2(self):
        """测试 groups=2 / Test groups=2"""
        x = paddle.randn([2, 4, 8, 8], dtype='float32')
        y = F.channel_shuffle(x, 2)
        self.assertEqual(y.shape, [2, 4, 8, 8])

    def test_channel_shuffle_groups_4(self):
        """测试 groups=4 / Test groups=4"""
        x = paddle.randn([1, 8, 4, 4], dtype='float32')
        y = F.channel_shuffle(x, 4)
        self.assertEqual(y.shape, [1, 8, 4, 4])

    def test_channel_shuffle_groups_1(self):
        """测试 groups=1（无变化）/ Test groups=1 (no change)"""
        x = paddle.randn([1, 4, 4, 4], dtype='float32')
        y = F.channel_shuffle(x, 1)
        np.testing.assert_array_equal(x.numpy(), y.numpy())

    def test_channel_shuffle_nhwc(self):
        """测试 NHWC 格式 / Test NHWC format"""
        x = paddle.randn([1, 4, 4, 6], dtype='float32')
        y = F.channel_shuffle(x, 3, data_format='NHWC')
        self.assertEqual(y.shape, [1, 4, 4, 6])

    def test_channel_shuffle_not_4d_raises(self):
        """测试非 4D 输入报错 / Test non-4D input raises ValueError"""
        x = paddle.randn([2, 6, 6], dtype='float32')
        with self.assertRaises(ValueError):
            F.channel_shuffle(x, 3)

    def test_channel_shuffle_non_int_groups_raises(self):
        """测试非整数 groups 报错 / Test non-int groups raises TypeError"""
        x = paddle.randn([1, 6, 1, 1], dtype='float32')
        with self.assertRaises(TypeError):
            F.channel_shuffle(x, 2.5)

    def test_channel_shuffle_negative_groups_raises(self):
        """测试负 groups 报错 / Test negative groups raises ValueError"""
        x = paddle.randn([1, 6, 1, 1], dtype='float32')
        with self.assertRaises(ValueError):
            F.channel_shuffle(x, -1)

    def test_channel_shuffle_zero_groups_raises(self):
        """测试零 groups 报错 / Test zero groups raises ValueError"""
        x = paddle.randn([1, 6, 1, 1], dtype='float32')
        with self.assertRaises(ValueError):
            F.channel_shuffle(x, 0)

    def test_channel_shuffle_invalid_format_raises(self):
        """测试无效 data_format 报错 / Test invalid data_format raises ValueError"""
        x = paddle.randn([1, 6, 1, 1], dtype='float32')
        with self.assertRaises(ValueError):
            F.channel_shuffle(x, 3, data_format='NCHWD')

    def test_channel_shuffle_preserves_values(self):
        """测试 channel_shuffle 保持值不变（仅重排）/ Test channel_shuffle preserves values (just reorders)"""
        x = paddle.arange(0, 12, 1, 'float32')
        x = paddle.reshape(x, [1, 12, 1, 1])
        y = F.channel_shuffle(x, 4)
        # 所有元素应保持不变，仅重新排序 / All elements should be preserved, just reordered
        sorted_x = sorted(x.flatten().numpy().tolist())
        sorted_y = sorted(y.flatten().numpy().tolist())
        self.assertEqual(sorted_x, sorted_y)


class TestGridSample(unittest.TestCase):
    """grid_sample 测试 / grid_sample tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_grid_sample_bilinear(self):
        """测试双线性插值 grid_sample / Test bilinear grid_sample"""
        x = paddle.randn([1, 1, 3, 3], dtype='float32')
        grid = paddle.randn([1, 3, 3, 2], dtype='float32')
        out = F.grid_sample(x, grid, mode='bilinear')
        self.assertEqual(out.shape, [1, 1, 3, 3])

    def test_grid_sample_nearest(self):
        """测试最近邻 grid_sample / Test nearest grid_sample"""
        x = paddle.randn([1, 1, 3, 3], dtype='float32')
        grid = paddle.randn([1, 3, 3, 2], dtype='float32')
        out = F.grid_sample(x, grid, mode='nearest')
        self.assertEqual(out.shape, [1, 1, 3, 3])

    def test_grid_sample_align_corners(self):
        """测试 align_corners grid_sample / Test align_corners grid_sample"""
        x = paddle.randn([1, 1, 3, 3], dtype='float32')
        grid = paddle.randn([1, 2, 2, 2], dtype='float32')
        out = F.grid_sample(x, grid, mode='bilinear', align_corners=True)
        self.assertEqual(out.shape, [1, 1, 2, 2])

    def test_grid_sample_3d(self):
        """测试 3D 输入 grid_sample / Test 3D input grid_sample"""
        x = paddle.randn([1, 1, 3, 3, 3], dtype='float32')
        grid = paddle.randn([1, 2, 2, 2, 3], dtype='float32')
        out = F.grid_sample(x, grid, mode='bilinear')
        self.assertEqual(out.shape, [1, 1, 2, 2, 2])

    def test_grid_sample_padding_mode(self):
        """测试不同 padding_mode / Test different padding_mode"""
        x = paddle.randn([1, 1, 3, 3], dtype='float32')
        grid = paddle.randn([1, 2, 2, 2], dtype='float32')
        for mode in ['zeros', 'border', 'reflection']:
            out = F.grid_sample(x, grid, mode='bilinear', padding_mode=mode)
            self.assertEqual(out.shape, [1, 1, 2, 2])


class TestPixelShuffle(unittest.TestCase):
    """pixel_shuffle 测试 / pixel_shuffle tests"""

    def setUp(self):
        """测试前准备 / Setup before tests"""
        paddle.disable_static()

    def tearDown(self):
        """测试后清理 / Teardown after tests"""
        pass

    def test_pixel_shuffle_basic(self):
        """测试基本 pixel_shuffle / Test basic pixel_shuffle"""
        x = paddle.randn([2, 9, 4, 4], dtype='float32')
        out = F.pixel_shuffle(x, 3)
        self.assertEqual(out.shape, [2, 1, 12, 12])

    def test_pixel_shuffle_factor_2(self):
        """测试 upscale_factor=2 / Test upscale_factor=2"""
        x = paddle.randn([1, 16, 4, 4], dtype='float32')
        out = F.pixel_shuffle(x, 2)
        self.assertEqual(out.shape, [1, 4, 8, 8])

    def test_pixel_shuffle_nhwc(self):
        """测试 NHWC 格式 / Test NHWC format"""
        x = paddle.randn([2, 4, 4, 9], dtype='float32')
        out = F.pixel_shuffle(x, 3, data_format='NHWC')
        self.assertEqual(out.shape, [2, 12, 12, 1])


if __name__ == '__main__':
    unittest.main()
