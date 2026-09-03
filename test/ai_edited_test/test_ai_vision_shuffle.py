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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.vision
# 自动生成的单测，覆盖 paddle.nn.layer.vision 模块中未覆盖的代码
# Target: paddle/nn/layer/vision.py

"""
测试模块：paddle.nn.layer.vision
Test Module: paddle.nn.layer.vision

本测试覆盖以下功能：
This test covers the following functions:
1. PixelShuffle - 像素重排 / Pixel shuffle with different upscale factors
2. PixelUnshuffle - 像素逆重排 / Pixel unshuffle with different downscale factors
3. ChannelShuffle - 通道重排 / Channel shuffle with different groups
4. Error handling - 错误处理 / Type errors and value errors
"""

import unittest

import paddle
from paddle import nn


class TestPixelShuffle(unittest.TestCase):
    """测试PixelShuffle像素重排
    Test PixelShuffle"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_pixel_shuffle_basic(self):
        """测试基本PixelShuffle / Test basic PixelShuffle"""
        ps = nn.PixelShuffle(upscale_factor=3)
        x = paddle.randn([2, 9, 4, 4])
        out = ps(x)
        self.assertEqual(out.shape, [2, 1, 12, 12])

    def test_pixel_shuffle_factor_2(self):
        """测试upscale_factor=2 / Test upscale_factor=2"""
        ps = nn.PixelShuffle(2)
        x = paddle.randn([2, 4, 4, 4])
        out = ps(x)
        self.assertEqual(out.shape, [2, 1, 8, 8])

    def test_pixel_shuffle_nhwc(self):
        """测试NHWC格式 / Test PixelShuffle with NHWC"""
        ps = nn.PixelShuffle(2, data_format='NHWC')
        x = paddle.randn([2, 4, 4, 4])
        out = ps(x)
        self.assertEqual(out.shape, [2, 8, 8, 1])

    def test_pixel_shuffle_extra_repr_default(self):
        """测试默认extra_repr / Test extra_repr with default params"""
        ps = nn.PixelShuffle(3)
        r = ps.extra_repr()
        self.assertIn('upscale_factor=3', r)
        self.assertNotIn('data_format', r)

    def test_pixel_shuffle_extra_repr_nhwc(self):
        """测试NHWC的extra_repr / Test extra_repr with NHWC"""
        ps = nn.PixelShuffle(3, data_format='NHWC')
        r = ps.extra_repr()
        self.assertIn('NHWC', r)

    def test_pixel_shuffle_extra_repr_name(self):
        """测试带name的extra_repr / Test extra_repr with name"""
        ps = nn.PixelShuffle(3, name='test_ps')
        r = ps.extra_repr()
        self.assertIn('test_ps', r)


class TestPixelUnshuffle(unittest.TestCase):
    """测试PixelUnshuffle像素逆重排
    Test PixelUnshuffle"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_pixel_unshuffle_basic(self):
        """测试基本PixelUnshuffle / Test basic PixelUnshuffle"""
        pu = nn.PixelUnshuffle(downscale_factor=3)
        x = paddle.randn([2, 1, 12, 12])
        out = pu(x)
        self.assertEqual(out.shape, [2, 9, 4, 4])

    def test_pixel_unshuffle_factor_2(self):
        """测试downscale_factor=2 / Test downscale_factor=2"""
        pu = nn.PixelUnshuffle(2)
        x = paddle.randn([2, 1, 8, 8])
        out = pu(x)
        self.assertEqual(out.shape, [2, 4, 4, 4])

    def test_pixel_unshuffle_nhwc(self):
        """测试NHWC格式 / Test PixelUnshuffle with NHWC"""
        pu = nn.PixelUnshuffle(2, data_format='NHWC')
        x = paddle.randn([2, 8, 8, 1])
        out = pu(x)
        self.assertEqual(out.shape, [2, 4, 4, 4])

    def test_pixel_unshuffle_extra_repr(self):
        """测试extra_repr / Test extra_repr"""
        pu = nn.PixelUnshuffle(3, data_format='NHWC', name='test_pu')
        r = pu.extra_repr()
        self.assertIn('downscale_factor=3', r)
        self.assertIn('NHWC', r)
        self.assertIn('test_pu', r)


class TestChannelShuffle(unittest.TestCase):
    """测试ChannelShuffle通道重排
    Test ChannelShuffle"""

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def tearDown(self):
        paddle.enable_static()

    def test_channel_shuffle_basic(self):
        """测试基本ChannelShuffle / Test basic ChannelShuffle"""
        cs = nn.ChannelShuffle(groups=3)
        x = paddle.randn([2, 6, 4, 4])
        out = cs(x)
        self.assertEqual(out.shape, [2, 6, 4, 4])

    def test_channel_shuffle_groups_2(self):
        """测试groups=2 / Test groups=2"""
        cs = nn.ChannelShuffle(groups=2)
        x = paddle.randn([2, 4, 4, 4])
        out = cs(x)
        self.assertEqual(out.shape, [2, 4, 4, 4])

    def test_channel_shuffle_nhwc(self):
        """测试NHWC格式 / Test ChannelShuffle with NHWC"""
        cs = nn.ChannelShuffle(groups=2, data_format='NHWC')
        x = paddle.randn([2, 4, 4, 4])
        out = cs(x)
        self.assertEqual(out.shape, [2, 4, 4, 4])

    def test_channel_shuffle_extra_repr(self):
        """测试extra_repr / Test extra_repr"""
        cs = nn.ChannelShuffle(groups=3, data_format='NHWC', name='test_cs')
        r = cs.extra_repr()
        self.assertIn('groups=3', r)
        self.assertIn('NHWC', r)
        self.assertIn('test_cs', r)


class TestVisionErrorHandling(unittest.TestCase):
    """测试视觉层错误处理
    Test vision layers error handling"""

    def setUp(self):
        paddle.disable_static()

    def tearDown(self):
        paddle.enable_static()

    def test_pixel_shuffle_invalid_type(self):
        """测试PixelShuffle非整数因子 / Test PixelShuffle with non-int factor"""
        with self.assertRaises(TypeError):
            nn.PixelShuffle(upscale_factor=2.5)

    def test_pixel_shuffle_invalid_format(self):
        """测试PixelShuffle无效格式 / Test PixelShuffle with invalid format"""
        with self.assertRaises(ValueError):
            nn.PixelShuffle(2, data_format='INVALID')

    def test_pixel_unshuffle_invalid_type(self):
        """测试PixelUnshuffle非整数因子 / Test PixelUnshuffle with non-int factor"""
        with self.assertRaises(TypeError):
            nn.PixelUnshuffle(downscale_factor=2.5)

    def test_pixel_unshuffle_non_positive(self):
        """测试PixelUnshuffle非正因子 / Test PixelUnshuffle with non-positive factor"""
        with self.assertRaises(ValueError):
            nn.PixelUnshuffle(downscale_factor=0)

    def test_pixel_unshuffle_invalid_format(self):
        """测试PixelUnshuffle无效格式 / Test PixelUnshuffle with invalid format"""
        with self.assertRaises(ValueError):
            nn.PixelUnshuffle(2, data_format='INVALID')

    def test_channel_shuffle_invalid_type(self):
        """测试ChannelShuffle非整数groups / Test ChannelShuffle with non-int groups"""
        with self.assertRaises(TypeError):
            nn.ChannelShuffle(groups=2.5)

    def test_channel_shuffle_non_positive(self):
        """测试ChannelShuffle非正groups / Test ChannelShuffle with non-positive groups"""
        with self.assertRaises(ValueError):
            nn.ChannelShuffle(groups=0)

    def test_channel_shuffle_invalid_format(self):
        """测试ChannelShuffle无效格式 / Test ChannelShuffle with invalid format"""
        with self.assertRaises(ValueError):
            nn.ChannelShuffle(2, data_format='INVALID')


if __name__ == '__main__':
    unittest.main()
