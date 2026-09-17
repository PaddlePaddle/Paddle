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

# [AUTO-GENERATED] Test file for paddle.nn.clip
# 覆盖模块: paddle/nn/clip.py
# 未覆盖行: 88,89,92,94,95,99,103,110,143,144,145,151,206,207,208,211,212,213,219,229,230,231,232,242,246,273,274,277,278,280,281,282,283,288,291,334,335,336,338,339
# Covered module: paddle/nn/clip.py
# Uncovered lines: 88,89,92,94,95,99,103,110,143,144,145,151,206,207,208,211,212,213,219,229,230,231,232,242,246,273,274,277,278,280,281,282,283,288,291,334,335,336,338,339

import unittest

import numpy as np

import paddle
from paddle.nn.clip import (
    ClipGradByGlobalNorm,
    ClipGradByNorm,
    ClipGradByValue,
)


class TestClipGradByValue(unittest.TestCase):
    """测试 ClipGradByValue 类
    Test ClipGradByValue class"""

    def test_clip_grad_by_value_init(self):
        """测试 ClipGradByValue 初始化
        Test ClipGradByValue initialization"""
        clip = ClipGradByValue(min=-1.0, max=1.0)
        self.assertEqual(clip.min, -1.0)
        self.assertEqual(clip.max, 1.0)

    def test_clip_grad_by_value_callable(self):
        """测试 ClipGradByValue 可调用
        Test ClipGradByValue is callable"""
        clip = ClipGradByValue(min=-1.0, max=1.0)
        self.assertTrue(callable(clip))


class TestClipGradByNorm(unittest.TestCase):
    """测试 ClipGradByNorm 类
    Test ClipGradByNorm class"""

    def test_clip_grad_by_norm_init(self):
        """测试 ClipGradByNorm 初始化
        Test ClipGradByNorm initialization"""
        clip = ClipGradByNorm(clip_norm=1.0)
        self.assertEqual(clip.clip_norm, 1.0)

    def test_clip_grad_by_norm_callable(self):
        """测试 ClipGradByNorm 可调用
        Test ClipGradByNorm is callable"""
        clip = ClipGradByNorm(clip_norm=1.0)
        self.assertTrue(callable(clip))


class TestClipGradByGlobalNorm(unittest.TestCase):
    """测试 ClipGradByGlobalNorm 类
    Test ClipGradByGlobalNorm class"""

    def test_clip_grad_by_global_norm_init(self):
        """测试 ClipGradByGlobalNorm 初始化
        Test ClipGradByGlobalNorm initialization"""
        clip = ClipGradByGlobalNorm(clip_norm=1.0)
        self.assertEqual(clip.clip_norm, 1.0)

    def test_clip_grad_by_global_norm_with_group_name(self):
        """测试带 group_name 的 ClipGradByGlobalNorm
        Test ClipGradByGlobalNorm with group_name"""
        clip = ClipGradByGlobalNorm(clip_norm=1.0, group_name="default")
        self.assertEqual(clip.group_name, "default")

    def test_clip_grad_by_global_norm_callable(self):
        """测试 ClipGradByGlobalNorm 可调用
        Test ClipGradByGlobalNorm is callable"""
        clip = ClipGradByGlobalNorm(clip_norm=1.0)
        self.assertTrue(callable(clip))


class TestTensorClip(unittest.TestCase):
    """测试 paddle.clip 张量操作
    Test paddle.clip tensor operation"""

    def test_clip_min_max(self):
        """测试带 min 和 max 的 clip
        Test clip with min and max"""
        x = paddle.to_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = paddle.clip(x, min=-1.0, max=1.0)
        np.testing.assert_array_equal(
            result.numpy(), [-1.0, -1.0, 0.0, 1.0, 1.0]
        )

    def test_clip_min_only(self):
        """测试只带 min 的 clip
        Test clip with min only"""
        x = paddle.to_tensor([-2.0, 0.0, 2.0])
        result = paddle.clip(x, min=0.0)
        np.testing.assert_array_equal(result.numpy(), [0.0, 0.0, 2.0])

    def test_clip_max_only(self):
        """测试只带 max 的 clip
        Test clip with max only"""
        x = paddle.to_tensor([-2.0, 0.0, 2.0])
        result = paddle.clip(x, max=0.0)
        np.testing.assert_array_equal(result.numpy(), [-2.0, 0.0, 0.0])

    def test_clip_2d(self):
        """测试2D输入的 clip
        Test clip with 2D input"""
        x = paddle.to_tensor([[-2.0, 0.0], [1.0, 3.0]])
        result = paddle.clip(x, min=-1.0, max=2.0)
        expected = [[-1.0, 0.0], [1.0, 2.0]]
        np.testing.assert_array_equal(result.numpy(), expected)

    def test_clip_float64(self):
        """测试 float64 类型的 clip
        Test clip with float64 dtype"""
        x = paddle.to_tensor([-2.0, 0.0, 2.0], dtype='float64')
        result = paddle.clip(x, min=-1.0, max=1.0)
        self.assertEqual(result.dtype, paddle.float64)

    def test_clip_scalar_min(self):
        """测试标量 min 的 clip
        Test clip with scalar min"""
        min_val = paddle.to_tensor(-0.5)
        x = paddle.to_tensor([-2.0, 0.0, 2.0])
        result = paddle.clip(x, min=min_val)
        np.testing.assert_allclose(result.numpy(), [-0.5, 0.0, 2.0])

    def test_clip_scalar_max(self):
        """测试标量 max 的 clip
        Test clip with scalar max"""
        max_val = paddle.to_tensor(0.5)
        x = paddle.to_tensor([-2.0, 0.0, 2.0])
        result = paddle.clip(x, max=max_val)
        np.testing.assert_allclose(result.numpy(), [-2.0, 0.0, 0.5])


if __name__ == '__main__':
    unittest.main()
