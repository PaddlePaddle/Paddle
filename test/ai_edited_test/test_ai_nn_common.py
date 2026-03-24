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

# [AUTO-GENERATED] Unit test for paddle.nn.functional.common
# 自动生成的单测，覆盖 paddle.nn.functional.common 模块中未覆盖的代码

"""
测试模块：paddle.nn.functional.common (pad, label_smooth, alpha_dropout, dropout)
Test Module: paddle.nn.functional.common

本测试覆盖以下功能：
This test covers the following functions:
1. pad - 填充操作 / Padding with reflect/replicate/circular modes
2. label_smooth - 标签平滑 / Label smoothing
3. alpha_dropout - Alpha dropout / Alpha dropout
4. dropout - Dropout / Dropout with axis parameter

覆盖的未覆盖行：pad reflect/replicate模式, label_smooth, alpha_dropout
"""

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F


class TestPadModes(unittest.TestCase):
    """测试pad不同填充模式
    Test pad with different padding modes"""

    def setUp(self):
        paddle.disable_static()

    def test_pad_constant_1d(self):
        """1D常数填充 / 1D constant pad"""
        x = (
            paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
            .unsqueeze(0)
            .unsqueeze(0)
        )
        out = F.pad(x, [1, 2], mode='constant', value=0.0)
        self.assertEqual(out.shape[-1], 6)

    def test_pad_reflect_1d(self):
        """1D反射填充 / 1D reflect pad"""
        x = (
            paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype='float32')
            .unsqueeze(0)
            .unsqueeze(0)
        )
        out = F.pad(x, [1, 1], mode='reflect')
        expected = np.array([[[2.0, 1.0, 2.0, 3.0, 4.0, 3.0]]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_pad_replicate_1d(self):
        """1D复制填充 / 1D replicate pad"""
        x = (
            paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
            .unsqueeze(0)
            .unsqueeze(0)
        )
        out = F.pad(x, [2, 2], mode='replicate')
        expected = np.array([[[1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_pad_circular_1d(self):
        """1D循环填充 / 1D circular pad"""
        x = (
            paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
            .unsqueeze(0)
            .unsqueeze(0)
        )
        out = F.pad(x, [1, 1], mode='circular')
        expected = np.array([[[3.0, 1.0, 2.0, 3.0, 1.0]]])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5)

    def test_pad_2d(self):
        """2D填充 / 2D pad"""
        x = paddle.randn([1, 1, 4, 4])
        out = F.pad(x, [1, 1, 1, 1], mode='constant', value=0.0)
        self.assertEqual(list(out.shape), [1, 1, 6, 6])


class TestLabelSmooth(unittest.TestCase):
    """测试label_smooth标签平滑
    Test label_smooth function"""

    def setUp(self):
        paddle.disable_static()

    def test_label_smooth_basic(self):
        """基本标签平滑 / Basic label smoothing"""
        label = paddle.to_tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype='float32'
        )
        out = F.label_smooth(label, epsilon=0.1)
        self.assertIsNotNone(out)
        # 平滑后最大值应该小于1.0
        self.assertTrue(float(out.max().numpy()) < 1.0)
        # 平滑后最小值应该大于0.0
        self.assertTrue(float(out.min().numpy()) > 0.0)

    def test_label_smooth_epsilon_zero(self):
        """epsilon=0不平滑 / No smoothing when epsilon=0"""
        label = paddle.to_tensor([[1.0, 0.0, 0.0]], dtype='float32')
        out = F.label_smooth(label, epsilon=0.0)
        np.testing.assert_allclose(out.numpy(), label.numpy(), rtol=1e-5)


class TestDropout(unittest.TestCase):
    """测试dropout操作
    Test dropout operations"""

    def setUp(self):
        paddle.disable_static()

    def test_dropout_eval(self):
        """eval模式下dropout / Dropout in eval mode"""
        x = paddle.ones([3, 4], dtype='float32')
        out = F.dropout(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-5)

    def test_dropout_training(self):
        """training模式下dropout / Dropout in training mode"""
        x = paddle.ones([100, 100], dtype='float32')
        out = F.dropout(x, p=0.5, training=True)
        # 期望大约一半的元素被置零
        zero_ratio = float((out == 0).astype('float32').mean().numpy())
        self.assertTrue(0.3 < zero_ratio < 0.7)

    def test_dropout2d(self):
        """2D dropout / Dropout 2D"""
        x = paddle.ones([2, 3, 4, 4], dtype='float32')
        out = F.dropout2d(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-5)

    def test_dropout3d(self):
        """3D dropout / Dropout 3D"""
        x = paddle.ones([2, 3, 4, 4, 4], dtype='float32')
        out = F.dropout3d(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-5)


class TestAlphaDropout(unittest.TestCase):
    """测试alpha_dropout
    Test alpha_dropout function"""

    def setUp(self):
        paddle.disable_static()

    def test_alpha_dropout_eval(self):
        """eval模式 / Alpha dropout in eval mode"""
        x = paddle.randn([3, 4])
        out = F.alpha_dropout(x, p=0.5, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), rtol=1e-5)

    def test_alpha_dropout_training(self):
        """training模式 / Alpha dropout in training mode"""
        x = paddle.randn([100, 100])
        out = F.alpha_dropout(x, p=0.5, training=True)
        self.assertEqual(list(out.shape), [100, 100])


if __name__ == '__main__':
    unittest.main()
