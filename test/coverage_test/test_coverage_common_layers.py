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

# [AUTO-GENERATED] Unit test for paddle.nn.layer.common (Dropout, Linear, Flatten, etc.)
# 自动生成的单测，覆盖 paddle.nn.layer.common 模块中未覆盖的代码路径
# Target: cover uncovered lines in paddle/python/paddle/nn/layer/common.py
# 目标：覆盖 Linear, Dropout, Flatten, Pad 等 common layer 的各种参数路径

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. Linear - 各种参数 (in_features, out_features, weight_attr, bias_attr, name)
2. Dropout - p, mode, axis 参数
3. Flatten - start_axis, stop_axis 参数
4. Pad1D, Pad2D, Pad3D - 各种 padding 模式
5. Identity - 恒等层
"""

import unittest

import numpy as np

import paddle
from paddle import nn


class TestLinear(unittest.TestCase):
    """Test Linear layer.
    测试 Linear 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_linear_basic(self):
        """Basic Linear."""
        linear = nn.Linear(10, 5)
        x = paddle.randn([4, 10])
        out = linear(x)
        self.assertEqual(out.shape, [4, 5])

    def test_linear_no_bias(self):
        """Linear without bias."""
        linear = nn.Linear(10, 5, bias_attr=False)
        self.assertIsNone(linear.bias)
        x = paddle.randn([4, 10])
        out = linear(x)
        self.assertEqual(out.shape, [4, 5])

    def test_linear_with_name(self):
        """Linear with name."""
        linear = nn.Linear(10, 5, name='my_linear')
        x = paddle.randn([4, 10])
        out = linear(x)
        self.assertEqual(out.shape, [4, 5])

    def test_linear_3d_input(self):
        """Linear with 3D input."""
        linear = nn.Linear(10, 5)
        x = paddle.randn([2, 3, 10])
        out = linear(x)
        self.assertEqual(out.shape, [2, 3, 5])

    def test_linear_1d_input(self):
        """Linear with 1D input (per-sample)."""
        linear = nn.Linear(10, 5)
        x = paddle.randn([10])
        out = linear(x)
        self.assertEqual(out.shape, [5])


class TestDropout(unittest.TestCase):
    """Test Dropout layer.
    测试 Dropout 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_dropout_train(self):
        """Dropout in training mode."""
        dp = nn.Dropout(p=0.5)
        dp.train()
        x = paddle.ones([1000])
        out = dp(x)
        # Some values should be 0
        self.assertTrue(paddle.sum(out == 0).numpy() > 0)

    def test_dropout_eval(self):
        """Dropout in eval mode should be identity."""
        dp = nn.Dropout(p=0.5)
        dp.eval()
        x = paddle.ones([10])
        out = dp(x)
        np.testing.assert_allclose(out.numpy(), np.ones([10]))

    def test_dropout_zero_p(self):
        """Dropout with p=0 should be identity."""
        dp = nn.Dropout(p=0.0)
        x = paddle.randn([10])
        out = dp(x)
        np.testing.assert_allclose(out.numpy(), x.numpy())

    def test_dropout_axis(self):
        """Dropout along specific axis."""
        dp = nn.Dropout(p=0.5, axis=1)
        dp.train()
        x = paddle.ones([4, 10])
        out = dp(x)
        self.assertEqual(out.shape, [4, 10])


class TestFlatten(unittest.TestCase):
    """Test Flatten layer.
    测试 Flatten 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_flatten_default(self):
        """Flatten with default start_axis=1."""
        flatten = nn.Flatten()
        x = paddle.randn([2, 3, 4, 5])
        out = flatten(x)
        self.assertEqual(out.shape, [2, 60])

    def test_flatten_start_axis_0(self):
        """Flatten from axis 0."""
        flatten = nn.Flatten(start_axis=0)
        x = paddle.randn([2, 3, 4])
        out = flatten(x)
        self.assertEqual(out.shape, [24])

    def test_flatten_start_stop(self):
        """Flatten with custom start_axis and stop_axis."""
        flatten = nn.Flatten(start_axis=1, stop_axis=2)
        x = paddle.randn([2, 3, 4, 5])
        out = flatten(x)
        self.assertEqual(out.shape, [2, 12, 5])


class TestIdentity(unittest.TestCase):
    """Test Identity layer.
    测试 Identity 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_identity_basic(self):
        """Identity should pass through."""
        identity = nn.Identity()
        x = paddle.randn([2, 3, 4])
        out = identity(x)
        np.testing.assert_allclose(out.numpy(), x.numpy())


class TestPad2D(unittest.TestCase):
    """Test Pad2D layer.
    测试 Pad2D 层。
    """

    def setUp(self):
        paddle.disable_static()

    def test_pad2d_constant(self):
        """Pad2D with constant mode."""
        pad = nn.Pad2D(padding=1, mode='constant', value=0)
        x = paddle.randn([2, 3, 4, 4])
        out = pad(x)
        self.assertEqual(out.shape, [2, 3, 6, 6])

    def test_pad2d_reflect(self):
        """Pad2D with reflect mode."""
        pad = nn.Pad2D(padding=1, mode='reflect')
        x = paddle.randn([2, 3, 4, 4])
        out = pad(x)
        self.assertEqual(out.shape, [2, 3, 6, 6])

    def test_pad2d_replicate(self):
        """Pad2D with replicate mode."""
        pad = nn.Pad2D(padding=1, mode='replicate')
        x = paddle.randn([2, 3, 4, 4])
        out = pad(x)
        self.assertEqual(out.shape, [2, 3, 6, 6])

    def test_pad2d_tuple_padding(self):
        """Pad2D with tuple padding."""
        # Paddle Pad2D tuple: (pad_left, pad_right, pad_top, pad_bottom)
        # Input [2, 3, 4, 4] -> H=4+1+2=7, W=4+1+2=7 -> [2, 3, 7, 7]
        pad = nn.Pad2D(padding=(1, 2, 1, 2), mode='constant')
        x = paddle.randn([2, 3, 4, 4])
        out = pad(x)
        self.assertEqual(out.shape, [2, 3, 7, 7])

    def test_pad1d(self):
        """Pad1D basic."""
        pad = nn.Pad1D(padding=1, mode='constant')
        x = paddle.randn([2, 3, 10])
        out = pad(x)
        self.assertEqual(out.shape, [2, 3, 12])

    def test_pad3d(self):
        """Pad3D basic."""
        pad = nn.Pad3D(padding=1, mode='constant')
        x = paddle.randn([2, 3, 4, 4, 4])
        out = pad(x)
        self.assertEqual(out.shape, [2, 3, 6, 6, 6])


if __name__ == '__main__':
    unittest.main()
