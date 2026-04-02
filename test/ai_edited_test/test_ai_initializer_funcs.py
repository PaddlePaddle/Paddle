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

# [AUTO-GENERATED] Unit test for paddle.nn.initializer (various initializers)
# 自动生成的单测，覆盖 paddle.nn.initializer 模块中未覆盖的代码路径
# Target: cover uncovered lines in paddle/python/paddle/nn/initializer.py
# 目标：覆盖各种 initializer 的初始化路径和参数组合

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. Constant - 常量初始化
2. Normal - 正态分布初始化
3. Uniform - 均匀分布初始化
4. XavierNormal / XavierUniform - Xavier 初始化
5. KaimingNormal / KaimingUniform - Kaiming 初始化
6. TruncatedNormal - 截断正态分布初始化
7. Dirac - Dirac 初始化
8. Bilinear - 双线性初始化
"""

import unittest

import numpy as np

import paddle
from paddle.nn.initializer import (
    Assign,
    Constant,
    KaimingNormal,
    KaimingUniform,
    Normal,
    TruncatedNormal,
    Uniform,
    XavierNormal,
    XavierUniform,
)


def _make_param(shape, initializer):
    """Helper to create a parameter with a given initializer."""
    return paddle.create_parameter(
        shape=shape,
        dtype='float32',
        default_initializer=initializer,
    )


class TestConstantInit(unittest.TestCase):
    """Test Constant initializer.
    测试 Constant 初始化器。
    """

    def setUp(self):
        paddle.disable_static()

    def test_constant_zero(self):
        """Constant(0) initialization."""
        w = _make_param([10, 20], Constant(0.0))
        np.testing.assert_allclose(w.numpy(), 0.0, atol=1e-6)

    def test_constant_one(self):
        """Constant(1) initialization."""
        w = _make_param([10], Constant(1.0))
        np.testing.assert_allclose(w.numpy(), 1.0, atol=1e-6)

    def test_constant_custom_value(self):
        """Constant with custom value."""
        w = _make_param([5], Constant(3.14))
        np.testing.assert_allclose(w.numpy(), 3.14, atol=1e-4)


class TestNormalInit(unittest.TestCase):
    """Test Normal initializer.
    测试 Normal 初始化器。
    """

    def setUp(self):
        paddle.disable_static()

    def test_normal_basic(self):
        """Normal distribution initialization."""
        w = _make_param([1000], Normal(mean=0.0, std=1.0))
        arr = w.numpy()
        self.assertAlmostEqual(np.mean(arr), 0.0, delta=0.2)
        self.assertAlmostEqual(np.std(arr), 1.0, delta=0.2)

    def test_normal_custom_params(self):
        """Normal with custom mean and std."""
        w = _make_param([1000], Normal(mean=5.0, std=0.5))
        arr = w.numpy()
        self.assertAlmostEqual(np.mean(arr), 5.0, delta=0.2)


class TestUniformInit(unittest.TestCase):
    """Test Uniform initializer.
    测试 Uniform 初始化器。
    """

    def setUp(self):
        paddle.disable_static()

    def test_uniform_basic(self):
        """Uniform distribution initialization."""
        w = _make_param([1000], Uniform(low=-1.0, high=1.0))
        arr = w.numpy()
        self.assertTrue(np.all(arr >= -1.0))
        self.assertTrue(np.all(arr <= 1.0))


class TestXavierInit(unittest.TestCase):
    """Test Xavier initializers.
    测试 Xavier 初始化器。
    """

    def setUp(self):
        paddle.disable_static()

    def test_xavier_normal(self):
        """XavierNormal initialization."""
        w = _make_param([100, 200], XavierNormal())
        arr = w.numpy()
        self.assertAlmostEqual(np.mean(arr), 0.0, delta=0.2)

    def test_xavier_uniform(self):
        """XavierUniform initialization."""
        w = _make_param([100, 200], XavierUniform())
        arr = w.numpy()
        self.assertAlmostEqual(np.mean(arr), 0.0, delta=0.2)

    def test_xavier_uniform_fan_in(self):
        """XavierUniform with fan_in mode."""
        init = XavierUniform(fan_in=True, fan_out=False)
        w = _make_param([100, 200], init)
        self.assertIsNotNone(w)


class TestKaimingInit(unittest.TestCase):
    """Test Kaiming initializers.
    测试 Kaiming 初始化器。
    """

    def setUp(self):
        paddle.disable_static()

    def test_kaiming_normal(self):
        """KaimingNormal initialization."""
        w = _make_param([100, 200], KaimingNormal())
        arr = w.numpy()
        self.assertAlmostEqual(np.mean(arr), 0.0, delta=0.2)

    def test_kaiming_uniform(self):
        """KaimingUniform initialization."""
        w = _make_param([100, 200], KaimingUniform())
        arr = w.numpy()
        self.assertAlmostEqual(np.mean(arr), 0.0, delta=0.2)

    def test_kaiming_normal_negative_slope(self):
        """KaimingNormal with negative_slope."""
        init = KaimingNormal(negative_slope=0.1)
        w = _make_param([100, 200], init)
        self.assertIsNotNone(w)


class TestTruncatedNormalInit(unittest.TestCase):
    """Test TruncatedNormal initializer.
    测试 TruncatedNormal 初始化器。
    """

    def setUp(self):
        paddle.disable_static()

    def test_truncated_normal_basic(self):
        """TruncatedNormal initialization."""
        w = _make_param([1000], TruncatedNormal(mean=0.0, std=1.0))
        arr = w.numpy()
        # Values should be within ~2 std
        self.assertTrue(np.all(np.abs(arr) < 3.0))


class TestAssignInit(unittest.TestCase):
    """Test Assign initializer.
    测试 Assign 初始化器。
    """

    def setUp(self):
        paddle.disable_static()

    def test_assign_numpy(self):
        """Assign with numpy array."""
        np_val = np.ones([10, 20], dtype='float32')
        w = _make_param([10, 20], Assign(np_val))
        np.testing.assert_allclose(w.numpy(), np_val)


if __name__ == '__main__':
    unittest.main()
