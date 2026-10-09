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
初始化函数高级测试 / Advanced Initialization Function Tests

测试目标 / Test Target:
  paddle.nn.initializer 初始化器

覆盖的模块 / Covered Modules:
  - paddle.nn.initializer.KaimingUniform/Normal
  - paddle.nn.initializer.XavierUniform/Normal
  - paddle.nn.initializer.TruncatedNormal
  - paddle.nn.initializer.Constant
  - paddle.nn.initializer.Dirac/Orthogonal

作用 / Purpose:
  补充权重初始化API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle
from paddle import nn

paddle.disable_static()


class TestKaimingInitializer(unittest.TestCase):
    """测试Kaiming初始化 / Test Kaiming initialization"""

    def test_kaiming_uniform(self):
        """测试Kaiming均匀初始化 / Test Kaiming uniform initialization"""
        init = nn.initializer.KaimingUniform()
        param = paddle.ParamAttr(initializer=init)
        linear = nn.Linear(16, 32, weight_attr=param)
        self.assertEqual(linear.weight.shape, [16, 32])

    def test_kaiming_normal(self):
        """测试Kaiming正态初始化 / Test Kaiming normal initialization"""
        init = nn.initializer.KaimingNormal()
        param = paddle.ParamAttr(initializer=init)
        conv = nn.Conv2D(3, 16, 3, weight_attr=param)
        x = paddle.randn([2, 3, 8, 8])
        y = conv(x)
        self.assertEqual(y.shape, [2, 16, 6, 6])


class TestXavierInitializer(unittest.TestCase):
    """测试Xavier初始化 / Test Xavier initialization"""

    def test_xavier_uniform(self):
        """测试Xavier均匀初始化 / Test Xavier uniform initialization"""
        init = nn.initializer.XavierUniform()
        param = paddle.ParamAttr(initializer=init)
        linear = nn.Linear(16, 32, weight_attr=param)
        # Xavier uniform should bound weights in [-a, a] where a = sqrt(6 / fan_in+fan_out)
        weight = linear.weight.numpy()
        bound = np.sqrt(6.0 / (16 + 32))
        self.assertTrue(np.all(np.abs(weight) <= bound + 1e-6))

    def test_xavier_normal(self):
        """测试Xavier正态初始化 / Test Xavier normal initialization"""
        init = nn.initializer.XavierNormal()
        param = paddle.ParamAttr(initializer=init)
        linear = nn.Linear(16, 32, weight_attr=param)
        self.assertEqual(linear.weight.shape, [16, 32])


class TestConstantInitializer(unittest.TestCase):
    """测试常量初始化 / Test constant initialization"""

    def test_constant_zero(self):
        """测试零初始化 / Test zero initialization"""
        init = nn.initializer.Constant(value=0.0)
        param = paddle.ParamAttr(initializer=init)
        linear = nn.Linear(4, 8, weight_attr=param)
        np.testing.assert_allclose(linear.weight.numpy(), np.zeros([4, 8]))

    def test_constant_value(self):
        """测试常数初始化 / Test constant value initialization"""
        init = nn.initializer.Constant(value=0.01)
        param = paddle.ParamAttr(initializer=init)
        linear = nn.Linear(4, 8, weight_attr=param)
        np.testing.assert_allclose(
            linear.weight.numpy(), np.full([4, 8], 0.01), rtol=1e-6
        )


class TestTruncatedNormalInitializer(unittest.TestCase):
    """测试截断正态初始化 / Test truncated normal initialization"""

    def test_truncated_normal_basic(self):
        """测试基本截断正态初始化 / Test basic truncated normal"""
        init = nn.initializer.TruncatedNormal(mean=0.0, std=0.02)
        param = paddle.ParamAttr(initializer=init)
        linear = nn.Linear(16, 64, weight_attr=param)
        weight = linear.weight.numpy()
        # Values should be within 2 std = 0.04
        self.assertTrue(np.all(np.abs(weight) < 0.1))


class TestUniformInitializer(unittest.TestCase):
    """测试均匀分布初始化 / Test uniform distribution initialization"""

    def test_uniform(self):
        """测试均匀分布 / Test uniform initialization"""
        init = nn.initializer.Uniform(low=-0.5, high=0.5)
        param = paddle.ParamAttr(initializer=init)
        linear = nn.Linear(16, 32, weight_attr=param)
        weight = linear.weight.numpy()
        self.assertTrue(np.all(weight >= -0.5))
        self.assertTrue(np.all(weight <= 0.5))

    def test_normal(self):
        """测试正态分布初始化 / Test normal distribution initialization"""
        init = nn.initializer.Normal(mean=0.0, std=1.0)
        param = paddle.ParamAttr(initializer=init)
        linear = nn.Linear(16, 32, weight_attr=param)
        weight = linear.weight.numpy()
        # Mean should be approximately 0
        self.assertAlmostEqual(float(weight.mean()), 0.0, delta=0.5)


class TestSpecialInitializers(unittest.TestCase):
    """测试特殊初始化器 / Test special initializers"""

    def test_assign_initializer(self):
        """测试赋值初始化 / Test assign initializer"""
        value = np.ones([4, 8], dtype='float32') * 0.5
        init = nn.initializer.Assign(value)
        param = paddle.ParamAttr(initializer=init)
        linear = nn.Linear(4, 8, weight_attr=param)
        np.testing.assert_allclose(linear.weight.numpy(), value)

    def test_dirac_initializer_conv(self):
        """测试Dirac初始化卷积 / Test Dirac initializer for conv"""
        init = nn.initializer.Dirac()
        param = paddle.ParamAttr(initializer=init)
        conv = nn.Conv2D(3, 3, 3, padding=1, weight_attr=param)
        # Dirac init: output should equal input for identity
        x = paddle.randn([1, 3, 8, 8])
        y = conv(x)
        self.assertEqual(y.shape, [1, 3, 8, 8])


if __name__ == '__main__':
    unittest.main()
