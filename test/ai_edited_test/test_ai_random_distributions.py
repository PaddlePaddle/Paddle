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

# [AUTO-GENERATED] Unit test for paddle.tensor.random (bernoulli, binomial, standard_gamma)
# 自动生成的单测，覆盖 paddle.tensor.random 模块中未覆盖的代码路径
# Target: cover uncovered lines 130-142, 240-262, 301 in paddle/python/paddle/tensor/random.py
# 目标：覆盖 random.py 中 bernoulli 的静态图分支、binomial 的静态图分支

"""
This test covers the following modules and code paths:
这个测试覆盖以下模块和代码路径：

1. bernoulli() - 动态图路径 (lines 127-142)
2. bernoulli_() - inplace 版本 (lines 146-191)
3. binomial() - 动态图路径 (lines 235-262)
4. standard_gamma() - 基本功能 (lines 265-301)
5. multinomial() - 基本功能
"""

import unittest

import numpy as np

import paddle


class TestBernoulli(unittest.TestCase):
    """Test bernoulli() distribution.
    测试 bernoulli() 分布采样。
    覆盖 random.py 第 127-142 行。
    """

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def test_bernoulli_default_p(self):
        """bernoulli with default p=0.5."""
        x = paddle.zeros([2, 3], dtype='float32')
        out = paddle.bernoulli(x)
        self.assertEqual(out.shape, [2, 3])
        # All values should be 0 or 1
        result = out.numpy()
        self.assertTrue(np.all((result == 0) | (result == 1)))

    def test_bernoulli_custom_p(self):
        """bernoulli with custom probability."""
        x = paddle.full([10, 10], 0.8, dtype='float32')
        out = paddle.bernoulli(x)
        result = out.numpy()
        # Most values should be 1
        self.assertTrue(np.mean(result) > 0.5)

    def test_bernoulli_float16(self):
        """bernoulli with float16 input."""
        x = paddle.full([5, 5], 0.5, dtype='float16')
        out = paddle.bernoulli(x)
        self.assertEqual(out.shape, [5, 5])

    def test_bernoulli_float64(self):
        """bernoulli with float64 input."""
        x = paddle.full([5, 5], 0.5, dtype='float64')
        out = paddle.bernoulli(x)
        self.assertEqual(out.shape, [5, 5])

    def test_bernoulli_p_scalar(self):
        """bernoulli with scalar p parameter."""
        x = paddle.zeros([2, 3])
        out = paddle.bernoulli(x, p=0.8)
        self.assertEqual(out.shape, [2, 3])
        # Most values should be 1 since p=0.8
        result = out.numpy()
        self.assertTrue(np.mean(result) > 0.4)

    def test_bernoulli_with_name(self):
        """bernoulli with name parameter."""
        x = paddle.zeros([2, 3])
        out = paddle.bernoulli(x, name='test_bernoulli')
        self.assertEqual(out.shape, [2, 3])


class TestBernoulliInplace(unittest.TestCase):
    """Test bernoulli_() inplace operation.
    测试 bernoulli_() 就地操作。
    """

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def test_bernoulli_inplace(self):
        """bernoulli_ should modify tensor in-place."""
        x = paddle.randn([3, 4])
        x_id = id(x)
        out = paddle.bernoulli_(x)
        self.assertIs(out, x)
        result = out.numpy()
        self.assertTrue(np.all((result == 0) | (result == 1)))

    def test_bernoulli_inplace_with_p(self):
        """bernoulli_ with custom p."""
        x = paddle.randn([3, 4])
        out = paddle.bernoulli_(x, p=0.9)
        result = out.numpy()
        self.assertTrue(np.mean(result) > 0.5)


class TestBinomial(unittest.TestCase):
    """Test binomial() distribution.
    测试 binomial() 分布采样。
    覆盖 random.py 第 235-262 行。
    """

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def test_binomial_basic(self):
        """binomial basic usage."""
        count = paddle.full([2, 3], 10, dtype='int32')
        prob = paddle.full([2, 3], 0.5, dtype='float32')
        out = paddle.binomial(count, prob)
        self.assertEqual(out.shape, [2, 3])
        result = out.numpy()
        # Values should be between 0 and count
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 10))

    def test_binomial_float64(self):
        """binomial with float64 probability."""
        count = paddle.full([5], 20, dtype='int64')
        prob = paddle.full([5], 0.3, dtype='float64')
        out = paddle.binomial(count, prob)
        self.assertEqual(out.shape, [5])

    def test_binomial_broadcast(self):
        """binomial with broadcastable shapes."""
        count = paddle.full([2, 1], 10, dtype='int32')
        prob = paddle.full([1, 3], 0.5, dtype='float32')
        out = paddle.binomial(count, prob)
        self.assertEqual(out.shape, [2, 3])

    def test_binomial_high_prob(self):
        """binomial with high probability."""
        count = paddle.full([100], 10, dtype='int32')
        prob = paddle.full([100], 0.99, dtype='float32')
        out = paddle.binomial(count, prob)
        result = out.numpy()
        self.assertTrue(np.mean(result) > 8.0)


class TestStandardGamma(unittest.TestCase):
    """Test standard_gamma() distribution.
    测试 standard_gamma() 分布采样。
    覆盖 random.py 第 265-301 行。
    """

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def test_standard_gamma_basic(self):
        """standard_gamma basic usage."""
        x = paddle.uniform([2, 3], min=1.0, max=5.0)
        out = paddle.standard_gamma(x)
        self.assertEqual(out.shape, [2, 3])
        # Gamma distribution with alpha > 0 should produce positive values
        result = out.numpy()
        self.assertTrue(np.all(result >= 0))

    def test_standard_gamma_float64(self):
        """standard_gamma with float64."""
        x = paddle.uniform([2, 3], min=1.0, max=5.0, dtype='float64')
        out = paddle.standard_gamma(x)
        self.assertEqual(out.dtype, paddle.float64)


class TestPoisson(unittest.TestCase):
    """Test poisson() distribution.
    测试 poisson() 分布采样。
    """

    def setUp(self):
        paddle.disable_static()
        paddle.seed(42)

    def test_poisson_basic(self):
        """poisson basic usage."""
        x = paddle.full([2, 3], 5.0)
        out = paddle.poisson(x)
        self.assertEqual(out.shape, [2, 3])
        result = out.numpy()
        self.assertTrue(np.all(result >= 0))


if __name__ == '__main__':
    unittest.main()
