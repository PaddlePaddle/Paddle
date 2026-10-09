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
随机操作高级测试 / Advanced Random Operations Tests

测试目标 / Test Target:
  paddle.tensor.random 随机张量操作

覆盖的模块 / Covered Modules:
  - paddle.seed/get_cuda_rng_state/set_cuda_rng_state: 随机状态
  - paddle.randperm: 随机排列
  - paddle.bernoulli: 伯努利采样
  - paddle.multinomial: 多项式采样

作用 / Purpose:
  补充随机操作API的测试，提升覆盖率。
"""

import unittest

import numpy as np

import paddle

paddle.disable_static()


class TestRandomSeed(unittest.TestCase):
    """测试随机种子 / Test random seed"""

    def test_seed_reproducibility(self):
        """测试随机种子可重复性 / Test seed reproducibility"""
        paddle.seed(42)
        x1 = paddle.randn([5])
        paddle.seed(42)
        x2 = paddle.randn([5])
        np.testing.assert_allclose(x1.numpy(), x2.numpy())

    def test_different_seeds(self):
        """测试不同种子 / Test different seeds"""
        paddle.seed(42)
        x1 = paddle.randn([10])
        paddle.seed(99)
        x2 = paddle.randn([10])
        # Different seeds should produce different results (very likely)
        self.assertFalse(np.allclose(x1.numpy(), x2.numpy()))


class TestRandperm(unittest.TestCase):
    """测试随机排列 / Test random permutation"""

    def test_randperm_basic(self):
        """测试基本随机排列 / Test basic randperm"""
        result = paddle.randperm(10)
        self.assertEqual(result.shape[0], 10)
        # All values 0-9 should be present
        sorted_result = np.sort(result.numpy())
        np.testing.assert_array_equal(sorted_result, np.arange(10))

    def test_randperm_dtype(self):
        """测试随机排列数据类型 / Test randperm dtype"""
        result = paddle.randperm(5, dtype='int64')
        self.assertEqual(result.dtype, paddle.int64)

    def test_randperm_shuffle(self):
        """测试随机排列用于数据打乱 / Test randperm for shuffling"""
        data = paddle.to_tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        perm = paddle.randperm(5)
        shuffled = data[perm]
        self.assertEqual(shuffled.shape, [5])
        # All values should still be present
        np.testing.assert_array_equal(
            np.sort(shuffled.numpy()), [10.0, 20.0, 30.0, 40.0, 50.0]
        )


class TestBernoulli(unittest.TestCase):
    """测试伯努利采样 / Test Bernoulli sampling"""

    def test_bernoulli_basic(self):
        """测试基本伯努利采样 / Test basic Bernoulli sampling"""
        probs = paddle.to_tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        result = paddle.bernoulli(probs)
        # All values should be 0 or 1
        self.assertTrue(bool(((result == 0) | (result == 1)).all().numpy()))

    def test_bernoulli_all_ones(self):
        """测试全1伯努利采样 / Test Bernoulli with all ones"""
        probs = paddle.ones([5])
        result = paddle.bernoulli(probs)
        np.testing.assert_array_equal(result.numpy(), np.ones(5))

    def test_bernoulli_all_zeros(self):
        """测试全0伯努利采样 / Test Bernoulli with all zeros"""
        probs = paddle.zeros([5])
        result = paddle.bernoulli(probs)
        np.testing.assert_array_equal(result.numpy(), np.zeros(5))


class TestMultinomial(unittest.TestCase):
    """测试多项式采样 / Test multinomial sampling"""

    def test_multinomial_basic(self):
        """测试基本多项式采样 / Test basic multinomial sampling"""
        weights = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        result = paddle.multinomial(weights, num_samples=100, replacement=True)
        self.assertEqual(result.shape, [100])
        # All indices should be valid
        self.assertTrue(bool((result >= 0).all().numpy()))
        self.assertTrue(bool((result < 4).all().numpy()))

    def test_multinomial_without_replacement(self):
        """测试无重复多项式采样 / Test multinomial without replacement"""
        weights = paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        result = paddle.multinomial(weights, num_samples=3, replacement=False)
        self.assertEqual(result.shape, [3])
        # All values should be unique
        self.assertEqual(len(np.unique(result.numpy())), 3)

    def test_multinomial_2d(self):
        """测试2D多项式采样 / Test 2D multinomial sampling"""
        weights = paddle.to_tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        result = paddle.multinomial(weights, num_samples=5, replacement=True)
        self.assertEqual(result.shape, [2, 5])


class TestRandomDropout(unittest.TestCase):
    """测试随机丢弃 / Test random dropout"""

    def test_dropout_training(self):
        """测试训练模式dropout / Test dropout in training mode"""
        dropout = paddle.nn.Dropout(p=0.5)
        dropout.train()
        x = paddle.ones([100, 100])
        result = dropout(x)
        # In training mode, some values should be 0
        zero_fraction = float((result == 0).sum().numpy()) / result.numel()
        # Should be approximately 0.5 (with tolerance)
        self.assertGreater(zero_fraction, 0.3)
        self.assertLess(zero_fraction, 0.7)

    def test_dropout_eval(self):
        """测试评估模式dropout / Test dropout in eval mode"""
        dropout = paddle.nn.Dropout(p=0.5)
        dropout.eval()
        x = paddle.ones([10, 10])
        result = dropout(x)
        # In eval mode, no dropout
        np.testing.assert_allclose(result.numpy(), x.numpy())


if __name__ == '__main__':
    unittest.main()
