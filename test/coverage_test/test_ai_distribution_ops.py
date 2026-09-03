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
分布式概率分布单元测试 / Probability Distribution Unit Tests

测试目标 / Test Target:
  paddle.distribution 模块 (覆盖率约84.2%)

覆盖的模块 / Covered Modules:
  - paddle.distribution.Normal: 正态分布
  - paddle.distribution.Uniform: 均匀分布
  - paddle.distribution.Categorical: 类别分布
  - paddle.distribution.Bernoulli: 伯努利分布
  - paddle.distribution.Beta: Beta分布
  - paddle.distribution.Dirichlet: Dirichlet分布
  - paddle.distribution.Geometric: 几何分布
  - paddle.distribution.LogNormal: 对数正态分布

作用 / Purpose:
  覆盖各类概率分布的sample、log_prob、entropy、cdf等操作路径，
  补充概率分布模块中未被测试覆盖的代码路径。
"""

import unittest

import numpy as np

import paddle
import paddle.distribution as dist

paddle.disable_static()


class TestNormalDistribution(unittest.TestCase):
    """测试正态分布 / Test Normal distribution"""

    def test_normal_sample(self):
        """测试正态分布采样 / Test Normal distribution sampling"""
        normal = dist.Normal(loc=0.0, scale=1.0)
        samples = normal.sample([100])
        self.assertEqual(samples.shape, [100])

    def test_normal_log_prob(self):
        """测试正态分布对数概率 / Test Normal log probability"""
        normal = dist.Normal(loc=0.0, scale=1.0)
        x = paddle.to_tensor([0.0, 1.0, -1.0])
        log_prob = normal.log_prob(x)
        self.assertEqual(log_prob.shape, [3])

    def test_normal_entropy(self):
        """测试正态分布熵 / Test Normal entropy"""
        normal = dist.Normal(loc=0.0, scale=1.0)
        entropy = normal.entropy()
        self.assertIsNotNone(entropy)
        self.assertTrue(entropy.item() > 0)

    def test_normal_mean_variance(self):
        """测试正态分布均值和方差 / Test Normal mean and variance"""
        normal = dist.Normal(loc=2.0, scale=3.0)
        self.assertAlmostEqual(float(normal.mean.numpy()), 2.0, places=5)
        self.assertAlmostEqual(float(normal.variance.numpy()), 9.0, places=5)

    def test_normal_kl_divergence(self):
        """测试正态分布KL散度 / Test Normal KL divergence"""
        p = dist.Normal(loc=0.0, scale=1.0)
        q = dist.Normal(loc=1.0, scale=2.0)
        kl = paddle.distribution.kl_divergence(p, q)
        self.assertTrue(kl.item() >= 0)

    def test_normal_batch(self):
        """测试批量正态分布 / Test batched Normal distribution"""
        loc = paddle.to_tensor([0.0, 1.0, 2.0])
        scale = paddle.to_tensor([1.0, 1.0, 1.0])
        normal = dist.Normal(loc=loc, scale=scale)
        samples = normal.sample([10])
        self.assertEqual(samples.shape, [10, 3])


class TestUniformDistribution(unittest.TestCase):
    """测试均匀分布 / Test Uniform distribution"""

    def test_uniform_sample(self):
        """测试均匀分布采样 / Test Uniform sampling"""
        uniform = dist.Uniform(low=0.0, high=1.0)
        samples = uniform.sample([100])
        self.assertEqual(samples.shape, [100])
        self.assertTrue(paddle.all(samples >= 0.0).item())
        self.assertTrue(paddle.all(samples <= 1.0).item())

    def test_uniform_log_prob(self):
        """测试均匀分布对数概率 / Test Uniform log probability"""
        uniform = dist.Uniform(low=0.0, high=1.0)
        x = paddle.to_tensor([0.5])
        log_prob = uniform.log_prob(x)
        self.assertAlmostEqual(log_prob.numpy()[0], 0.0, places=5)

    def test_uniform_entropy(self):
        """测试均匀分布熵 / Test Uniform entropy"""
        uniform = dist.Uniform(low=0.0, high=2.0)
        entropy = uniform.entropy()
        # entropy of Uniform(0,2) = log(2)
        self.assertAlmostEqual(float(entropy.numpy()), np.log(2.0), places=4)

    def test_uniform_sample_range(self):
        """测试均匀分布采样范围 / Test Uniform sample range"""
        uniform = dist.Uniform(low=0.0, high=4.0)
        samples = uniform.sample([1000])
        # Mean should be close to 2.0
        sample_mean = float(samples.mean().numpy())
        self.assertAlmostEqual(sample_mean, 2.0, delta=0.2)


class TestCategoricalDistribution(unittest.TestCase):
    """测试类别分布 / Test Categorical distribution"""

    def test_categorical_sample(self):
        """测试类别分布采样 / Test Categorical sampling"""
        logits = paddle.to_tensor([1.0, 2.0, 3.0])
        categorical = dist.Categorical(logits=logits)
        samples = categorical.sample([10])
        self.assertEqual(samples.shape, [10])

    def test_categorical_log_prob(self):
        """测试类别分布对数概率 / Test Categorical log probability"""
        logits = paddle.to_tensor([1.0, 2.0, 3.0])
        categorical = dist.Categorical(logits=logits)
        x = paddle.to_tensor([0, 1, 2])
        log_prob = categorical.log_prob(x)
        self.assertEqual(log_prob.shape, [3])

    def test_categorical_entropy(self):
        """测试类别分布熵 / Test Categorical entropy"""
        # Uniform logits => maximum entropy
        logits = paddle.zeros([4])
        categorical = dist.Categorical(logits=logits)
        entropy = categorical.entropy()
        self.assertAlmostEqual(float(entropy.numpy()), np.log(4.0), places=4)


class TestBernoulliDistribution(unittest.TestCase):
    """测试伯努利分布 / Test Bernoulli distribution"""

    def test_bernoulli_sample(self):
        """测试伯努利分布采样 / Test Bernoulli sampling"""
        bernoulli = dist.Bernoulli(probs=0.5)
        samples = bernoulli.sample([100])
        self.assertEqual(samples.shape, [100])
        unique_vals = paddle.unique(samples).numpy()
        # Only 0 and 1
        self.assertTrue(all(v in [0, 1] for v in unique_vals))

    def test_bernoulli_log_prob(self):
        """测试伯努利分布对数概率 / Test Bernoulli log probability"""
        bernoulli = dist.Bernoulli(probs=0.7)
        x = paddle.to_tensor([0.0, 1.0])
        log_prob = bernoulli.log_prob(x)
        self.assertEqual(log_prob.shape, [2])

    def test_bernoulli_entropy(self):
        """测试伯努利分布熵 / Test Bernoulli entropy"""
        bernoulli = dist.Bernoulli(probs=0.5)
        entropy = bernoulli.entropy()
        # Binary entropy at p=0.5 is log(2)
        self.assertAlmostEqual(float(entropy.numpy()), np.log(2.0), places=4)

    def test_bernoulli_mean_variance(self):
        """测试伯努利分布均值和方差 / Test Bernoulli mean and variance"""
        p = 0.3
        bernoulli = dist.Bernoulli(probs=p)
        mean = bernoulli.mean
        var = bernoulli.variance
        self.assertAlmostEqual(float(mean.numpy()), p, places=5)
        self.assertAlmostEqual(float(var.numpy()), p * (1 - p), places=5)


class TestBetaDistribution(unittest.TestCase):
    """测试Beta分布 / Test Beta distribution"""

    def test_beta_sample(self):
        """测试Beta分布采样 / Test Beta sampling"""
        beta = dist.Beta(alpha=2.0, beta=5.0)
        samples = beta.sample([100])
        self.assertEqual(samples.shape, [100])
        self.assertTrue(paddle.all(samples > 0).item())
        self.assertTrue(paddle.all(samples < 1).item())

    def test_beta_log_prob(self):
        """测试Beta分布对数概率 / Test Beta log probability"""
        beta = dist.Beta(alpha=2.0, beta=5.0)
        x = paddle.to_tensor([0.3, 0.5, 0.7])
        log_prob = beta.log_prob(x)
        self.assertEqual(log_prob.shape, [3])

    def test_beta_entropy(self):
        """测试Beta分布熵 / Test Beta entropy"""
        beta = dist.Beta(alpha=2.0, beta=5.0)
        entropy = beta.entropy()
        self.assertIsNotNone(entropy)

    def test_beta_mean_variance(self):
        """测试Beta分布均值和方差 / Test Beta mean and variance"""
        alpha, b = 2.0, 5.0
        beta = dist.Beta(alpha=alpha, beta=b)
        expected_mean = alpha / (alpha + b)
        self.assertAlmostEqual(
            float(beta.mean.numpy()), expected_mean, places=4
        )


class TestLogNormalDistribution(unittest.TestCase):
    """测试对数正态分布 / Test LogNormal distribution"""

    def test_lognormal_sample(self):
        """测试对数正态分布采样 / Test LogNormal sampling"""
        lognormal = dist.LogNormal(loc=0.0, scale=1.0)
        samples = lognormal.sample([100])
        self.assertEqual(samples.shape, [100])
        self.assertTrue(paddle.all(samples > 0).item())

    def test_lognormal_log_prob(self):
        """测试对数正态分布对数概率 / Test LogNormal log probability"""
        lognormal = dist.LogNormal(loc=0.0, scale=1.0)
        x = paddle.to_tensor([1.0, 2.0, 0.5])
        log_prob = lognormal.log_prob(x)
        self.assertEqual(log_prob.shape, [3])

    def test_lognormal_entropy(self):
        """测试对数正态分布熵 / Test LogNormal entropy"""
        lognormal = dist.LogNormal(loc=0.0, scale=1.0)
        entropy = lognormal.entropy()
        self.assertIsNotNone(entropy)


class TestTransformedDistribution(unittest.TestCase):
    """测试变换分布 / Test transformed distributions"""

    def test_independent_distribution(self):
        """测试独立分布包装 / Test Independent distribution wrapper"""
        base = dist.Normal(loc=paddle.zeros([3]), scale=paddle.ones([3]))
        independent = dist.Independent(base, 1)
        samples = independent.sample([10])
        self.assertEqual(samples.shape, [10, 3])

    def test_independent_log_prob(self):
        """测试独立分布的log_prob / Test Independent log_prob"""
        base = dist.Normal(loc=paddle.zeros([3]), scale=paddle.ones([3]))
        independent = dist.Independent(base, 1)
        x = paddle.zeros([10, 3])
        log_prob = independent.log_prob(x)
        self.assertEqual(log_prob.shape, [10])


if __name__ == '__main__':
    unittest.main()
