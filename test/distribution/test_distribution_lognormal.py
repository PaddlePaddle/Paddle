# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math
import unittest

import numpy as np
import scipy.stats
from distribution import config
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand
from test_distribution import DistributionNumpy

import paddle
from paddle.distribution.kl import kl_divergence
from paddle.distribution.lognormal import LogNormal
from paddle.distribution.normal import Normal


class LogNormalNumpy(DistributionNumpy):
    def __init__(self, loc, scale):
        self.loc = np.array(loc)
        self.scale = np.array(scale)
        if str(self.loc.dtype) not in ['float32', 'float64']:
            self.loc = self.loc.astype('float32')
            self.scale = self.scale.astype('float32')

    @property
    def mean(self):
        var = self.scale * self.scale
        return np.exp(self.loc + var / 2)

    @property
    def variance(self):
        var = self.scale * self.scale
        return (np.exp(var) - 1) * np.exp(2 * self.loc + var)

    def log_prob(self, value):
        var = self.scale * self.scale
        log_scale = np.log(self.scale)
        return (
            -((np.log(value) - self.loc) * (np.log(value) - self.loc))
            / (2.0 * var)
            - log_scale
            - math.log(math.sqrt(2.0 * math.pi))
            - np.log(value)
        )

    def probs(self, value):
        var = self.scale * self.scale
        return np.exp(
            -1.0
            * ((np.log(value) - self.loc) * (np.log(value) - self.loc))
            / (2.0 * var)
        ) / (math.sqrt(2 * math.pi) * self.scale * value)

    def entropy(self):
        return (
            0.5
            + self.loc
            + 0.5 * np.log(np.array(2.0 * math.pi).astype(self.loc.dtype))
            + np.log(self.scale)
        )

    def kl_divergence(self, other):
        var_ratio = self.scale / other.scale
        var_ratio = var_ratio * var_ratio
        t1 = (self.loc - other.loc) / other.scale
        t1 = t1 * t1
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc', 'scale', 'value'),
    [
        ('one-dim', xrand((2,)), xrand((2,)), xrand((2,))),
        ('multi-dim', xrand((3, 3)), xrand((3, 3)), xrand((3, 3))),
    ],
)
class LogNormalTest(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.paddle_lognormal = LogNormal(
            loc=paddle.to_tensor(self.loc), scale=paddle.to_tensor(self.scale)
        )
        self.np_lognormal = LogNormalNumpy(self.loc, self.scale)

    def test_mean(self):
        mean = self.paddle_lognormal.mean
        np_mean = self.np_lognormal.mean
        self.assertEqual(mean.numpy().dtype, np_mean.dtype)
        np.testing.assert_allclose(
            mean,
            np_mean,
            rtol=config.RTOL.get(str(self.scale.dtype)),
            atol=config.ATOL.get(str(self.scale.dtype)),
        )

    def test_variance(self):
        var = self.paddle_lognormal.variance
        np_var = self.np_lognormal.variance
        self.assertEqual(var.numpy().dtype, np_var.dtype)
        np.testing.assert_allclose(
            var,
            np_var,
            rtol=config.RTOL.get(str(self.scale.dtype)),
            atol=config.ATOL.get(str(self.scale.dtype)),
        )

    def test_entropy(self):
        entropy = self.paddle_lognormal.entropy()
        np_entropy = self.np_lognormal.entropy()
        self.assertEqual(entropy.numpy().dtype, np_entropy.dtype)
        np.testing.assert_allclose(
            entropy,
            np_entropy,
            rtol=config.RTOL.get(str(self.scale.dtype)),
            atol=config.ATOL.get(str(self.scale.dtype)),
        )

    def test_probs(self):
        with paddle.base.dygraph.guard(self.place):
            probs = self.paddle_lognormal.probs(paddle.to_tensor(self.value))
            np_probs = self.np_lognormal.probs(self.value)
            np.testing.assert_allclose(
                probs,
                np_probs,
                rtol=config.RTOL.get(str(self.scale.dtype)),
                atol=config.ATOL.get(str(self.scale.dtype)),
            )

    def test_log_prob(self):
        with paddle.base.dygraph.guard(self.place):
            log_prob = self.paddle_lognormal.log_prob(
                paddle.to_tensor(self.value)
            )
            np_log_prob = self.np_lognormal.log_prob(self.value)
            np.testing.assert_allclose(
                log_prob,
                np_log_prob,
                rtol=config.RTOL.get(str(self.scale.dtype)),
                atol=config.ATOL.get(str(self.scale.dtype)),
            )


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc', 'scale'),
    [('sample', xrand((4,)), xrand((4,), min=0, max=1))],
)
class TestLogNormalSample(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.paddle_lognormal = LogNormal(loc=self.loc, scale=self.scale)
        n = 1000000
        self.sample_shape = (n,)
        self.rsample_shape = (n,)
        self.samples = self.paddle_lognormal.sample(self.sample_shape)
        self.rsamples = self.paddle_lognormal.rsample(self.rsample_shape)

    def test_sample(self):
        samples_mean = self.samples.mean(axis=0)
        samples_var = self.samples.var(axis=0)
        np.testing.assert_allclose(
            samples_mean, self.paddle_lognormal.mean, rtol=0.1, atol=0
        )
        np.testing.assert_allclose(
            samples_var, self.paddle_lognormal.variance, rtol=0.1, atol=0
        )

        rsamples_mean = self.rsamples.mean(axis=0)
        rsamples_var = self.rsamples.var(axis=0)
        np.testing.assert_allclose(
            rsamples_mean, self.paddle_lognormal.mean, rtol=0.1, atol=0
        )
        np.testing.assert_allclose(
            rsamples_var, self.paddle_lognormal.variance, rtol=0.1, atol=0
        )

        batch_shape = (self.loc + self.scale).shape
        self.assertEqual(
            self.samples.shape, list(self.sample_shape + batch_shape)
        )
        self.assertEqual(
            self.rsamples.shape, list(self.rsample_shape + batch_shape)
        )

        for i in range(len(self.scale)):
            self.assertTrue(
                self._kstest(self.loc[i], self.scale[i], self.samples[:, i])
            )
            self.assertTrue(
                self._kstest(self.loc[i], self.scale[i], self.rsamples[:, i])
            )

    def _kstest(self, loc, scale, samples):
        # Uses the Kolmogorov-Smirnov test for goodness of fit.
        ks, _ = scipy.stats.kstest(
            samples, scipy.stats.lognorm(s=scale, scale=np.exp(loc)).cdf
        )
        return ks < 0.02


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc1', 'scale1', 'loc2', 'scale2'),
    [
        ('one-dim', xrand((2,)), xrand((2,)), xrand((2,)), xrand((2,))),
        (
            'multi-dim',
            xrand((2, 2)),
            xrand((2, 2)),
            xrand((2, 2)),
            xrand((2, 2)),
        ),
    ],
)
class TestLogNormalKL(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.ln_a = LogNormal(
            loc=paddle.to_tensor(self.loc1), scale=paddle.to_tensor(self.scale1)
        )
        self.ln_b = LogNormal(
            loc=paddle.to_tensor(self.loc2), scale=paddle.to_tensor(self.scale2)
        )
        self.normal_a = Normal(
            loc=paddle.to_tensor(self.loc1), scale=paddle.to_tensor(self.scale1)
        )
        self.normal_b = Normal(
            loc=paddle.to_tensor(self.loc2), scale=paddle.to_tensor(self.scale2)
        )

    def test_kl_divergence(self):
        kl0 = self.ln_a.kl_divergence(self.ln_b)
        kl1 = kl_divergence(self.ln_a, self.ln_b)
        kl_normal = kl_divergence(self.normal_a, self.normal_b)
        kl_formula = self._kl(self.ln_a, self.ln_b)

        self.assertEqual(tuple(kl0.shape), self.scale1.shape)
        self.assertEqual(tuple(kl1.shape), self.scale1.shape)
        np.testing.assert_allclose(
            kl0,
            kl_formula,
            rtol=config.RTOL.get(str(self.scale1.dtype)),
            atol=config.ATOL.get(str(self.scale1.dtype)),
        )
        np.testing.assert_allclose(
            kl1,
            kl_formula,
            rtol=config.RTOL.get(str(self.scale1.dtype)),
            atol=config.ATOL.get(str(self.scale1.dtype)),
        )
        np.testing.assert_allclose(
            kl_normal,
            kl_formula,
            rtol=config.RTOL.get(str(self.scale1.dtype)),
            atol=config.ATOL.get(str(self.scale1.dtype)),
        )

    def _kl(self, dist1, dist2):
        loc1 = np.array(dist1.loc)
        loc2 = np.array(dist2.loc)
        scale1 = np.array(dist1.scale)
        scale2 = np.array(dist2.scale)
        var_ratio = scale1 / scale2
        var_ratio = var_ratio * var_ratio
        t1 = (loc1 - loc2) / scale2
        t1 = t1 * t1
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


if __name__ == '__main__':
    unittest.main()
