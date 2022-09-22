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
import scipy.stats

import numpy as np
import paddle

import config
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand
from paddle.distribution import LogNormal
from test_distribution import DistributionNumpy
from paddle.distribution.kl import kl_divergence

np.random.seed(2022)


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
        return -(
            (np.log(value) - self.loc) *
            (np.log(value) - self.loc)) / (2. * var) - log_scale - math.log(
                math.sqrt(2. * math.pi)) - np.log(value)

    def probs(self, value):
        var = self.scale * self.scale
        return np.exp(
            -1. * ((np.log(value) - self.loc) * (np.log(value) - self.loc)) /
            (2. * var)) / (math.sqrt(2 * math.pi) * self.scale * value)

    def entropy(self):
        return 0.5 + self.loc + 0.5 * np.log(
            np.array(2. * math.pi).astype(self.loc.dtype)) + np.log(self.scale)

    def kl_divergence(self, other):
        var_ratio = (self.scale / other.scale)
        var_ratio = var_ratio * var_ratio
        t1 = ((self.loc - other.loc) / other.scale)
        t1 = (t1 * t1)
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


@place(config.DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'loc', 'scale'),
                  [('float', xrand(), xrand()),
                   ('one-dim', xrand((3, )), xrand((3, ))),
                   ('multi-dim', xrand((5, 5)), xrand((5, 5)))])
class LogNormalTest(unittest.TestCase):

    def setUp(self):
        self._paddle_lognormal = LogNormal(loc=paddle.to_tensor(self.loc),
                                           scale=paddle.to_tensor(self.scale))
        self._np_lognormal = LogNormalNumpy(self.loc, self.scale)

    def test_mean(self):
        mean = self._paddle_lognormal.mean
        np_mean = self._np_lognormal.mean
        self.assertEqual(mean.numpy().dtype, np_mean.dtype)
        np.testing.assert_allclose(mean,
                                   np_mean,
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_variance(self):
        var = self._paddle_lognormal.variance
        np_var = self._np_lognormal.variance
        self.assertEqual(var.numpy().dtype, np_var.dtype)
        np.testing.assert_allclose(var,
                                   np_var,
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_entropy(self):
        entropy = self._paddle_lognormal.entropy()
        np_entropy = self._np_lognormal.entropy()
        self.assertEqual(entropy.numpy().dtype, np_entropy.dtype)
        np.testing.assert_allclose(entropy,
                                   np_entropy,
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_probs(self):
        value = [np.random.rand(*self.scale.shape)]

        for v in value:
            with paddle.fluid.dygraph.guard(self.place):
                probs = self._paddle_lognormal.probs(paddle.to_tensor(v))
                np_probs = self._np_lognormal.probs(v)
                np.testing.assert_allclose(
                    probs,
                    np_probs,
                    rtol=config.RTOL.get(str(self.scale.dtype)),
                    atol=config.ATOL.get(str(self.scale.dtype)))

    def test_log_prob(self):
        value = [np.random.rand(*self.scale.shape)]
        for v in value:
            with paddle.fluid.dygraph.guard(self.place):
                log_prob = self._paddle_lognormal.log_prob(paddle.to_tensor(v))
                np_log_prob = self._np_lognormal.log_prob(v)
                np.testing.assert_allclose(
                    log_prob,
                    np_log_prob,
                    rtol=config.RTOL.get(str(self.scale.dtype)),
                    atol=config.ATOL.get(str(self.scale.dtype)))


@place(config.DEVICES)
@parameterize_cls((TEST_CASE_NAME, 'loc', 'scale'), [('sample1', xrand(
    (2, )), xrand((2, ))), ('sample2', xrand((5, )), xrand((5, )))])
class LogNormalTestSample(unittest.TestCase):

    def test_sample(self):
        self._paddle_lognormal = LogNormal(loc=self.loc, scale=self.scale)
        shape = [8000]
        samples = self._paddle_lognormal.sample(shape)
        for i in range(len(self.scale)):
            self.assertTrue(
                self._kstest(self.loc[i], self.scale[i], samples[:, i]))

    def _kstest(self, loc, scale, samples):
        # Uses the Kolmogorov-Smirnov test for goodness of fit.
        ks, _ = scipy.stats.kstest(
            samples,
            scipy.stats.lognorm(s=scale, scale=np.exp(loc)).cdf)
        return ks < 0.02


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc1', 'scale1', 'loc2', 'scale2'),
    [('one-dim', xrand((2, )), xrand((2, )), xrand((2, )), xrand((2, ))),
     ('multi-dim', xrand((2, 2)), xrand((2, 2)), xrand((2, 2)), xrand((2, 2)))])
class TestLognormalKL(unittest.TestCase):

    def setUp(self):
        self._paddle_lognormal = LogNormal(loc=paddle.to_tensor(self.loc1),
                                           scale=paddle.to_tensor(self.scale1))
        self._paddle_lognormal_other = LogNormal(
            loc=paddle.to_tensor(self.loc2),
            scale=paddle.to_tensor(self.scale2))

    def test_kl_divergence(self):
        kl1 = kl_divergence(self._paddle_lognormal,
                            self._paddle_lognormal_other)
        kl2 = self._kl(self._paddle_lognormal, self._paddle_lognormal_other)
        np.testing.assert_allclose(kl1,
                                   kl2,
                                   rtol=config.RTOL.get(str(self.scale1.dtype)),
                                   atol=config.ATOL.get(str(self.scale1.dtype)))

    def _kl(self, dist1, dist2):
        loc1 = np.array(dist1.loc)
        loc2 = np.array(dist2.loc)
        scale1 = np.array(dist1.scale)
        scale2 = np.array(dist2.scale)
        var_ratio = (scale1 / scale2)
        var_ratio = var_ratio * var_ratio
        t1 = ((loc1 - loc2) / scale2)
        t1 = (t1 * t1)
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


if __name__ == '__main__':
    unittest.main()
