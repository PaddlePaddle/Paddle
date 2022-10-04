# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import paddle
import scipy.stats

import config
import parameterize

from paddle.distribution.gumbel import Gumbel


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'loc', 'scale'), [
    ('one-dim', parameterize.xrand((4, )), parameterize.xrand((4, ))),
    ('multi-dim', parameterize.xrand((5, 3)), parameterize.xrand((5, 3))),
])
class TestGumbel(unittest.TestCase):

    def setUp(self):
        self._dist = Gumbel(loc=paddle.to_tensor(self.loc,dtype='float32'),
                                                 scale=paddle.to_tensor(\
                                                     self.scale,dtype='float32'))

    def test_mean(self):
        mean = self._dist.mean
        mean = np.array(mean.numpy(), dtype=np.float64)

        self.assertEqual(mean.dtype, self.scale.dtype)
        np.testing.assert_allclose(mean,
                                   self._np_mean(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_variance(self):
        var = self._dist.variance
        var = np.array(var.numpy(), dtype=np.float64)

        self.assertEqual(var.dtype, self.scale.dtype)
        np.testing.assert_allclose(var,
                                   self._np_variance(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_stddev(self):
        stddev = self._dist.stddev
        stddev = np.array(stddev.numpy(), dtype=np.float64)

        self.assertEqual(stddev.dtype, self.scale.dtype)
        np.testing.assert_allclose(stddev,
                                   self._np_stddev(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_entropy(self):
        entropy = self._dist.entropy()
        entropy = np.array(entropy.numpy(), dtype=np.float64)

        self.assertEqual(entropy.dtype, self.scale.dtype)

    def test_sample(self):

        sample_shape = [10000]
        samples = self._dist.sample(sample_shape)
        samples = np.array(samples.numpy(), dtype=np.float64)
        sample_values = samples
        self.assertEqual(samples.dtype, self.scale.dtype)

        # There is a loss of accuracy in this conversion.
        tolerance = 1e-3

        np.testing.assert_allclose(sample_values.mean(axis=0),
                                   scipy.stats.gumbel_r.mean(self.loc,
                                                             scale=self.scale),
                                   rtol=0.1,
                                   atol=tolerance)
        np.testing.assert_allclose(sample_values.var(axis=0),
                                   scipy.stats.gumbel_r.var(self.loc,
                                                            scale=self.scale),
                                   rtol=0.1,
                                   atol=tolerance)


    def test_rsample(self):

        sample_shape = [10000]
        samples = self._dist.rsample(sample_shape)
        samples = np.array(samples.numpy(), dtype=np.float64)
        sample_values = samples
        self.assertEqual(samples.dtype, self.scale.dtype)

        # There is a loss of accuracy in this conversion.
        tolerance = 1e-4

        np.testing.assert_allclose(sample_values.mean(axis=0),
                                   scipy.stats.gumbel_r.mean(self.loc,
                                                             scale=self.scale),
                                   rtol=0.1,
                                   atol=tolerance)
        np.testing.assert_allclose(sample_values.var(axis=0),
                                   scipy.stats.gumbel_r.var(self.loc,scale=self.scale),
                                   rtol=0.1,
                                   atol=tolerance)

    def _np_mean(self):
        return self.loc + self.scale * np.euler_gamma

    def _np_stddev(self):
        return np.sqrt(self._np_variance())

    def _np_variance(self):
        return np.divide(
            np.multiply(np.power(self.scale, 2), np.power(np.pi, 2)), 6)

    def _np_entropy(self):
        return np.log(self.scale) + 1 + np.euler_gamma


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'loc', 'scale', 'value'), [
        ('value-float', np.array([0.1, 0.4]),np.array([1., 4.
                                                       ]), np.array([3., 7.])),
        ('value-int', np.array([0.1, 0.4]),np.array([1, 4]), np.array([3, 7])),
        ('value-multi-dim', np.array([0.1, 0.4]), np.array(
            [1, 4]), np.array([[5., 4], [6, 2]])),
    ])
class TestGumbelPDF(unittest.TestCase):

    def setUp(self):
        self._dist = Gumbel(loc=paddle.to_tensor(self.loc),
                                                 scale=paddle.to_tensor(\
                                                     self.scale))

    def test_prob(self):
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
            scipy.stats.gumbel_r.pdf(self.value, self.loc, self.scale),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)))

    def test_log_prob(self):
        np.testing.assert_allclose(
            self._dist.log_prob(paddle.to_tensor(self.value)),
            scipy.stats.gumbel_r.logpdf(self.value, self.loc, self.scale),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)))

    def test_cdf(self):
        np.testing.assert_allclose(self._dist.cdf(paddle.to_tensor(self.value)),
                                   scipy.stats.gumbel_r.cdf(
                                       self.value, self.loc, self.scale),
                                   rtol=config.RTOL.get(str(self.loc.dtype)),
                                   atol=config.ATOL.get(str(self.loc.dtype)))


if __name__ == '__main__':
    unittest.main()
