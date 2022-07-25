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


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'loc', 'scale'), [
    ('one-dim', parameterize.xrand((2, )),\
             parameterize.xrand((2, ))),
    ('multi-dim', parameterize.xrand((10, 20)),\
             parameterize.xrand((10, 20))),
    ])
class TestLaplace(unittest.TestCase):

    def setUp(self):
        self._dist = paddle.distribution.Laplace(loc=paddle.to_tensor(self.loc),
                                                 scale=paddle.to_tensor(\
                                                     self.scale))

    def test_mean(self):
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.scale.dtype)
        np.testing.assert_allclose(mean,
                                   self._np_mean(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_variance(self):
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.scale.dtype)
        np.testing.assert_allclose(var,
                                   self._np_variance(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_stddev(self):
        stddev = self._dist.stddev
        self.assertEqual(stddev.numpy().dtype, self.scale.dtype)
        np.testing.assert_allclose(stddev,
                                   self._np_stddev(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_entropy(self):
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.scale.dtype)

    def test_sample(self):
        sample_shape = ()
        samples = self._dist.sample(sample_shape)
        self.assertEqual(samples.numpy().dtype, self.scale.dtype)
        self.assertEqual(tuple(samples.shape),
                         tuple(self._dist._extend_shape(sample_shape)))

        sample_shape = (20000, )
        samples = self._dist.sample(sample_shape)
        sample_mean = samples.mean(axis=0)
        np.testing.assert_allclose(sample_mean,
                                   self._dist.mean,
                                   atol=0,
                                   rtol=0.30)

    def _np_mean(self):
        return self.loc

    def _np_stddev(self):
        return (2**0.5) * self.scale

    def _np_variance(self):
        stddev = (2**0.5) * self.scale
        return np.power(stddev, 2)

    def _np_entropy(self):
        return scipy.stats.laplace.entropy(loc=self.loc, scale=self.scale)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'loc', 'scale', 'value'),
    [
        ('value-float', np.array([0.2, 0.3]),\
             np.array([2, 3]), np.array([2., 5.])),
        ('value-int', np.array([0.2, 0.3]),\
             np.array([2, 3]), np.array([2, 5])),
        ('value-multi-dim', np.array([0.2, 0.3]), np.array([2, 3]),\
                         np.array([[4., 6], [8, 2]])),
    ])
class TestMultinomialPdf(unittest.TestCase):

    def setUp(self):
        self._dist = paddle.distribution.Laplace(loc=paddle.to_tensor(self.loc),
                                                 scale=paddle.to_tensor(\
                                                     self.scale))

    def test_prob(self):
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
            scipy.stats.laplace.pdf(self.value, self.loc, self.scale),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)))

    def test_log_prob(self):
        np.testing.assert_allclose(
            self._dist.log_prob(paddle.to_tensor(self.value)),
            scipy.stats.laplace.logpdf(self.value, self.loc, self.scale),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)))

    def test_cdf(self):
        np.testing.assert_allclose(self._dist.cdf(paddle.to_tensor(self.value)),
                                   scipy.stats.laplace.cdf(
                                       self.value, self.loc, self.scale),
                                   rtol=config.RTOL.get(str(self.loc.dtype)),
                                   atol=config.ATOL.get(str(self.loc.dtype)))

    def test_icdf(self):
        np.testing.assert_allclose(
            self._dist.icdf(paddle.to_tensor(self.value)),
            scipy.stats.laplace.ppf(self.value, self.loc, self.scale),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)))


if __name__ == '__main__':
    unittest.main()
