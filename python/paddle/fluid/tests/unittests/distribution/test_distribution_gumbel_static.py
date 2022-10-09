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

import unittest

import numpy as np
import scipy.stats

import paddle
import config
import parameterize

from paddle.distribution.gumbel import Gumbel

paddle.enable_static()


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls((parameterize.TEST_CASE_NAME, 'loc', 'scale'), [
    ('one-dim', parameterize.xrand((4, )), parameterize.xrand((4, ))),
    ('multi-dim', parameterize.xrand((5, 3)), parameterize.xrand((5, 3))),
])
class TestGumbel(unittest.TestCase):

    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data('scale', self.scale.shape,
                                       self.scale.dtype)
            self._dist = Gumbel(loc=loc, scale=scale)
            self.sample_shape = (50000, )
            mean = self._dist.mean
            var = self._dist.variance
            stddev = self._dist.stddev
            entropy = self._dist.entropy()
            samples = self._dist.sample(self.sample_shape)
        fetch_list = [mean, var, stddev, entropy, samples]
        self.feeds = {'loc': self.loc, 'scale': self.scale}

        executor.run(startup_program)
        [self.mean, self.var, self.stddev, self.entropy,
         self.samples] = executor.run(main_program,
                                      feed=self.feeds,
                                      fetch_list=fetch_list)

    def test_mean(self):
        self.assertEqual(str(self.mean.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(self.mean,
                                   self._np_mean(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_variance(self):
        self.assertEqual(str(self.var.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(self.var,
                                   self._np_variance(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_stddev(self):
        self.assertEqual(
            str(self.stddev.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(self.stddev,
                                   self._np_stddev(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_entropy(self):
        self.assertEqual(
            str(self.entropy.dtype).split('.')[-1], self.scale.dtype)

    def test_sample(self):
        self.assertEqual(self.samples.dtype, self.scale.dtype)
        self.assertEqual(tuple(self.samples.shape),
                         tuple(self._dist._extend_shape(self.sample_shape)))

        self.assertEqual(self.samples.shape, self.sample_shape + self.loc.shape)
        self.assertEqual(self.samples.shape, self.sample_shape + self.loc.shape)

        tolerance = 1e-3
        np.testing.assert_allclose(self.samples.mean(axis=0),
                                   scipy.stats.gumbel_r.mean(self.loc,
                                                             scale=self.scale),
                                   rtol=0.1,
                                   atol=tolerance)
        np.testing.assert_allclose(self.samples.var(axis=0),
                                   scipy.stats.gumbel_r.var(self.loc,
                                                            scale=self.scale),
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
        ('value-float', np.array([0.1, 0.4]), np.array([1., 4.
                                                        ]), np.array([3., 7.])),
        ('value-int', np.array([0.1, 0.4]), np.array([1, 4]), np.array([3, 7])),
        ('value-multi-dim', np.array([0.1, 0.4]), np.array(
            [1, 4]), np.array([[5., 4], [6, 2]])),
    ])
class TestGumbelPDF(unittest.TestCase):

    def setUp(self):
        self._dist = Gumbel(loc=paddle.to_tensor(self.loc),
                            scale=paddle.to_tensor(self.scale))

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
