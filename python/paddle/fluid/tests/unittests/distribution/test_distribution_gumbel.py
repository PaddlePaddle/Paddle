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

SEED=2022

@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'loc', 'scale'), [
    ('one-dim', parameterize.xrand((2, )),\
             parameterize.xrand((2, ))),
    ('multi-dim', parameterize.xrand((5, 5)),\
             parameterize.xrand((5, 5))),
    ])
class TestGumbel(unittest.TestCase):

    def setUp(self):
        self._dist = paddle.distribution.Gumbel(loc=paddle.to_tensor(self.loc),
                                                 scale=paddle.to_tensor(\
                                                     self.scale))

    def test_mean(self):
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.scale.dtype)
        np.testing.assert_allclose(mean,
                                   self.np_mean(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_variance(self):
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.scale.dtype)
        np.testing.assert_allclose(var,
                                   self.np_variance(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_stddev(self):
        stddev = self._dist.stddev
        self.assertEqual(stddev.numpy().dtype, self.scale.dtype)
        np.testing.assert_allclose(stddev,
                                   self.np_stddev(),
                                   rtol=config.RTOL.get(str(self.scale.dtype)),
                                   atol=config.ATOL.get(str(self.scale.dtype)))

    def test_entropy(self):
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.scale.dtype)

    def test_sample(self):

        sample_shape = (30000, )
        samples = self._dist.sample(sample_shape, seed=SEED)
        sample_values = samples.numpy()

        self.assertEqual(samples.numpy().dtype, self.scale.dtype)
        self.assertEqual(tuple(samples.shape),
                         tuple(self._dist._extend_shape(sample_shape)))

        self.assertEqual(samples.shape, list(sample_shape + self.loc.shape))
        self.assertEqual(sample_values.shape, sample_shape + self.loc.shape)

        np.testing.assert_allclose(sample_values.mean(axis=0),
                                   scipy.stats.gumbel.mean(self.loc,
                                                            scale=self.scale),
                                   rtol=0.1,
                                   atol=0.)
        np.testing.assert_allclose(sample_values.var(axis=0),
                                   scipy.stats.gumbel.var(self.loc,
                                                           scale=self.scale),
                                   rtol=0.1,
                                   atol=0.)

        tmp_loc = 4.0
        tmp_scale = 3.0
        tmp_dist = paddle.distribution.Gumbel(loc=tmp_loc, scale=tmp_scale)
        samples = tmp_dist.sample(sample_shape, seed=SEED)
        sample_values = samples.numpy()
        self.assertTrue(self.kstest(tmp_loc, tmp_scale, sample_values))

    def np_mean(self):
        return self.loc

    def np_stddev(self):
        return (2 ** 0.5) * self.scale

    def np_variance(self):
        stddev = (2 ** 0.5) * self.scale
        return np.power(stddev, 2)

    def np_entropy(self):
        return scipy.stats.laplace.entropy(loc=self.loc, scale=self.scale)

    def kstest(self, loc, scale, samples):
        # Uses the Kolmogorov-Smirnov test for goodness of fit.
        ks, p_value = scipy.stats.kstest(
            samples,
            scipy.stats.gumbel(loc, scale=scale).cdf)
        return p_value < 0.05

if __name__ == '__main__':
    unittest.main()
