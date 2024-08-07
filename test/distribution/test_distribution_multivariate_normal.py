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
import parameterize
import scipy
from distribution import config
from parameterize import (
    TEST_CASE_NAME,
    parameterize_cls,
)

import paddle
from paddle.distribution.multivariate_normal import MultivariateNormal


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'loc', 'covariance_matrix'),
    [
        (
            'one-batch',
            parameterize.xrand((2,), dtype='float32', min=1, max=2),
            np.array([[2.0, 1.0], [1.0, 2.0]]),
        ),
        (
            'multi-batch',
            parameterize.xrand((2, 3), dtype='float64', min=-2, max=-1),
            np.array([[4.0, 2.5, 2.0], [2.5, 3.0, 1.2], [2.0, 1.2, 4.0]]),
        ),
    ],
)
class TestMVN(unittest.TestCase):
    def setUp(self):
        self._dist = MultivariateNormal(
            loc=paddle.to_tensor(self.loc),
            covariance_matrix=paddle.to_tensor(self.covariance_matrix),
        )

    def test_mean(self):
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.loc.dtype)
        np.testing.assert_allclose(
            mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_variance(self):
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.loc.dtype)
        np.testing.assert_allclose(
            var,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_entropy(self):
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.loc.dtype)
        np.testing.assert_allclose(
            entropy,
            self._np_entropy(),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_sample(self):
        sample_shape = ()
        samples = self._dist.sample(sample_shape)
        self.assertEqual(samples.numpy().dtype, self.loc.dtype)
        self.assertEqual(
            tuple(samples.shape),
            sample_shape + self._dist.batch_shape + self._dist.event_shape,
        )

        sample_shape = (50000,)
        samples = self._dist.sample(sample_shape)
        sample_mean = samples.mean(axis=0)
        sample_variance = samples.var(axis=0)

        # `atol` and `rtol` refer to ``test_distribution_normal`` and ``test_distribution_lognormal``
        np.testing.assert_allclose(
            sample_mean, self._dist.mean, atol=0.0, rtol=0.1
        )
        np.testing.assert_allclose(
            sample_variance, self._dist.variance, atol=0.0, rtol=0.1
        )

    def _np_variance(self):
        batch_shape = np.broadcast_shapes(
            self.covariance_matrix.shape[:-2], self.loc.shape[:-1]
        )
        event_shape = self.loc.shape[-1:]
        return np.broadcast_to(
            np.diag(self.covariance_matrix), batch_shape + event_shape
        )

    def _np_mean(self):
        return self.loc

    def _np_entropy(self):
        if len(self.loc.shape) <= 1:
            return scipy.stats.multivariate_normal.entropy(
                self.loc, self.covariance_matrix
            )
        else:
            return np.apply_along_axis(
                lambda i: scipy.stats.multivariate_normal.entropy(
                    i, self.covariance_matrix
                ),
                axis=1,
                arr=self.loc,
            )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'loc', 'precision_matrix', 'value'),
    [
        (
            'value-same-shape',
            parameterize.xrand((2,), dtype='float32', min=-2, max=2),
            np.array([[2.0, 1.0], [1.0, 2.0]]),
            parameterize.xrand((2,), dtype='float32', min=-5, max=5),
        ),
        (
            'value-broadcast-shape',
            parameterize.xrand((2,), dtype='float64', min=-2, max=2),
            np.array([[2.0, 1.0], [1.0, 2.0]]),
            parameterize.xrand((3, 2), dtype='float64', min=-5, max=5),
        ),
    ],
)
class TestMVNProbs(unittest.TestCase):
    def setUp(self):
        self._dist = MultivariateNormal(
            loc=paddle.to_tensor(self.loc),
            precision_matrix=paddle.to_tensor(self.precision_matrix),
        )
        self.cov = np.linalg.inv(self.precision_matrix)

    def test_prob(self):
        if len(self.value.shape) <= 1:
            scipy_pdf = scipy.stats.multivariate_normal.pdf(
                self.value, self.loc, self.cov
            )
        else:
            scipy_pdf = np.apply_along_axis(
                lambda i: scipy.stats.multivariate_normal.pdf(
                    i, self.loc, self.cov
                ),
                axis=1,
                arr=self.value,
            )
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
            scipy_pdf,
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_log_prob(self):
        if len(self.value.shape) <= 1:
            scipy_logpdf = scipy.stats.multivariate_normal.logpdf(
                self.value, self.loc, self.cov
            )
        else:
            scipy_logpdf = np.apply_along_axis(
                lambda i: scipy.stats.multivariate_normal.logpdf(
                    i, self.loc, self.cov
                ),
                axis=1,
                arr=self.value,
            )
        np.testing.assert_allclose(
            self._dist.log_prob(paddle.to_tensor(self.value)),
            scipy_logpdf,
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'mu_1', 'tril_1', 'mu_2', 'tril_2'),
    [
        (
            'one-batch',
            parameterize.xrand((2,), dtype='float32', min=-2, max=2),
            np.array([[2.0, 0.0], [1.0, 2.0]]),
            parameterize.xrand((2,), dtype='float32', min=-2, max=2),
            np.array([[3.0, 0.0], [2.0, 3.0]]),
        )
    ],
)
class TestMVNKL(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self._dist1 = MultivariateNormal(
            loc=paddle.to_tensor(self.mu_1),
            scale_tril=paddle.to_tensor(self.tril_1),
        )
        self._dist2 = MultivariateNormal(
            loc=paddle.to_tensor(self.mu_2),
            scale_tril=paddle.to_tensor(self.tril_2),
        )

    def test_kl_divergence(self):
        kl0 = self._dist1.kl_divergence(self._dist2)
        kl1 = self.kl_divergence(self._dist1, self._dist2)

        self.assertEqual(tuple(kl0.shape), self._dist1.batch_shape)
        self.assertEqual(tuple(kl1.shape), self._dist1.batch_shape)
        np.testing.assert_allclose(
            kl0,
            kl1,
            rtol=config.RTOL.get(str(self.mu_1.dtype)),
            atol=config.ATOL.get(str(self.mu_1.dtype)),
        )

    def kl_divergence(self, dist1, dist2):
        t1 = np.array(dist1._unbroadcasted_scale_tril)
        t2 = np.array(dist2._unbroadcasted_scale_tril)
        half_log_det_1 = np.log(t1.diagonal(axis1=-2, axis2=-1)).sum(-1)
        half_log_det_2 = np.log(t2.diagonal(axis1=-2, axis2=-1)).sum(-1)
        new_perm = list(range(len(t1.shape)))
        new_perm[-1], new_perm[-2] = new_perm[-2], new_perm[-1]
        cov_mat_1 = np.matmul(t1, t1.transpose(new_perm))
        cov_mat_2 = np.matmul(t2, t2.transpose(new_perm))
        expectation = (
            np.linalg.solve(cov_mat_2, cov_mat_1)
            .diagonal(axis1=-2, axis2=-1)
            .sum(-1)
        )
        tmp = np.linalg.solve(t2, self.mu_1 - self.mu_2)
        expectation += np.matmul(tmp.T, tmp)
        return half_log_det_2 - half_log_det_1 + 0.5 * (expectation - 2.0)


@parameterize.place(config.DEVICES)
@parameterize_cls([TEST_CASE_NAME], ['MVNTestError'])
class MVNTestError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(self.place)


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=3, exit=False)
