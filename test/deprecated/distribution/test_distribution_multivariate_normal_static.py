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

import paddle
from paddle.distribution.multivariate_normal import MultivariateNormal

paddle.enable_static()


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
            np.array([[6.0, 2.5, 3.0], [2.5, 4.0, 5.0], [3.0, 5.0, 7.0]]),
        ),
    ],
)
class TestMVN(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            covariance_matrix = paddle.static.data(
                'covariance_matrix',
                self.covariance_matrix.shape,
                self.covariance_matrix.dtype,
            )
            dist = MultivariateNormal(
                loc=loc, covariance_matrix=covariance_matrix
            )
            mean = dist.mean
            var = dist.variance
            entropy = dist.entropy()
            mini_samples = dist.sample(shape=())
            large_samples = dist.sample(shape=(10000,))
        fetch_list = [mean, var, entropy, mini_samples, large_samples]
        feed = {'loc': self.loc, 'covariance_matrix': self.covariance_matrix}

        executor.run(startup_program)
        [
            self.mean,
            self.var,
            self.entropy,
            self.mini_samples,
            self.large_samples,
        ] = executor.run(main_program, feed=feed, fetch_list=fetch_list)

    def test_mean(self):
        self.assertEqual(str(self.mean.dtype).split('.')[-1], self.loc.dtype)
        np.testing.assert_allclose(
            self.mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_variance(self):
        self.assertEqual(str(self.var.dtype).split('.')[-1], self.loc.dtype)
        np.testing.assert_allclose(
            self.var,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_entropy(self):
        self.assertEqual(str(self.entropy.dtype).split('.')[-1], self.loc.dtype)
        np.testing.assert_allclose(
            self.entropy,
            self._np_entropy(),
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )

    def test_sample(self):
        self.assertEqual(
            str(self.mini_samples.dtype).split('.')[-1], self.loc.dtype
        )
        sample_mean = self.large_samples.mean(axis=0)
        sample_variance = self.large_samples.var(axis=0)

        # `atol` and `rtol` refer to ``test_distribution_normal`` and ``test_distribution_lognormal``
        np.testing.assert_allclose(sample_mean, self.mean, atol=0, rtol=0.1)
        np.testing.assert_allclose(sample_variance, self.var, atol=0, rtol=0.1)

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
    (parameterize.TEST_CASE_NAME, 'loc', 'covariance_matrix', 'value'),
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
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            covariance_matrix = paddle.static.data(
                'covariance_matrix',
                self.covariance_matrix.shape,
                self.covariance_matrix.dtype,
            )
            value = paddle.static.data(
                'value', self.value.shape, self.value.dtype
            )
            dist = MultivariateNormal(
                loc=loc, covariance_matrix=covariance_matrix
            )
            pmf = dist.prob(value)
        feed = {
            'loc': self.loc,
            'covariance_matrix': self.covariance_matrix,
            'value': self.value,
        }
        fetch_list = [pmf]

        executor.run(startup_program)
        [self.pmf] = executor.run(
            main_program, feed=feed, fetch_list=fetch_list
        )

    def test_prob(self):
        if len(self.value.shape) <= 1:
            scipy_pdf = scipy.stats.multivariate_normal.pdf(
                self.value, self.loc, self.covariance_matrix
            )
        else:
            scipy_pdf = np.apply_along_axis(
                lambda i: scipy.stats.multivariate_normal.pdf(
                    i, self.loc, self.covariance_matrix
                ),
                axis=1,
                arr=self.value,
            )
        np.testing.assert_allclose(
            self.pmf,
            scipy_pdf,
            rtol=config.RTOL.get(str(self.loc.dtype)),
            atol=config.ATOL.get(str(self.loc.dtype)),
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'mu_1', 'sig_1', 'mu_2', 'sig_2'),
    [
        (
            'one-batch',
            parameterize.xrand((2,), dtype='float32', min=-2, max=2),
            np.array([[2.0, 1.0], [1.0, 2.0]]).astype('float32'),
            parameterize.xrand((2,), dtype='float32', min=-2, max=2),
            np.array([[3.0, 2.0], [2.0, 3.0]]).astype('float32'),
        )
    ],
)
class TestMVNKL(unittest.TestCase):
    def setUp(self):
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(main_program, startup_program):
            mu_1 = paddle.static.data('mu_1', self.mu_1.shape)
            sig_1 = paddle.static.data('sig_1', self.sig_1.shape)
            mu_2 = paddle.static.data('mu_2', self.mu_2.shape)
            sig_2 = paddle.static.data('sig_2', self.sig_2.shape)
            dist1 = MultivariateNormal(loc=mu_1, covariance_matrix=sig_1)
            dist2 = MultivariateNormal(loc=mu_2, covariance_matrix=sig_2)
            kl_dist1_dist2 = dist1.kl_divergence(dist2)
        feed = {
            'mu_1': self.mu_1,
            'sig_1': self.sig_1,
            'mu_2': self.mu_2,
            'sig_2': self.sig_2,
        }
        fetch_list = [kl_dist1_dist2]

        executor.run(startup_program)
        [self.kl_dist1_dist2] = executor.run(
            main_program, feed=feed, fetch_list=fetch_list
        )

    def test_kl_divergence(self):
        kl0 = self.kl_dist1_dist2
        kl1 = self.kl_divergence()
        batch_shape = np.broadcast_shapes(
            self.sig_1.shape[:-2], self.mu_1.shape[:-1]
        )
        self.assertEqual(tuple(kl0.shape), batch_shape)
        self.assertEqual(tuple(kl1.shape), batch_shape)
        np.testing.assert_allclose(kl0, kl1, rtol=0.1, atol=0.1)

    def kl_divergence(self):
        t1 = np.array(np.linalg.cholesky(self.sig_1))
        t2 = np.array(np.linalg.cholesky(self.sig_2))
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


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=3, exit=False)
