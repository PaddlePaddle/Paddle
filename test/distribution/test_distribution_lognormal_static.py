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
from distribution import config
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand
from test_distribution_lognormal import LogNormalNumpy

import paddle
from paddle.distribution.kl import kl_divergence
from paddle.distribution.lognormal import LogNormal
from paddle.distribution.normal import Normal


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc', 'scale', 'value'),
    [
        ('one-dim', xrand((2,)), xrand((2,)), xrand((2,))),
        ('multi-dim', xrand((3, 3)), xrand((3, 3)), xrand((3, 3))),
    ],
    test_pir=True,
)
class TestLogNormal(unittest.TestCase):
    def run_program(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
            value = paddle.static.data(
                'value', self.value.shape, self.value.dtype
            )
            self.paddle_lognormal = LogNormal(loc=loc, scale=scale)
            self.np_lognormal = LogNormalNumpy(loc=self.loc, scale=self.scale)
            mean = self.paddle_lognormal.mean
            var = self.paddle_lognormal.variance
            entropy = self.paddle_lognormal.entropy()
            probs = self.paddle_lognormal.probs(value)
            log_prob = self.paddle_lognormal.log_prob(value)
        fetch_list = [mean, var, entropy, probs, log_prob]
        self.feeds = {'loc': self.loc, 'scale': self.scale, 'value': self.value}

        executor.run(startup_program)
        [
            self.mean,
            self.var,
            self.entropy,
            self.probs,
            self.log_prob,
        ] = executor.run(main_program, feed=self.feeds, fetch_list=fetch_list)

    def setUp(self):
        if self.test_pir:
            with paddle.pir_utils.IrGuard():
                self.run_program()
        else:
            self.run_program()

    def test_mean(self):
        np_mean = self.np_lognormal.mean
        self.assertEqual(str(self.mean.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(
            self.mean,
            np_mean,
            rtol=config.RTOL.get(str(self.scale.dtype)),
            atol=config.ATOL.get(str(self.scale.dtype)),
        )

    def test_var(self):
        np_var = self.np_lognormal.variance
        self.assertEqual(str(self.var.dtype).split('.')[-1], self.scale.dtype)
        np.testing.assert_allclose(
            self.var,
            np_var,
            rtol=config.RTOL.get(str(self.scale.dtype)),
            atol=config.ATOL.get(str(self.scale.dtype)),
        )

    def test_entropy(self):
        np_entropy = self.np_lognormal.entropy()
        self.assertEqual(
            str(self.entropy.dtype).split('.')[-1], self.scale.dtype
        )
        np.testing.assert_allclose(
            self.entropy,
            np_entropy,
            rtol=config.RTOL.get(str(self.scale.dtype)),
            atol=config.ATOL.get(str(self.scale.dtype)),
        )

    def test_probs(self):
        np_probs = self.np_lognormal.probs(self.value)
        np.testing.assert_allclose(
            self.probs,
            np_probs,
            rtol=config.RTOL.get(str(self.scale.dtype)),
            atol=config.ATOL.get(str(self.scale.dtype)),
        )

    def test_log_prob(self):
        np_log_prob = self.np_lognormal.log_prob(self.value)
        np.testing.assert_allclose(
            self.log_prob,
            np_log_prob,
            rtol=config.RTOL.get(str(self.scale.dtype)),
            atol=config.ATOL.get(str(self.scale.dtype)),
        )


@place(config.DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc', 'scale'),
    [('sample', xrand((4,)), xrand((4,), min=0, max=1))],
    test_pir=True,
)
class TestLogNormalSample(unittest.TestCase):
    def run_program(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc = paddle.static.data('loc', self.loc.shape, self.loc.dtype)
            scale = paddle.static.data(
                'scale', self.scale.shape, self.scale.dtype
            )
            n = 1000000
            self.sample_shape = (n,)
            self.rsample_shape = (n,)
            self.paddle_lognormal = LogNormal(loc=loc, scale=scale)
            mean = self.paddle_lognormal.mean
            variance = self.paddle_lognormal.variance
            samples = self.paddle_lognormal.sample(self.sample_shape)
            rsamples = self.paddle_lognormal.rsample(self.rsample_shape)
        fetch_list = [mean, variance, samples, rsamples]
        self.feeds = {'loc': self.loc, 'scale': self.scale}

        executor.run(startup_program)
        [self.mean, self.variance, self.samples, self.rsamples] = executor.run(
            main_program, feed=self.feeds, fetch_list=fetch_list
        )

    def setUp(self):
        if self.test_pir:
            with paddle.pir_utils.IrGuard():
                self.run_program()
        else:
            self.run_program()

    def test_sample(self):
        samples_mean = self.samples.mean(axis=0)
        samples_var = self.samples.var(axis=0)
        np.testing.assert_allclose(samples_mean, self.mean, rtol=0.1, atol=0)
        np.testing.assert_allclose(samples_var, self.variance, rtol=0.1, atol=0)

        rsamples_mean = self.rsamples.mean(axis=0)
        rsamples_var = self.rsamples.var(axis=0)
        np.testing.assert_allclose(rsamples_mean, self.mean, rtol=0.1, atol=0)
        np.testing.assert_allclose(
            rsamples_var, self.variance, rtol=0.1, atol=0
        )

        batch_shape = (self.loc + self.scale).shape
        self.assertEqual(self.samples.shape, self.sample_shape + batch_shape)
        self.assertEqual(self.rsamples.shape, self.rsample_shape + batch_shape)

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
    test_pir=True,
)
class TestLogNormalKL(unittest.TestCase):
    def run_program(self):
        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(main_program, startup_program):
            loc1 = paddle.static.data('loc1', self.loc1.shape, self.loc1.dtype)
            scale1 = paddle.static.data(
                'scale1', self.scale1.shape, self.scale1.dtype
            )
            loc2 = paddle.static.data('loc2', self.loc2.shape, self.loc2.dtype)
            scale2 = paddle.static.data(
                'scale2', self.scale2.shape, self.scale2.dtype
            )

            self.ln_a = LogNormal(loc=loc1, scale=scale1)
            self.ln_b = LogNormal(loc=loc2, scale=scale2)
            self.normal_a = Normal(loc=loc1, scale=scale1)
            self.normal_b = Normal(loc=loc2, scale=scale2)

            kl0 = self.ln_a.kl_divergence(self.ln_b)
            kl1 = kl_divergence(self.ln_a, self.ln_b)
            kl_normal = kl_divergence(self.normal_a, self.normal_b)
            kl_formula = self._kl(self.ln_a, self.ln_b)

        fetch_list = [kl0, kl1, kl_normal, kl_formula]
        self.feeds = {
            'loc1': self.loc1,
            'scale1': self.scale1,
            'loc2': self.loc2,
            'scale2': self.scale2,
        }

        executor.run(startup_program)
        [self.kl0, self.kl1, self.kl_normal, self.kl_formula] = executor.run(
            main_program, feed=self.feeds, fetch_list=fetch_list
        )

    def setUp(self):
        if self.test_pir:
            with paddle.pir_utils.IrGuard():
                self.run_program()
        else:
            self.run_program()

    def test_kl_divergence(self):
        np.testing.assert_allclose(
            self.kl0,
            self.kl_formula,
            rtol=config.RTOL.get(str(self.scale1.dtype)),
            atol=config.ATOL.get(str(self.scale1.dtype)),
        )

        np.testing.assert_allclose(
            self.kl1,
            self.kl_formula,
            rtol=config.RTOL.get(str(self.scale1.dtype)),
            atol=config.ATOL.get(str(self.scale1.dtype)),
        )

        np.testing.assert_allclose(
            self.kl_normal,
            self.kl_formula,
            rtol=config.RTOL.get(str(self.scale1.dtype)),
            atol=config.ATOL.get(str(self.scale1.dtype)),
        )

    def _kl(self, dist1, dist2):
        loc1 = dist1.loc
        loc2 = dist2.loc
        scale1 = dist1.scale
        scale2 = dist2.scale
        var_ratio = scale1 / scale2
        var_ratio = var_ratio * var_ratio
        t1 = (loc1 - loc2) / scale2
        t1 = t1 * t1
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


if __name__ == '__main__':
    unittest.main()
