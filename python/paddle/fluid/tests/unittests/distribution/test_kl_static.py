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

import numbers
import unittest

import numpy as np
import paddle
import scipy.special
import scipy.stats
from paddle.distribution import kl

import config
import mock_data as mock

paddle.enable_static()


@config.place(config.DEVICES)
@config.parameterize((config.TEST_CASE_NAME, 'a1', 'b1', 'a2', 'b2'), [
    ('test_regular_input', 6.0 * np.random.random((4, 5)) + 1e-4,
     6.0 * np.random.random((4, 5)) + 1e-4, 6.0 * np.random.random(
         (4, 5)) + 1e-4, 6.0 * np.random.random((4, 5)) + 1e-4),
])
class TestKLBetaBeta(unittest.TestCase):
    def setUp(self):
        self.mp = paddle.static.Program()
        self.sp = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(self.mp, self.sp):
            a1 = paddle.static.data('a1', self.a1.shape, dtype=self.a1.dtype)
            b1 = paddle.static.data('b1', self.b1.shape, dtype=self.b1.dtype)
            a2 = paddle.static.data('a2', self.a2.shape, dtype=self.a2.dtype)
            b2 = paddle.static.data('b2', self.b2.shape, dtype=self.b2.dtype)

            self.p = paddle.distribution.Beta(a1, b1)
            self.q = paddle.distribution.Beta(a2, b2)
            self.feeds = {
                'a1': self.a1,
                'b1': self.b1,
                'a2': self.a2,
                'b2': self.b2
            }

    def test_kl_divergence(self):
        with paddle.static.program_guard(self.mp, self.sp):
            out = paddle.distribution.kl_divergence(self.p, self.q)
            self.executor.run(self.sp)
            [out] = self.executor.run(self.mp,
                                      feed=self.feeds,
                                      fetch_list=[out])

            np.testing.assert_allclose(
                out,
                self.scipy_kl_beta_beta(self.a1, self.b1, self.a2, self.b2),
                rtol=config.RTOL.get(str(self.a1.dtype)),
                atol=config.ATOL.get(str(self.a1.dtype)))

    def scipy_kl_beta_beta(self, a1, b1, a2, b2):
        return (scipy.special.betaln(a2, b2) - scipy.special.betaln(a1, b1) +
                (a1 - a2) * scipy.special.digamma(a1) +
                (b1 - b2) * scipy.special.digamma(b1) +
                (a2 - a1 + b2 - b1) * scipy.special.digamma(a1 + b1))


@config.place(config.DEVICES)
@config.parameterize((config.TEST_CASE_NAME, 'conc1', 'conc2'), [
    ('test-regular-input', np.random.random((5, 7, 8, 10)), np.random.random(
        (5, 7, 8, 10))),
])
class TestKLDirichletDirichlet(unittest.TestCase):
    def setUp(self):
        self.mp = paddle.static.Program()
        self.sp = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.mp, self.sp):
            conc1 = paddle.static.data('conc1', self.conc1.shape,
                                       self.conc1.dtype)
            conc2 = paddle.static.data('conc2', self.conc2.shape,
                                       self.conc2.dtype)
            self.p = paddle.distribution.Dirichlet(conc1)
            self.q = paddle.distribution.Dirichlet(conc2)
            self.feeds = {'conc1': self.conc1, 'conc2': self.conc2}

    def test_kl_divergence(self):

        with paddle.static.program_guard(self.mp, self.sp):
            out = paddle.distribution.kl_divergence(self.p, self.q)
            self.executor.run(self.sp)
            [out] = self.executor.run(self.mp,
                                      feed=self.feeds,
                                      fetch_list=[out])
            np.testing.assert_allclose(
                out,
                self.scipy_kl_diric_diric(self.conc1, self.conc2),
                rtol=config.RTOL.get(str(self.conc1.dtype)),
                atol=config.ATOL.get(str(self.conc1.dtype)))

    def scipy_kl_diric_diric(self, conc1, conc2):
        return (
            scipy.special.gammaln(np.sum(conc1, -1)) -
            scipy.special.gammaln(np.sum(conc2, -1)) - np.sum(
                scipy.special.gammaln(conc1) - scipy.special.gammaln(conc2), -1)
            + np.sum((conc1 - conc2) *
                     (scipy.special.digamma(conc1) -
                      scipy.special.digamma(np.sum(conc1, -1, keepdims=True))),
                     -1))


class DummyDistribution(paddle.distribution.Distribution):
    pass


@config.place(config.DEVICES)
@config.parameterize((config.TEST_CASE_NAME, 'p', 'q'),
                     [('test-dispatch-exception')])
class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.mp = paddle.static.Program()
        self.sp = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.mp, self.sp):
            self.p = DummyDistribution()
            self.q = DummyDistribution()

    def test_dispatch_with_unregister(self):
        with self.assertRaises(NotImplementedError):
            with paddle.static.program_guard(self.mp, self.sp):
                out = paddle.distribution.kl_divergence(self.p, self.q)
                self.executor.run(self.sp)
                self.executor.run(self.mp, feed={}, fetch_list=[out])


@config.place(config.DEVICES)
@config.parameterize((config.TEST_CASE_NAME, 'rate1', 'rate2'),
                     [('test-diff-dist', np.random.rand(100, 200, 100) + 1.0,
                       np.random.rand(100, 200, 100) + 2.0),
                      ('test-same-dist', np.array([1.0]), np.array([1.0]))])
class TestKLExpfamilyExpFamily(unittest.TestCase):
    def setUp(self):
        self.mp = paddle.static.Program()
        self.sp = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.mp, self.sp):
            rate1 = paddle.static.data(
                'rate1', shape=self.rate1.shape, dtype=self.rate1.dtype)
            rate2 = paddle.static.data(
                'rate2', shape=self.rate2.shape, dtype=self.rate2.dtype)
            self.p = mock.Exponential(rate1)
            self.q = mock.Exponential(rate2)
            self.feeds = {'rate1': self.rate1, 'rate2': self.rate2}

    def test_kl_expfamily_expfamily(self):
        with paddle.static.program_guard(self.mp, self.sp):
            out1 = paddle.distribution.kl_divergence(self.p, self.q)
            out2 = kl._kl_expfamily_expfamily(self.p, self.q)
            self.executor.run(self.sp)
            [out1, out2] = self.executor.run(self.mp,
                                             feed=self.feeds,
                                             fetch_list=[out1, out2])

            np.testing.assert_allclose(
                out1,
                out2,
                rtol=config.RTOL.get(config.DEFAULT_DTYPE),
                atol=config.ATOL.get(config.DEFAULT_DTYPE))
