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

from config import (ATOL, DEVICES, RTOL, TEST_CASE_NAME, parameterize, place,
                    xrand)

paddle.enable_static()


@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'concentration'),
              [('test-one-dim', np.random.rand(89) + 5.0)])
class TestDirichlet(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor()
        with paddle.static.program_guard(self.program):
            conc = paddle.static.data('conc', self.concentration.shape,
                                      self.concentration.dtype)
            self._paddle_diric = paddle.distribution.Dirichlet(conc)
            self.feeds = {'conc': self.concentration}

    def test_mean(self):
        with paddle.static.program_guard(self.program):
            [out] = self.executor.run(self.program,
                                      feed=self.feeds,
                                      fetch_list=[self._paddle_diric.mean])
            np.testing.assert_allclose(
                out,
                scipy.stats.dirichlet.mean(self.concentration),
                rtol=RTOL.get(str(self.concentration.dtype)),
                atol=ATOL.get(str(self.concentration.dtype)))

    def test_variance(self):
        with paddle.static.program_guard(self.program):
            [out] = self.executor.run(self.program,
                                      feed=self.feeds,
                                      fetch_list=[self._paddle_diric.variance])
            np.testing.assert_allclose(
                out,
                scipy.stats.dirichlet.var(self.concentration),
                rtol=RTOL.get(str(self.concentration.dtype)),
                atol=ATOL.get(str(self.concentration.dtype)))

    def test_prob(self):
        with paddle.static.program_guard(self.program):
            random_number = np.random.rand(*self.concentration.shape)
            random_number = random_number / random_number.sum()
            feeds = dict(self.feeds, value=random_number)
            value = paddle.static.data('value', random_number.shape,
                                       random_number.dtype)
            out = self._paddle_diric.prob(value)
            [out] = self.executor.run(self.program,
                                      feed=feeds,
                                      fetch_list=[out])
            np.testing.assert_allclose(
                out,
                scipy.stats.dirichlet.pdf(random_number, self.concentration),
                rtol=RTOL.get(str(self.concentration.dtype)),
                atol=ATOL.get(str(self.concentration.dtype)))

    def test_log_prob(self):
        with paddle.static.program_guard(self.program):
            random_number = np.random.rand(*self.concentration.shape)
            random_number = random_number / random_number.sum()
            feeds = dict(self.feeds, value=random_number)
            value = paddle.static.data('value', random_number.shape,
                                       random_number.dtype)
            out = self._paddle_diric.log_prob(value)
            [out] = self.executor.run(self.program,
                                      feed=feeds,
                                      fetch_list=[out])
            np.testing.assert_allclose(
                out,
                scipy.stats.dirichlet.logpdf(random_number, self.concentration),
                rtol=RTOL.get(str(self.concentration.dtype)),
                atol=ATOL.get(str(self.concentration.dtype)))

    def test_entropy(self):
        with paddle.static.program_guard(self.program):
            [out] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_diric.entropy()])
            np.testing.assert_allclose(
                out,
                scipy.stats.dirichlet.entropy(self.concentration),
                rtol=RTOL.get(str(self.concentration.dtype)),
                atol=ATOL.get(str(self.concentration.dtype)))
