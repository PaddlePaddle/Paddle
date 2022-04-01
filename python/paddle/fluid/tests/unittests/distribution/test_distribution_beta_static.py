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
import scipy.stats

import config
import parameterize as param
from config import ATOL, RTOL
from parameterize import xrand

paddle.enable_static()


@param.place(config.DEVICES)
@param.parameterize_cls(
    (param.TEST_CASE_NAME, 'alpha', 'beta'), [('test-tensor', xrand(
        (10, 10)), xrand((10, 10))), ('test-broadcast', xrand((2, 1)), xrand(
            (2, 5))), ('test-larger-data', xrand((10, 20)), xrand((10, 20)))])
class TestBeta(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            # scale no need convert to tensor for scale input unittest
            alpha = paddle.static.data('alpha', self.alpha.shape,
                                       self.alpha.dtype)
            beta = paddle.static.data('beta', self.beta.shape, self.beta.dtype)
            self._paddle_beta = paddle.distribution.Beta(alpha, beta)
            self.feeds = {'alpha': self.alpha, 'beta': self.beta}

    def test_mean(self):
        with paddle.static.program_guard(self.program):
            [mean] = self.executor.run(self.program,
                                       feed=self.feeds,
                                       fetch_list=[self._paddle_beta.mean])
            np.testing.assert_allclose(
                mean,
                scipy.stats.beta.mean(self.alpha, self.beta),
                rtol=RTOL.get(str(self.alpha.dtype)),
                atol=ATOL.get(str(self.alpha.dtype)))

    def test_variance(self):
        with paddle.static.program_guard(self.program):
            [variance] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_beta.variance])
            np.testing.assert_allclose(
                variance,
                scipy.stats.beta.var(self.alpha, self.beta),
                rtol=RTOL.get(str(self.alpha.dtype)),
                atol=ATOL.get(str(self.alpha.dtype)))

    def test_prob(self):

        with paddle.static.program_guard(self.program):

            value = paddle.static.data('value', self._paddle_beta.alpha.shape,
                                       self._paddle_beta.alpha.dtype)
            prob = self._paddle_beta.prob(value)

            random_number = np.random.rand(*self._paddle_beta.alpha.shape)
            feeds = dict(self.feeds, value=random_number)
            [prob] = self.executor.run(self.program,
                                       feed=feeds,
                                       fetch_list=[prob])
            np.testing.assert_allclose(
                prob,
                scipy.stats.beta.pdf(random_number, self.alpha, self.beta),
                rtol=RTOL.get(str(self.alpha.dtype)),
                atol=ATOL.get(str(self.alpha.dtype)))

    def test_log_prob(self):
        with paddle.static.program_guard(self.program):
            value = paddle.static.data('value', self._paddle_beta.alpha.shape,
                                       self._paddle_beta.alpha.dtype)
            prob = self._paddle_beta.log_prob(value)
            random_number = np.random.rand(*self._paddle_beta.alpha.shape)
            feeds = dict(self.feeds, value=random_number)
            [prob] = self.executor.run(self.program,
                                       feed=feeds,
                                       fetch_list=[prob])
            np.testing.assert_allclose(
                prob,
                scipy.stats.beta.logpdf(random_number, self.alpha, self.beta),
                rtol=RTOL.get(str(self.alpha.dtype)),
                atol=ATOL.get(str(self.alpha.dtype)))

    def test_entropy(self):
        with paddle.static.program_guard(self.program):
            [entropy] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_beta.entropy()])
            np.testing.assert_allclose(
                entropy,
                scipy.stats.beta.entropy(self.alpha, self.beta),
                rtol=RTOL.get(str(self.alpha.dtype)),
                atol=ATOL.get(str(self.alpha.dtype)))

    def test_sample(self):
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(self.program,
                                       feed=self.feeds,
                                       fetch_list=self._paddle_beta.sample())
            self.assertTrue(data.shape,
                            np.broadcast_arrays(self.alpha, self.beta)[0].shape)
