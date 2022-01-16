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
import mock_data as mock


@config.place(config.DEVICES)
@config.parameterize(
    (config.TEST_CASE_NAME, 'dist'), [('test-mock-exp',
                                       mock.Exponential(rate=paddle.rand(
                                           [100, 200, 99],
                                           dtype=config.DEFAULT_DTYPE)))])
class TestExponentialFamily(unittest.TestCase):
    def test_entropy(self):
        np.testing.assert_allclose(
            self.dist.entropy(),
            paddle.distribution.ExponentialFamily.entropy(self.dist),
            rtol=config.RTOL.get(config.DEFAULT_DTYPE),
            atol=config.ATOL.get(config.DEFAULT_DTYPE))


@config.place(config.DEVICES)
@config.parameterize(
    (config.TEST_CASE_NAME, 'dist'),
    [('test-dummy', mock.DummyExpFamily(0.5, 0.5)),
     ('test-dirichlet',
      paddle.distribution.Dirichlet(paddle.to_tensor(config.xrand()))), (
          'test-beta', paddle.distribution.Beta(
              paddle.to_tensor(config.xrand()),
              paddle.to_tensor(config.xrand())))])
class TestExponentialFamilyException(unittest.TestCase):
    def test_entropy_exception(self):
        with self.assertRaises(NotImplementedError):
            paddle.distribution.ExponentialFamily.entropy(self.dist)
