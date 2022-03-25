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

paddle.enable_static()


@config.place(config.DEVICES)
class TestExponentialFamily(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor()
        with paddle.static.program_guard(self.program):
            rate_np = config.xrand((100, 200, 99))
            rate = paddle.static.data('rate', rate_np.shape, rate_np.dtype)
            self.mock_dist = mock.Exponential(rate)
            self.feeds = {'rate': rate_np}

    def test_entropy(self):
        with paddle.static.program_guard(self.program):
            [out1, out2] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[
                    self.mock_dist.entropy(),
                    paddle.distribution.ExponentialFamily.entropy(
                        self.mock_dist)
                ])

            np.testing.assert_allclose(
                out1,
                out2,
                rtol=config.RTOL.get(config.DEFAULT_DTYPE),
                atol=config.ATOL.get(config.DEFAULT_DTYPE))

    def test_entropy_exception(self):
        with paddle.static.program_guard(self.program):
            with self.assertRaises(NotImplementedError):
                paddle.distribution.ExponentialFamily.entropy(
                    mock.DummyExpFamily(0.5, 0.5))
