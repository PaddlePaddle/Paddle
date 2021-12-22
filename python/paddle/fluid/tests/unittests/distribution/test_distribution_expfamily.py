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


class TestExponentialFamily(unittest.TestCase):
    def test_entropy(self):
        mock_dist = mock.Exponential(rate=paddle.rand(
            [100, 200, 99], dtype=config.DEFAULT_DTYPE))
        np.testing.assert_allclose(
            mock_dist.entropy(),
            paddle.distribution.ExponentialFamily.entropy(mock_dist),
            rtol=config.RTOL.get(config.DEFAULT_DTYPE),
            atol=config.ATOL.get(config.DEFAULT_DTYPE))

    def test_entropy_expection(self):
        with self.assertRaises(NotImplementedError):
            paddle.distribution.ExponentialFamily.entropy(
                paddle.distribution.Beta(0.5, 0.5))


if __name__ == '__main__':
    unittest.main()
