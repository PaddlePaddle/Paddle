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


@param.place(config.DEVICES)
@param.param_cls((param.TEST_CASE_NAME, 'base', 'transforms'),
                 [('base_normal', paddle.distribution.Normal(0., 1.),
                   [paddle.distribution.ExpTransform()])])
class TestIndependent(unittest.TestCase):
    def setUp(self):
        self._t = paddle.distribution.TransformedDistribution(self.base,
                                                              self.transforms)

    def _np_sum_rightmost(self, value, n):
        return np.sum(value, tuple(range(-n, 0))) if n > 0 else value

    def test_log_prob(self):
        value = paddle.to_tensor(0.5)
        np.testing.assert_allclose(
            self.simple_log_prob(value, self.base, self.transforms),
            self._t.log_prob(value),
            rtol=config.RTOL.get(str(value.numpy().dtype)),
            atol=config.ATOL.get(str(value.numpy().dtype)))

    def simple_log_prob(self, value, base, transforms):
        log_prob = 0.0
        y = value
        for t in reversed(transforms):
            x = t.inverse(y)
            log_prob = log_prob - t.forward_log_det_jacobian(x)
            y = x
        log_prob += base.log_prob(y)
        return log_prob

    # TODO(cxxly): Add Kolmogorov-Smirnov test for sample result.
    def test_sample(self):
        shape = [5, 10, 8]
        expected_shape = (5, 10, 8)
        data = self._t.sample(shape)
        self.assertEqual(tuple(data.shape), expected_shape)
        self.assertEqual(data.dtype, self.base.loc.dtype)


if __name__ == '__main__':
    unittest.main()
