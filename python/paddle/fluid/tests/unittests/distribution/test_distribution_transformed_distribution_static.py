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

paddle.enable_static()


@param.place(config.DEVICES)
@param.param_cls((param.TEST_CASE_NAME, 'base', 'transforms'),
                 [('base_normal', paddle.distribution.Normal,
                   [paddle.distribution.ExpTransform()])])
class TestIndependent(unittest.TestCase):
    def setUp(self):
        value = np.array([0.5])
        loc = np.array([0.])
        scale = np.array([1.])
        shape = [5, 10, 8]
        self.dtype = value.dtype
        exe = paddle.static.Executor()
        sp = paddle.static.Program()
        mp = paddle.static.Program()
        with paddle.static.program_guard(mp, sp):
            static_value = paddle.static.data('value', value.shape, value.dtype)
            static_loc = paddle.static.data('loc', loc.shape, loc.dtype)
            static_scale = paddle.static.data('scale', scale.shape, scale.dtype)
            self.base = self.base(static_loc, static_scale)
            self._t = paddle.distribution.TransformedDistribution(
                self.base, self.transforms)
            actual_log_prob = self._t.log_prob(static_value)
            expected_log_prob = self.transformed_log_prob(
                static_value, self.base, self.transforms)
            sample_data = self._t.sample(shape)

        exe.run(sp)
        [self.actual_log_prob, self.expected_log_prob,
         self.sample_data] = exe.run(
             mp,
             feed={'value': value,
                   'loc': loc,
                   'scale': scale},
             fetch_list=[actual_log_prob, expected_log_prob, sample_data])

    def test_log_prob(self):
        np.testing.assert_allclose(
            self.actual_log_prob,
            self.expected_log_prob,
            rtol=config.RTOL.get(str(self.dtype)),
            atol=config.ATOL.get(str(self.dtype)))

    def transformed_log_prob(self, value, base, transforms):
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
        expected_shape = (5, 10, 8, 1)
        self.assertEqual(tuple(self.sample_data.shape), expected_shape)
        self.assertEqual(self.sample_data.dtype, self.dtype)


if __name__ == '__main__':
    unittest.main()
