# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import scipy.stats
from distribution.config import ATOL, DEVICES
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand

import paddle
from paddle.distribution import exponential

np.random.seed(2023)


# @place(DEVICES)
# @parameterize_cls(
#     (TEST_CASE_NAME, 'rate'),
#     [
#         (
#             'one-dim',
#             xrand(
#                 (2,),
#                 dtype='float32',
#                 min=0.001,
#             ),
#         ),
#         (
#             'multi-dim',
#             xrand(
#                 (2, 3),
#                 dtype='float32',
#                 min=0.001,
#             ),
#         ),
#     ],
# )
# class TestExponential(unittest.TestCase):
#     def setUp(self):
#         rate = self.rate

#         if isinstance(self.rate, numbers.Real):
#             self.scale = 1 / rate

#         else:
#             rate = paddle.to_tensor(self.rate, dtype=paddle.float32)
#             self.scale = rate.reciprocal()

#         self._paddle_expon = exponential.Exponential(rate)

#     def test_mean(self):
#         with paddle.base.dygraph.guard(self.place):
#             np.testing.assert_allclose(
#                 self._paddle_expon.mean,
#                 scipy.stats.expon.mean(scale=self.scale),
#                 rtol=RTOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#                 atol=ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#             )

#     def test_variance(self):
#         with paddle.base.dygraph.guard(self.place):
#             np.testing.assert_allclose(
#                 self._paddle_expon.variance,
#                 scipy.stats.expon.var(scale=self.scale),
#                 rtol=RTOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#                 atol=ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#             )

#     def test_prob(self):
#         value = [np.random.rand(*self._paddle_expon.rate.shape)]

#         for v in value:
#             with paddle.base.dygraph.guard(self.place):
#                 np.testing.assert_allclose(
#                     self._paddle_expon.prob(paddle.to_tensor(v)),
#                     scipy.stats.expon.pdf(v, scale=self.scale),
#                     rtol=RTOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#                     atol=ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#                 )

#     def test_log_prob(self):
#         value = [np.random.rand(*self._paddle_expon.rate.shape)]

#         for v in value:
#             with paddle.base.dygraph.guard(self.place):
#                 np.testing.assert_allclose(
#                     self._paddle_expon.log_prob(paddle.to_tensor(v)),
#                     scipy.stats.expon.logpdf(v, scale=self.scale),
#                     rtol=RTOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#                     atol=ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#                 )

#     def test_entropy(self):
#         with paddle.base.dygraph.guard(self.place):
#             np.testing.assert_allclose(
#                 self._paddle_expon.entropy(),
#                 scipy.stats.expon.entropy(scale=self.scale),
#                 rtol=RTOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#                 atol=ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
#             )


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'rate'),
    [
        (
            'one-dim',
            xrand(
                (2,),
                dtype='float32',
                min=0.001,
            ),
        ),
        # (
        #     'multi-dim',
        #     xrand(
        #         (2, 3),
        #         dtype='float32',
        #         min=0.001,
        #     ),
        # ),
    ],
)
class TestExponential(unittest.TestCase):
    def setUp(self):
        rate = self.rate

        if isinstance(self.rate, numbers.Real):
            self.scale = 1 / rate

        else:
            rate = paddle.to_tensor(self.rate, dtype=paddle.float32)
            self.scale = rate.reciprocal()

        self._paddle_expon = exponential.Exponential(rate)

    def test_sample_shape(self):
        cases = [
            {
                'input': (),
                'expect': ()
                + tuple(paddle.squeeze(self._paddle_expon.rate).shape),
            },
            {
                'input': (4, 2),
                'expect': (4, 2)
                + tuple(paddle.squeeze(self._paddle_expon.rate).shape),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_expon.sample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_sample(self):
        sample_shape = (20000,)
        samples = self._paddle_expon.sample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.rate.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.expon.mean(scale=self.scale),
            rtol=0.1,
            atol=ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.expon.var(scale=self.scale),
            rtol=0.1,
            atol=ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
        )

    def test_rsample_shape(self):
        cases = [
            {
                'input': (),
                'expect': ()
                + tuple(paddle.squeeze(self._paddle_expon.rate).shape),
            },
            {
                'input': (2, 5),
                'expect': (2, 5)
                + tuple(paddle.squeeze(self._paddle_expon.rate).shape),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_expon.rsample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_rsample(self):
        sample_shape = (20000,)
        samples = self._paddle_expon.rsample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.rate.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.expon.mean(scale=self.scale),
            rtol=0.1,
            atol=ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.expon.var(scale=self.scale),
            rtol=0.1,
            atol=ATOL.get(str(self._paddle_expon.rate.numpy().dtype)),
        )


if __name__ == '__main__':
    unittest.main()
