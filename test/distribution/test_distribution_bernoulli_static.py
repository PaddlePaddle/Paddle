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

import sys
import unittest

import numpy as np
from distribution.config import ATOL, DEVICES, RTOL
from parameterize import (
    TEST_CASE_NAME,
    parameterize_cls,
    parameterize_func,
    place,
)

sys.path.append("../../distribution")
from test_distribution_bernoulli import BernoulliNumpy, _kstest, _sigmoid

import paddle
from paddle.distribution import Bernoulli
from paddle.distribution.kl import kl_divergence

np.random.seed(2023)
paddle.seed(2023)
paddle.enable_static()
default_dtype = paddle.get_default_dtype()


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'params'),  # params: name, probs, probs_other, value
    [
        (
            'params',
            (
                # 1-D probs
                (
                    'probs_not_iterable',
                    0.3,
                    0.7,
                    1.0,
                ),
                (
                    'probs_not_iterable_and_broadcast_for_value',
                    0.3,
                    0.7,
                    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=default_dtype),
                ),
                # N-D probs
                (
                    'probs_tuple_0305',
                    (0.3, 0.5),
                    0.7,
                    1.0,
                ),
                (
                    'probs_tuple_03050104',
                    ((0.3, 0.5), (0.1, 0.4)),
                    0.7,
                    1.0,
                ),
            ),
        )
    ],
)
class BernoulliTestFeature(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

        self.params_len = len(self.params)

        with paddle.static.program_guard(self.program):
            self.init_numpy_data(self.params)
            self.init_static_data(self.params)

    def init_numpy_data(self, params):
        self.mean_np = []
        self.variance_np = []
        self.log_prob_np = []
        self.prob_np = []
        self.cdf_np = []
        self.entropy_np = []
        self.kl_np = []

        for _, probs, probs_other, value in params:
            rv_np = BernoulliNumpy(probs)
            rv_np_other = BernoulliNumpy(probs_other)

            self.mean_np.append(rv_np.mean)
            self.variance_np.append(rv_np.variance)
            self.log_prob_np.append(rv_np.log_prob(value))
            self.prob_np.append(rv_np.prob(value))
            self.cdf_np.append(rv_np.cdf(value))
            self.entropy_np.append(rv_np.entropy())
            self.kl_np.append(rv_np.kl_divergence(rv_np_other))

    def init_static_data(self, params):
        with paddle.static.program_guard(self.program):
            rv_paddles = []
            rv_paddles_other = []
            values = []
            for _, probs, probs_other, value in params:
                if not isinstance(value, np.ndarray):
                    value = paddle.full([1], value, dtype=default_dtype)
                else:
                    value = paddle.to_tensor(value, place=self.place)

                rv_paddles.append(Bernoulli(probs=paddle.to_tensor(probs)))
                rv_paddles_other.append(
                    Bernoulli(probs=paddle.to_tensor(probs_other))
                )
                values.append(value)

            results = self.executor.run(
                self.program,
                feed={},
                fetch_list=[
                    [
                        rv_paddles[i].mean,
                        rv_paddles[i].variance,
                        rv_paddles[i].log_prob(values[i]),
                        rv_paddles[i].prob(values[i]),
                        rv_paddles[i].cdf(values[i]),
                        rv_paddles[i].entropy(),
                        rv_paddles[i].kl_divergence(rv_paddles_other[i]),
                        kl_divergence(rv_paddles[i], rv_paddles_other[i]),
                    ]
                    for i in range(self.params_len)
                ],
            )

            self.mean_paddle = []
            self.variance_paddle = []
            self.log_prob_paddle = []
            self.prob_paddle = []
            self.cdf_paddle = []
            self.entropy_paddle = []
            self.kl_paddle = []
            self.kl_func_paddle = []
            for i in range(self.params_len):
                (
                    _mean,
                    _variance,
                    _log_prob,
                    _prob,
                    _cdf,
                    _entropy,
                    _kl,
                    _kl_func,
                ) = results[i * 8 : (i + 1) * 8]
                self.mean_paddle.append(_mean)
                self.variance_paddle.append(_variance)
                self.log_prob_paddle.append(_log_prob)
                self.prob_paddle.append(_prob)
                self.cdf_paddle.append(_cdf)
                self.entropy_paddle.append(_entropy)
                self.kl_paddle.append(_kl)
                self.kl_func_paddle.append(_kl_func)

    def test_all(self):
        for i in range(self.params_len):
            self._test_mean(i)
            self._test_variance(i)
            self._test_log_prob(i)
            self._test_prob(i)
            self._test_cdf(i)
            self._test_entropy(i)
            self._test_kl_divergence(i)

    def _test_mean(self, i):
        np.testing.assert_allclose(
            self.mean_np[i],
            self.mean_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

    def _test_variance(self, i):
        np.testing.assert_allclose(
            self.variance_np[i],
            self.variance_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

    def _test_log_prob(self, i):
        np.testing.assert_allclose(
            self.log_prob_np[i],
            self.log_prob_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

    def _test_prob(self, i):
        np.testing.assert_allclose(
            self.prob_np[i],
            self.prob_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

    def _test_cdf(self, i):
        np.testing.assert_allclose(
            self.cdf_np[i],
            self.cdf_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

    def _test_entropy(self, i):
        np.testing.assert_allclose(
            self.entropy_np[i],
            self.entropy_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

    def _test_kl_divergence(self, i):
        np.testing.assert_allclose(
            self.kl_np[i],
            self.kl_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

        np.testing.assert_allclose(
            self.kl_np[i],
            self.kl_func_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'probs', 'shape', 'temperature', 'expected_shape'),
    [
        # 1-D probs
        (
            'probs_03',
            (0.3,),
            [
                100,
            ],
            0.1,
            [100, 1],
        ),
        # N-D probs
        (
            'probs_0305',
            (0.3, 0.5),
            [
                100,
            ],
            0.1,
            [100, 2],
        ),
    ],
)
class BernoulliTestSample(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(self.program):
            self.init_numpy_data(self.probs, self.shape)
            self.init_static_data(self.probs, self.shape, self.temperature)

    def init_numpy_data(self, probs, shape):
        self.rv_np = BernoulliNumpy(probs)
        self.sample_np = self.rv_np.sample(shape)

    def init_static_data(self, probs, shape, temperature):
        with paddle.static.program_guard(self.program):
            self.rv_paddle = Bernoulli(probs=paddle.to_tensor(probs))

            [self.sample_paddle, self.rsample_paddle] = self.executor.run(
                self.program,
                feed={},
                fetch_list=[
                    self.rv_paddle.sample(shape),
                    self.rv_paddle.rsample(shape, temperature),
                ],
            )

    def test_sample(self):
        with paddle.static.program_guard(self.program):
            self.assertEqual(
                list(self.sample_paddle.shape), self.expected_shape
            )

            for i in range(len(self.probs)):
                self.assertTrue(
                    _kstest(
                        self.sample_np[..., i].reshape(-1),
                        self.sample_paddle[..., i].reshape(-1),
                    )
                )

    def test_rsample(self):
        """Compare two samples from `rsample` method, one from scipy and another from paddle."""
        with paddle.static.program_guard(self.program):
            self.assertEqual(
                list(self.rsample_paddle.shape), self.expected_shape
            )

            for i in range(len(self.probs)):
                self.assertTrue(
                    _kstest(
                        self.sample_np[..., i].reshape(-1),
                        (_sigmoid(self.rsample_paddle[..., i]) > 0.5).reshape(
                            -1
                        ),
                        self.temperature,
                    )
                )


@place(DEVICES)
@parameterize_cls([TEST_CASE_NAME], ['BernoulliTestError'])
class BernoulliTestError(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

    @parameterize_func(
        [
            (0,),  # int
            ((0.3,),),  # tuple
            (
                [
                    0.3,
                ],
            ),  # list
            (
                np.array(
                    [
                        0.3,
                    ]
                ),
            ),  # ndarray
            (-1j + 1,),  # complex
            ('0',),  # str
        ]
    )
    def test_bad_init_type(self, probs):
        with paddle.static.program_guard(self.program):
            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[Bernoulli(probs=probs)]
                )

    @parameterize_func(
        [
            (100,),  # int
            (100.0,),  # float
        ]
    )
    def test_bad_sample_shape_type(self, shape):
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(0.3)

            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.sample(shape)]
                )

            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.rsample(shape)]
                )

    @parameterize_func(
        [
            (1,),  # int
        ]
    )
    def test_bad_rsample_temperature_type(self, temperature):
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(0.3)

            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program,
                    feed={},
                    fetch_list=[rv.rsample([100], temperature)],
                )

    @parameterize_func(
        [
            (1,),  # int
            (1.0,),  # float
            ([1.0],),  # list
            ((1.0),),  # tuple
            (np.array(1.0),),  # ndarray
        ]
    )
    def test_bad_value_type(self, value):
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(0.3)

            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.log_prob(value)]
                )

            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.prob(value)]
                )

            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.cdf(value)]
                )

    @parameterize_func(
        [
            (np.array(1.0),),  # ndarray or other distribution
        ]
    )
    def test_bad_kl_other_type(self, other):
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(0.3)

            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.kl_divergence(other)]
                )

    @parameterize_func(
        [
            (paddle.to_tensor([0.1, 0.2, 0.3]),),
        ]
    )
    def test_bad_broadcast(self, value):
        with paddle.static.program_guard(self.program):
            rv = Bernoulli(paddle.to_tensor([0.3, 0.5]))

            # `logits, value = paddle.broadcast_tensors([self.logits, value])`
            # raise ValueError in dygraph, raise TypeError in static.
            with self.assertRaises((TypeError, ValueError)):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.cdf(value)]
                )

            with self.assertRaises((TypeError, ValueError)):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.log_prob(value)]
                )

            with self.assertRaises((TypeError, ValueError)):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.prob(value)]
                )


if __name__ == '__main__':
    unittest.main()
