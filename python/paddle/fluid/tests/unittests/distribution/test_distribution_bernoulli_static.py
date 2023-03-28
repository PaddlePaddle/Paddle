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
from config import ATOL, DEVICES, RTOL
from parameterize import (
    TEST_CASE_NAME,
    parameterize_cls,
    parameterize_func,
    place,
)
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
    (TEST_CASE_NAME, 'probs', 'probs_other', 'value'),
    [
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
        (
            'probs_iterable',
            [
                0.3,
            ],
            [
                0.7,
            ],
            1.0,
        ),
        # N-D probs
        (
            'probs_tuple_0305',
            (0.3, 0.5),
            [
                0.7,
            ],
            1.0,
        ),
        (
            'probs_tuple_03050104',
            ((0.3, 0.5), (0.1, 0.4)),
            [
                0.7,
            ],
            1.0,
        ),
    ],
)
class BernoulliTestFeature(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(self.program):
            self.init_numpy_data(self.probs, self.probs_other, self.value)
            self.init_static_data(self.probs, self.probs_other, self.value)

    def init_numpy_data(self, probs, probs_other, value):
        self.rv_np = BernoulliNumpy(probs)
        self.rv_np_other = BernoulliNumpy(probs_other)

        self.log_prob_np = self.rv_np.log_prob(value)
        self.prob_np = self.rv_np.prob(value)
        self.cdf_np = self.rv_np.cdf(value)

    def init_static_data(self, probs, probs_other, value):
        with paddle.static.program_guard(self.program):
            if not isinstance(value, np.ndarray):
                value = paddle.full([1], value, dtype=default_dtype)
            else:
                value = paddle.to_tensor(value, place=self.place)

            self.rv_paddle = Bernoulli(probs=probs)
            self.rv_paddle_other = Bernoulli(probs=probs_other)

            [
                self.mean,
                self.variance,
                self.log_prob,
                self.prob,
                self.cdf,
                self.entropy,
                self.kl_divergence,
            ] = self.executor.run(
                self.program,
                feed={},
                fetch_list=[
                    self.rv_paddle.mean,
                    self.rv_paddle.variance,
                    self.rv_paddle.log_prob(value),
                    self.rv_paddle.prob(value),
                    self.rv_paddle.cdf(value),
                    self.rv_paddle.entropy(),
                    self.rv_paddle.kl_divergence(self.rv_paddle_other),
                ],
            )

    def test_mean(self):
        with paddle.static.program_guard(self.program):
            np.testing.assert_allclose(
                self.mean,
                self.rv_np.mean,
                rtol=RTOL.get(default_dtype),
                atol=ATOL.get(default_dtype),
            )

    def test_variance(self):
        with paddle.static.program_guard(self.program):
            np.testing.assert_allclose(
                self.variance,
                self.rv_np.variance,
                rtol=RTOL.get(default_dtype),
                atol=ATOL.get(default_dtype),
            )

    def test_log_prob(self):
        with paddle.static.program_guard(self.program):
            np.testing.assert_allclose(
                self.log_prob,
                self.log_prob_np,
                rtol=RTOL.get(default_dtype),
                atol=ATOL.get(default_dtype),
            )

    def test_prob(self):
        with paddle.static.program_guard(self.program):
            np.testing.assert_allclose(
                self.prob,
                self.prob_np,
                rtol=RTOL.get(default_dtype),
                atol=ATOL.get(default_dtype),
            )

    def test_cdf(self):
        with paddle.static.program_guard(self.program):
            np.testing.assert_allclose(
                self.cdf,
                self.cdf_np,
                rtol=RTOL.get(default_dtype),
                atol=ATOL.get(default_dtype),
            )

    def test_entropy(self):
        with paddle.static.program_guard(self.program):
            np.testing.assert_allclose(
                self.entropy,
                self.rv_np.entropy(),
                rtol=RTOL.get(default_dtype),
                atol=ATOL.get(default_dtype),
            )

    def test_kl_divergence(self):
        with paddle.static.program_guard(self.program):
            np.testing.assert_allclose(
                self.kl_divergence,
                self.rv_np.kl_divergence(self.rv_np_other),
                rtol=RTOL.get(default_dtype),
                atol=ATOL.get(default_dtype),
            )

            [kl] = self.executor.run(
                self.program,
                feed={},
                fetch_list=[
                    kl_divergence(self.rv_paddle, self.rv_paddle_other)
                ],
            )

            np.testing.assert_allclose(
                kl,
                self.rv_np.kl_divergence(self.rv_np_other),
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
            [
                0.3,
            ],
            [
                100,
            ],
            0.1,
            [100, 1],
        ),
        # N-D probs
        (
            'probs_tuple_0305',
            [0.3, 0.5],
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
            self.rv_paddle = Bernoulli(probs=probs)

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
            rv = Bernoulli([0.3, 0.5])

            # `logits, value = paddle.broadcast_tensors([self.logits, value])`
            # raise ValueError in dygraph, raise TypeError in static.
            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.cdf(value)]
                )

            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.log_prob(value)]
                )

            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program, feed={}, fetch_list=[rv.prob(value)]
                )


if __name__ == '__main__':
    unittest.main()
