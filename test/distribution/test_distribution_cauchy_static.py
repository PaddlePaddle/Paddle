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
from test_distribution_cauchy import CauchyNumpy, _kstest

import paddle
from paddle.distribution import Cauchy
from paddle.distribution.kl import kl_divergence

np.random.seed(2023)
paddle.seed(2023)
paddle.enable_static()
default_dtype = paddle.get_default_dtype()


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'params'),
    # params: name, loc, scale, loc_other, scale_other, value
    [
        (
            'params',
            (
                # 1-D params
                (
                    'params_not_iterable',
                    0.3,
                    1.2,
                    -1.2,
                    2.3,
                    3.4,
                ),
                (
                    'params_not_iterable_and_broadcast_for_value',
                    0.3,
                    1.2,
                    -1.2,
                    2.3,
                    np.array([[0.1, 1.2], [1.2, 3.4]], dtype=default_dtype),
                ),
                # N-D params
                (
                    'params_tuple_0305',
                    (0.3, 0.5),
                    0.7,
                    -1.2,
                    2.3,
                    3.4,
                ),
                (
                    'params_tuple_03050104',
                    ((0.3, 0.5), (0.1, 0.4)),
                    0.7,
                    -1.2,
                    2.3,
                    3.4,
                ),
            ),
        )
    ],
    test_pir=True,
)
class CauchyTestFeature(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

        self.params_len = len(self.params)

        with paddle.static.program_guard(self.program):
            self.init_numpy_data(self.params)
            self.init_static_data(self.params)

    def init_numpy_data(self, params):
        self.log_prob_np = []
        self.prob_np = []
        self.cdf_np = []
        self.entropy_np = []
        self.kl_np = []
        self.shapes = []

        for _, loc, scale, loc_other, scale_other, value in params:
            rv_np = CauchyNumpy(loc=loc, scale=scale)
            rv_np_other = CauchyNumpy(loc=loc_other, scale=scale_other)

            self.log_prob_np.append(rv_np.log_prob(value))
            self.prob_np.append(rv_np.prob(value))
            self.cdf_np.append(rv_np.cdf(value))
            self.entropy_np.append(rv_np.entropy())
            self.kl_np.append(rv_np.kl_divergence(rv_np_other))
            # paddle return data ndim>0
            self.shapes.append(
                (np.array(loc) + np.array(scale) + np.array(value)).shape
                or (1,)
            )

    def init_static_data(self, params):
        with paddle.static.program_guard(self.program):
            rv_paddles = []
            rv_paddles_other = []
            values = []
            for name, loc, scale, loc_other, scale_other, value in params:
                if not isinstance(value, np.ndarray):
                    value = paddle.full([1], value, dtype=default_dtype)
                else:
                    value = paddle.to_tensor(value, place=self.place)

                # We should set name in static mode, or the executor confuse rv_paddles[i].
                rv_paddles.append(
                    Cauchy(
                        loc=paddle.to_tensor(loc),
                        scale=paddle.to_tensor(scale),
                        name=name,
                    )
                )
                rv_paddles_other.append(
                    Cauchy(
                        loc=paddle.to_tensor(loc_other),
                        scale=paddle.to_tensor(scale_other),
                        name=name,
                    )
                )
                values.append(value)

            results = self.executor.run(
                self.program,
                feed={},
                fetch_list=[
                    [
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

            self.log_prob_paddle = []
            self.prob_paddle = []
            self.cdf_paddle = []
            self.entropy_paddle = []
            self.kl_paddle = []
            self.kl_func_paddle = []
            for i in range(self.params_len):
                (
                    _log_prob,
                    _prob,
                    _cdf,
                    _entropy,
                    _kl,
                    _kl_func,
                ) = results[i * 6 : (i + 1) * 6]
                self.log_prob_paddle.append(_log_prob)
                self.prob_paddle.append(_prob)
                self.cdf_paddle.append(_cdf)
                self.entropy_paddle.append(_entropy)
                self.kl_paddle.append(_kl)
                self.kl_func_paddle.append(_kl_func)

    def test_all(self):
        for i in range(self.params_len):
            self._test_log_prob(i)
            self._test_prob(i)
            self._test_cdf(i)
            self._test_entropy(i)
            self._test_kl_divergence(i)

    def _test_log_prob(self, i):
        np.testing.assert_allclose(
            self.log_prob_np[i],
            self.log_prob_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

        # check shape
        self.assertTrue(self.log_prob_paddle[i].shape == self.shapes[i])

    def _test_prob(self, i):
        np.testing.assert_allclose(
            self.prob_np[i],
            self.prob_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

        # check shape
        self.assertTrue(self.prob_paddle[i].shape == self.shapes[i])

    def _test_cdf(self, i):
        np.testing.assert_allclose(
            self.cdf_np[i],
            self.cdf_paddle[i],
            rtol=RTOL.get(default_dtype),
            atol=ATOL.get(default_dtype),
        )

        # check shape
        self.assertTrue(self.cdf_paddle[i].shape == self.shapes[i])

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
    (
        TEST_CASE_NAME,
        'loc',
        'scale',
        'shape',
        'expected_shape',
    ),
    [
        # 1-D params
        (
            'params_1d',
            [0.1],
            [1.2],
            [100],
            [100, 1],
        ),
        # N-D params
        (
            'params_2d',
            [0.3],
            [1.2, 2.3],
            [100],
            [100, 2],
        ),
    ],
    test_pir=True,
)
class CauchyTestSample(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

        with paddle.static.program_guard(self.program):
            self.init_numpy_data(self.loc, self.scale, self.shape)
            self.init_static_data(self.loc, self.scale, self.shape)

    def init_numpy_data(self, loc, scale, shape):
        self.rv_np = CauchyNumpy(loc=loc, scale=scale)
        self.sample_np = self.rv_np.sample(shape)

    def init_static_data(self, loc, scale, shape):
        with paddle.static.program_guard(self.program):
            self.rv_paddle = Cauchy(
                loc=paddle.to_tensor(loc),
                scale=paddle.to_tensor(scale),
            )

            [self.sample_paddle, self.rsample_paddle] = self.executor.run(
                self.program,
                feed={},
                fetch_list=[
                    self.rv_paddle.sample(shape),
                    self.rv_paddle.rsample(shape),
                ],
            )

    def test_sample(self):
        with paddle.static.program_guard(self.program):
            self.assertEqual(
                list(self.sample_paddle.shape), self.expected_shape
            )

            for i in range(self.expected_shape[-1]):
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

            for i in range(self.expected_shape[-1]):
                self.assertTrue(
                    _kstest(
                        self.sample_np[..., i].reshape(-1),
                        self.rsample_paddle[..., i].reshape(-1),
                    )
                )


@place(DEVICES)
@parameterize_cls([TEST_CASE_NAME], ['CauchyTestError'], test_pir=True)
class CauchyTestError(unittest.TestCase):
    def setUp(self):
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)

    @parameterize_func(
        [
            ((0.3,),),  # tuple
            ([0.3],),  # list
            (np.array([0.3]),),  # ndarray
            (-1j + 1,),  # complex
            ('0',),  # str
        ]
    )
    def test_bad_init_type(self, param):
        """Test bad init for loc/scale"""
        with paddle.static.program_guard(self.program):
            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program,
                    feed={},
                    fetch_list=[Cauchy(loc=0.0, scale=param).scale],
                )
            with self.assertRaises(TypeError):
                [_] = self.executor.run(
                    self.program,
                    feed={},
                    fetch_list=[Cauchy(loc=param, scale=1.0).loc],
                )

    def test_bad_property(self):
        """For property like mean/variance/stddev which is undefined in math,
        we should raise `ValueError` instead of `NotImplementedError`.
        """
        with paddle.static.program_guard(self.program):
            rv = Cauchy(loc=0.0, scale=1.0)
            with self.assertRaises(ValueError):
                _ = rv.mean
            with self.assertRaises(ValueError):
                _ = rv.variance
            with self.assertRaises(ValueError):
                _ = rv.stddev

    @parameterize_func(
        [
            (100,),  # int
            (100.0,),  # float
        ]
    )
    def test_bad_sample_shape_type(self, shape):
        with paddle.static.program_guard(self.program):
            rv = Cauchy(loc=0.0, scale=1.0)

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
            (1.0,),  # float
            ([1.0],),  # list
            ((1.0),),  # tuple
            (np.array(1.0),),  # ndarray
        ]
    )
    def test_bad_value_type(self, value):
        with paddle.static.program_guard(self.program):
            rv = Cauchy(loc=0.0, scale=1.0)

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
            rv = Cauchy(loc=0.0, scale=1.0)

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
            rv = Cauchy(
                loc=paddle.to_tensor(0.0), scale=paddle.to_tensor((1.0, 2.0))
            )

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
