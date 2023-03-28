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
import scipy.special
import scipy.stats
from config import ATOL, DEVICES, RTOL
from parameterize import (
    TEST_CASE_NAME,
    parameterize_cls,
    parameterize_func,
    place,
)
from test_distribution import DistributionNumpy

import paddle
from paddle.distribution import Bernoulli
from paddle.distribution.kl import kl_divergence
from paddle.fluid.data_feeder import convert_dtype

np.random.seed(2023)
paddle.seed(2023)

# smallest representable number
EPS = {
    'float32': 1e-03,
    'float64': 1e-05,
}


def _clip_probs_ndarray(probs, dtype):
    """Clip probs from [0, 1] to (0, 1) with `eps`"""
    eps = EPS.get(dtype, 1e-05)
    return np.clip(probs, a_min=eps, a_max=1 - eps)


def _sigmoid(z):
    return scipy.special.expit(z)


def _kstest(samples_a, samples_b, temperature=1):
    """Uses the Kolmogorov-Smirnov test for goodness of fit."""
    _, p_value = scipy.stats.ks_2samp(samples_a, samples_b)
    return not (p_value < 0.02 * (min(1, temperature)))


class BernoulliNumpy(DistributionNumpy):
    def __init__(self, probs):
        probs = np.array(probs)
        if str(probs.dtype) not in ['float32', 'float64']:
            self.dtype = 'float32'
        else:
            self.dtype = probs.dtype

        self.batch_shape = np.shape(probs) or [1]

        self.probs = _clip_probs_ndarray(
            np.array(probs, dtype=self.dtype), str(self.dtype)
        )
        self.logits = self._probs_to_logits(self.probs, is_binary=True)

        self.rv = scipy.stats.bernoulli(self.probs)

    @property
    def mean(self):
        return self.rv.mean()

    @property
    def variance(self):
        return self.rv.var()

    def sample(self, shape):
        return self.rv.rvs(
            size=np.array(shape).tolist() + list(self.batch_shape)
        )

    def log_prob(self, value):
        return self.rv.logpmf(value)

    def prob(self, value):
        return self.rv.pmf(value)

    def cdf(self, value):
        return self.rv.cdf(value)

    def entropy(self):
        return self.rv.entropy()

    def kl_divergence(self, other):
        """
        .. math::

            KL[a || b] = Pa * Log[Pa / Pb] + (1 - Pa) * Log[(1 - Pa) / (1 - Pb)]
        """
        p_a = self.probs
        p_b = other.probs
        return p_a * np.log(p_a / p_b) + (1 - p_a) * np.log(
            (1 - p_a) / (1 - p_b)
        )

    def _probs_to_logits(self, probs, is_binary=False):
        return (
            (np.log(probs) - np.log1p(-probs)) if is_binary else np.log(probs)
        )


class BernoulliTest(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(self.place)
        with paddle.fluid.dygraph.guard(self.place):
            # just for convenience
            self.dtype = self.expected_dtype

            # init numpy with `dtype`
            self.init_numpy_data(self.probs, self.dtype)

            # init paddle and check dtype convert.
            self.init_dynamic_data(self.probs, self.default_dtype, self.dtype)

    def init_numpy_data(self, probs, dtype):
        probs = np.array(probs).astype(dtype)
        self.rv_np = BernoulliNumpy(probs)

    def init_dynamic_data(self, probs, default_dtype, dtype):
        if default_dtype.startswith('int'):
            """Check `dtype` and convert from `int` to `float32` instead of `float64` what `Numpy` does.
            ``` python
                a = np.array(0.3)
                b = paddle.to_tensor(0.3)
                print(a.dtype, b.dtype)
                # (dtype('float64'), paddle.float32)
            ```
            """
            with self.assertWarns(UserWarning):
                self.rv_paddle = Bernoulli(probs)
                self.assertTrue(
                    'float32' == convert_dtype(self.rv_paddle.probs.dtype),
                    (dtype, self.rv_paddle.probs.dtype),
                )
        else:
            self.rv_paddle = Bernoulli(probs)
            self.assertTrue(
                dtype == convert_dtype(self.rv_paddle.probs.dtype),
                (dtype, self.rv_paddle.probs.dtype),
            )

        self.assertTrue(list(self.rv_paddle.batch_shape) != [])


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'probs', 'default_dtype', 'expected_dtype'),
    [
        # 1-D probs
        ('probs_00', 0.0, 'float64', 'float32'),
        ('probs_03', 0.3, 'float64', 'float32'),
        ('probs_10', 1.0, 'float64', 'float32'),
        ('probs_03_tuple', (0.3,), 'float64', 'float32'),
        ('probs_03_tuple_int', (0,), 'int64', 'float32'),
        (
            'probs_03_list',
            [
                0.3,
            ],
            'float64',
            'float32',
        ),
        ('probs_ndarray_03_int', np.array(0), 'int64', 'float32'),
        (
            'probs_ndarray_03_32',
            np.array(0.3, dtype='float32'),
            'float32',
            'float32',
        ),
        ('probs_ndarray_03_64', np.array(0.3), 'float64', 'float64'),
        (
            'probs_ndarray_03_list_32',
            np.array(
                [
                    0.3,
                ],
                dtype='float32',
            ),
            'float32',
            'float32',
        ),
        (
            'probs_ndarray_03_list_64',
            np.array(
                [
                    0.3,
                ]
            ),
            'float64',
            'float64',
        ),
        ('probs_tensor_03_int', paddle.to_tensor(0), 'int64', 'float32'),
        ('probs_tensor_03_32', paddle.to_tensor(0.3), 'float32', 'float32'),
        (
            'probs_tensor_03_64',
            paddle.to_tensor(0.3, dtype='float64'),
            'float64',
            'float64',
        ),
        (
            'probs_tensor_03_list_int',
            paddle.to_tensor(
                [
                    0,
                ]
            ),
            'int64',
            'float32',
        ),
        (
            'probs_tensor_03_list_32',
            paddle.to_tensor(
                [
                    0.3,
                ]
            ),
            'float32',
            'float32',
        ),
        (
            'probs_tensor_03_list_64',
            paddle.to_tensor(
                [
                    0.3,
                ],
                dtype='float64',
            ),
            'float64',
            'float64',
        ),
        # N-D probs
        ('probs_tuple_0305', (0.3, 0.5), 'float64', 'float32'),
        (
            'probs_tuple_03050104',
            ((0.3, 0.5), (0.1, 0.4)),
            'float64',
            'float32',
        ),
    ],
)
class BernoulliTestFeature(BernoulliTest):
    def test_mean(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self.rv_paddle.mean,
                self.rv_np.mean,
                rtol=RTOL.get(self.dtype),
                atol=ATOL.get(self.dtype),
            )

    def test_variance(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self.rv_paddle.variance,
                self.rv_np.variance,
                rtol=RTOL.get(self.dtype),
                atol=ATOL.get(self.dtype),
            )

    @parameterize_func(
        [
            (
                paddle.to_tensor(
                    [
                        0.0,
                    ]
                ),
            ),
            (
                paddle.to_tensor(
                    0.0,
                ),
            ),
            (paddle.to_tensor(1.0),),
            (paddle.to_tensor(0.0, dtype='float64'),),
        ]
    )
    def test_log_prob(self, value):
        with paddle.fluid.dygraph.guard(self.place):
            if convert_dtype(value.dtype) == convert_dtype(
                self.rv_paddle.probs.dtype
            ):
                log_prob = self.rv_paddle.log_prob(value)
                np.testing.assert_allclose(
                    log_prob,
                    self.rv_np.log_prob(value),
                    rtol=RTOL.get(self.dtype),
                    atol=ATOL.get(self.dtype),
                )
                self.assertTrue(self.dtype == convert_dtype(log_prob.dtype))

            else:
                with self.assertWarns(UserWarning):
                    self.rv_paddle.log_prob(value)

    @parameterize_func(
        [
            (
                paddle.to_tensor(
                    [
                        0.0,
                    ]
                ),
            ),
            (paddle.to_tensor(0.0),),
            (paddle.to_tensor(1.0),),
            (paddle.to_tensor(0.0, dtype='float64'),),
        ]
    )
    def test_prob(self, value):
        with paddle.fluid.dygraph.guard(self.place):
            if convert_dtype(value.dtype) == convert_dtype(
                self.rv_paddle.probs.dtype
            ):
                prob = self.rv_paddle.prob(value)
                np.testing.assert_allclose(
                    prob,
                    self.rv_np.prob(value),
                    rtol=RTOL.get(self.dtype),
                    atol=ATOL.get(self.dtype),
                )
                self.assertTrue(self.dtype == convert_dtype(prob.dtype))

            else:
                with self.assertWarns(UserWarning):
                    self.rv_paddle.prob(value)

    @parameterize_func(
        [
            (
                paddle.to_tensor(
                    [
                        0.0,
                    ]
                ),
            ),
            (paddle.to_tensor(0.0),),
            (paddle.to_tensor(0.3),),
            (paddle.to_tensor(0.7),),
            (paddle.to_tensor(1.0),),
            (paddle.to_tensor(0.0, dtype='float64'),),
        ]
    )
    def test_cdf(self, value):
        with paddle.fluid.dygraph.guard(self.place):
            if convert_dtype(value.dtype) == convert_dtype(
                self.rv_paddle.probs.dtype
            ):
                cdf = self.rv_paddle.cdf(value)
                np.testing.assert_allclose(
                    cdf,
                    self.rv_np.cdf(value),
                    rtol=RTOL.get(self.dtype),
                    atol=ATOL.get(self.dtype),
                )
                self.assertTrue(self.dtype == convert_dtype(cdf.dtype))

            else:
                with self.assertWarns(UserWarning):
                    self.rv_paddle.cdf(value)

    def test_entropy(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self.rv_paddle.entropy(),
                self.rv_np.entropy(),
                rtol=RTOL.get(self.dtype),
                atol=ATOL.get(self.dtype),
            )

    def test_kl_divergence(self):
        with paddle.fluid.dygraph.guard(self.place):
            other_probs = np.array(0.9, dtype=self.dtype)

            rv_paddle_other = Bernoulli(other_probs)
            rv_np_other = BernoulliNumpy(other_probs)

            np.testing.assert_allclose(
                self.rv_paddle.kl_divergence(rv_paddle_other),
                self.rv_np.kl_divergence(rv_np_other),
                rtol=RTOL.get(self.dtype),
                atol=ATOL.get(self.dtype),
            )

            np.testing.assert_allclose(
                kl_divergence(self.rv_paddle, rv_paddle_other),
                self.rv_np.kl_divergence(rv_np_other),
                rtol=RTOL.get(self.dtype),
                atol=ATOL.get(self.dtype),
            )


@place(DEVICES)
@parameterize_cls(
    (
        TEST_CASE_NAME,
        'probs',
        'default_dtype',
        'expected_dtype',
        'shape',
        'expected_shape',
    ),
    [
        # 1-D probs
        (
            'probs_03',
            (0.3,),
            'float64',
            'float32',
            [
                100,
            ],
            [100, 1],
        ),
        ('probs_03_tuple', (0.3,), 'float64', 'float32', (100,), [100, 1]),
        (
            'probs_03_ndarray',
            (0.3,),
            'float64',
            'float32',
            np.array(
                [
                    100,
                ]
            ),
            [100, 1],
        ),
        (
            'probs_03_tensor',
            (0.3,),
            'float64',
            'float32',
            paddle.to_tensor(
                [
                    100,
                ]
            ),
            [100, 1],
        ),
        ('probs_03_2d', (0.3,), 'float64', 'float32', [100, 2], [100, 2, 1]),
        (
            'probs_03_3d',
            (0.3,),
            'float64',
            'float32',
            [100, 2, 3],
            [100, 2, 3, 1],
        ),
        # N-D probs
        (
            'probs_tuple_0305',
            (0.3, 0.5),
            'float64',
            'float32',
            [
                100,
            ],
            [100, 2],
        ),
        (
            'probs_tuple_0305_2d',
            (0.3, 0.5),
            'float64',
            'float32',
            [100, 3],
            [100, 3, 2],
        ),
        (
            'probs_tuple_0305_3d',
            (0.3, 0.5),
            'float64',
            'float32',
            [100, 4, 3],
            [100, 4, 3, 2],
        ),
    ],
)
class BernoulliTestSample(BernoulliTest):
    def test_sample(self):
        with paddle.fluid.dygraph.guard(self.place):
            sample_np = self.rv_np.sample(self.shape)
            sample_paddle = self.rv_paddle.sample(self.shape)

            self.assertEqual(list(sample_paddle.shape), self.expected_shape)
            self.assertEqual(sample_paddle.dtype, self.rv_paddle.probs.dtype)

            for i in range(len(self.probs)):
                self.assertTrue(
                    _kstest(
                        sample_np[..., i].reshape(-1),
                        sample_paddle.numpy()[..., i].reshape(-1),
                    )
                )

    @parameterize_func(
        [
            (1.0,),
            (0.1,),
        ]
    )
    def test_rsample(self, temperature):
        """Compare two samples from `rsample` method, one from scipy and another from paddle."""
        with paddle.fluid.dygraph.guard(self.place):
            sample_np = self.rv_np.sample(self.shape)
            rsample_paddle = self.rv_paddle.rsample(self.shape, temperature)

            self.assertEqual(list(rsample_paddle.shape), self.expected_shape)
            self.assertEqual(rsample_paddle.dtype, self.rv_paddle.probs.dtype)

            for i in range(len(self.probs)):
                self.assertTrue(
                    _kstest(
                        sample_np[..., i].reshape(-1),
                        (
                            _sigmoid(rsample_paddle.numpy()[..., i]) > 0.5
                        ).reshape(-1),
                        temperature,
                    )
                )


@place(DEVICES)
@parameterize_cls([TEST_CASE_NAME], ['BernoulliTestError'])
class BernoulliTestError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(self.place)

    @parameterize_func(
        [
            (-0.1, ValueError),
            (1.1, ValueError),
            (np.nan, ValueError),
            (-1j + 1, TypeError),
        ]
    )
    def test_bad_init(self, probs, error):
        with paddle.fluid.dygraph.guard(self.place):
            self.assertRaises(error, Bernoulli, probs)

    @parameterize_func(
        [
            (paddle.to_tensor([0.1, 0.2, 0.3]),),
        ]
    )
    def test_bad_broadcast(self, value):
        with paddle.fluid.dygraph.guard(self.place):
            rv = Bernoulli([0.3, 0.5])
            self.assertRaises(ValueError, rv.cdf, value)
            self.assertRaises(ValueError, rv.log_prob, value)
            self.assertRaises(ValueError, rv.prob, value)


if __name__ == '__main__':
    unittest.main()
