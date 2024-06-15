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
from distribution.config import ATOL, DEVICES, RTOL
from parameterize import (
    TEST_CASE_NAME,
    parameterize_cls,
    parameterize_func,
    place,
)
from test_distribution import DistributionNumpy

import paddle
from paddle.base.data_feeder import convert_dtype
from paddle.distribution import Bernoulli
from paddle.distribution.kl import kl_divergence

np.random.seed(2023)
paddle.seed(2023)

# Smallest representable number.
EPS = {
    'float32': np.finfo('float32').eps,
    'float64': np.finfo('float64').eps,
}


def _clip_probs_ndarray(probs, dtype):
    """Clip probs from [0, 1] to (0, 1) with ``eps``"""
    eps = EPS.get(dtype)
    return np.clip(probs, a_min=eps, a_max=1 - eps).astype(dtype)


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

        self.batch_shape = np.shape(probs)

        self.probs = _clip_probs_ndarray(
            np.array(probs, dtype=self.dtype), str(self.dtype)
        )
        self.logits = self._probs_to_logits(self.probs, is_binary=True)

        self.rv = scipy.stats.bernoulli(self.probs.astype('float64'))

    @property
    def mean(self):
        return self.rv.mean().astype(self.dtype)

    @property
    def variance(self):
        return self.rv.var().astype(self.dtype)

    def sample(self, shape):
        shape = np.array(shape, dtype='int')
        if shape.ndim:
            shape = shape.tolist()
        else:
            shape = [shape.tolist()]
        return self.rv.rvs(size=shape + list(self.batch_shape)).astype(
            self.dtype
        )

    def log_prob(self, value):
        return self.rv.logpmf(value).astype(self.dtype)

    def prob(self, value):
        return self.rv.pmf(value).astype(self.dtype)

    def cdf(self, value):
        return self.rv.cdf(value).astype(self.dtype)

    def entropy(self):
        return (
            np.maximum(
                self.logits,
                0,
            )
            - self.logits * self.probs
            + np.log(1 + np.exp(-np.abs(self.logits)))
        ).astype(self.dtype)

    def kl_divergence(self, other):
        """
        .. math::

            KL[a || b] = Pa * Log[Pa / Pb] + (1 - Pa) * Log[(1 - Pa) / (1 - Pb)]
        """
        p_a = self.probs
        p_b = other.probs
        return (
            p_a * np.log(p_a / p_b) + (1 - p_a) * np.log((1 - p_a) / (1 - p_b))
        ).astype(self.dtype)

    def _probs_to_logits(self, probs, is_binary=False):
        return (
            (np.log(probs) - np.log1p(-probs)) if is_binary else np.log(probs)
        ).astype(self.dtype)


class BernoulliTest(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(self.place)
        with paddle.base.dygraph.guard(self.place):
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
        self.rv_paddle = Bernoulli(probs)
        self.assertTrue(
            dtype == convert_dtype(self.rv_paddle.probs.dtype),
            (dtype, self.rv_paddle.probs.dtype),
        )


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'probs', 'default_dtype', 'expected_dtype'),
    [
        # 0-D probs
        ('probs_00_32', paddle.full((), 0.0), 'float32', 'float32'),
        ('probs_03_32', paddle.full((), 0.3), 'float32', 'float32'),
        ('probs_10_32', paddle.full((), 1.0), 'float32', 'float32'),
        (
            'probs_00_64',
            paddle.full((), 0.0, dtype='float64'),
            'float64',
            'float64',
        ),
        (
            'probs_03_64',
            paddle.full((), 0.3, dtype='float64'),
            'float64',
            'float64',
        ),
        (
            'probs_10_64',
            paddle.full((), 1.0, dtype='float64'),
            'float64',
            'float64',
        ),
        # 1-D probs
        ('probs_00', 0.0, 'float64', 'float32'),
        ('probs_03', 0.3, 'float64', 'float32'),
        ('probs_10', 1.0, 'float64', 'float32'),
        ('probs_tensor_03_32', paddle.to_tensor([0.3]), 'float32', 'float32'),
        (
            'probs_tensor_03_64',
            paddle.to_tensor([0.3], dtype='float64'),
            'float64',
            'float64',
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
        (
            'probs_tensor_0305',
            paddle.to_tensor((0.3, 0.5)),
            'float32',
            'float32',
        ),
        (
            'probs_tensor_03050104',
            paddle.to_tensor(((0.3, 0.5), (0.1, 0.4))),
            'float32',
            'float32',
        ),
    ],
)
class BernoulliTestFeature(BernoulliTest):
    def test_mean(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self.rv_paddle.mean,
                self.rv_np.mean,
                rtol=RTOL.get(self.dtype),
                atol=ATOL.get(self.dtype),
            )

    def test_variance(self):
        with paddle.base.dygraph.guard(self.place):
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
                    [0.0],
                ),
            ),
            (paddle.to_tensor([1.0]),),
            (paddle.to_tensor([0.0], dtype='float64'),),
        ]
    )
    def test_log_prob(self, value):
        with paddle.base.dygraph.guard(self.place):
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
            (paddle.to_tensor([0.0]),),
            (paddle.to_tensor([1.0]),),
            (paddle.to_tensor([0.0], dtype='float64'),),
        ]
    )
    def test_prob(self, value):
        with paddle.base.dygraph.guard(self.place):
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
            (paddle.to_tensor([0.0]),),
            (paddle.to_tensor([0.3]),),
            (paddle.to_tensor([0.7]),),
            (paddle.to_tensor([1.0]),),
            (paddle.to_tensor([0.0], dtype='float64'),),
        ]
    )
    def test_cdf(self, value):
        with paddle.base.dygraph.guard(self.place):
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
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self.rv_paddle.entropy(),
                self.rv_np.entropy(),
                rtol=RTOL.get(self.dtype),
                atol=ATOL.get(self.dtype),
            )

    def test_kl_divergence(self):
        with paddle.base.dygraph.guard(self.place):
            other_probs = paddle.to_tensor([0.9], dtype=self.dtype)

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
        # 0-D probs
        (
            'probs_0d_1d',
            paddle.full((), 0.3),
            'float32',
            'float32',
            [
                100,
            ],
            [
                100,
            ],
        ),
        (
            'probs_0d_2d',
            paddle.full((), 0.3),
            'float32',
            'float32',
            [100, 1],
            [100, 1],
        ),
        (
            'probs_0d_3d',
            paddle.full((), 0.3),
            'float32',
            'float32',
            [100, 2, 3],
            [100, 2, 3],
        ),
        # 1-D probs
        (
            'probs_1d_1d_32',
            paddle.to_tensor([0.3]),
            'float32',
            'float32',
            [
                100,
            ],
            [100, 1],
        ),
        (
            'probs_1d_1d_64',
            paddle.to_tensor([0.3], dtype='float64'),
            'float64',
            'float64',
            paddle.to_tensor(
                [
                    100,
                ]
            ),
            [100, 1],
        ),
        (
            'probs_1d_2d',
            paddle.to_tensor([0.3]),
            'float32',
            'float32',
            [100, 2],
            [100, 2, 1],
        ),
        (
            'probs_1d_3d',
            paddle.to_tensor([0.3]),
            'float32',
            'float32',
            [100, 2, 3],
            [100, 2, 3, 1],
        ),
        # N-D probs
        (
            'probs_2d_1d',
            paddle.to_tensor((0.3, 0.5)),
            'float32',
            'float32',
            [
                100,
            ],
            [100, 2],
        ),
        (
            'probs_2d_2d',
            paddle.to_tensor((0.3, 0.5)),
            'float32',
            'float32',
            [100, 3],
            [100, 3, 2],
        ),
        (
            'probs_2d_3d',
            paddle.to_tensor((0.3, 0.5)),
            'float32',
            'float32',
            [100, 4, 3],
            [100, 4, 3, 2],
        ),
    ],
)
class BernoulliTestSample(BernoulliTest):
    def test_sample(self):
        with paddle.base.dygraph.guard(self.place):
            sample_np = self.rv_np.sample(self.shape)
            sample_paddle = self.rv_paddle.sample(self.shape)

            self.assertEqual(list(sample_paddle.shape), self.expected_shape)
            self.assertEqual(sample_paddle.dtype, self.rv_paddle.probs.dtype)

            if self.probs.ndim:
                for i in range(len(self.probs)):
                    self.assertTrue(
                        _kstest(
                            sample_np[..., i].reshape(-1),
                            sample_paddle.numpy()[..., i].reshape(-1),
                        )
                    )
            else:
                self.assertTrue(
                    _kstest(
                        sample_np.reshape(-1),
                        sample_paddle.numpy().reshape(-1),
                    )
                )

    @parameterize_func(
        [
            (1.0,),
            (0.1,),
        ]
    )
    def test_rsample(self, temperature):
        """Compare two samples from `rsample` method, one from scipy `sample` and another from paddle `rsample`."""
        with paddle.base.dygraph.guard(self.place):
            sample_np = self.rv_np.sample(self.shape)
            rsample_paddle = self.rv_paddle.rsample(self.shape, temperature)

            self.assertEqual(list(rsample_paddle.shape), self.expected_shape)
            self.assertEqual(rsample_paddle.dtype, self.rv_paddle.probs.dtype)

            if self.probs.ndim:
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
            else:
                self.assertTrue(
                    _kstest(
                        sample_np.reshape(-1),
                        (_sigmoid(rsample_paddle.numpy()) > 0.5).reshape(-1),
                        temperature,
                    )
                )

    def test_rsample_backpropagation(self):
        with paddle.base.dygraph.guard(self.place):
            self.rv_paddle.probs.stop_gradient = False
            rsample_paddle = self.rv_paddle.rsample(self.shape)
            rsample_paddle = paddle.nn.functional.sigmoid(rsample_paddle)
            grads = paddle.grad([rsample_paddle], [self.rv_paddle.probs])
            self.assertEqual(len(grads), 1)
            self.assertEqual(grads[0].dtype, self.rv_paddle.probs.dtype)
            self.assertEqual(grads[0].shape, self.rv_paddle.probs.shape)


@place(DEVICES)
@parameterize_cls([TEST_CASE_NAME], ['BernoulliTestError'])
class BernoulliTestError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(self.place)

    @parameterize_func(
        [
            (
                [0.3, 0.5],
                paddle.to_tensor([0.1, 0.2, 0.3]),
            ),
        ]
    )
    def test_bad_broadcast(self, probs, value):
        with paddle.base.dygraph.guard(self.place):
            rv = Bernoulli(probs)
            self.assertRaises(ValueError, rv.cdf, value)
            self.assertRaises(ValueError, rv.log_prob, value)
            self.assertRaises(ValueError, rv.prob, value)


if __name__ == '__main__':
    unittest.main()
