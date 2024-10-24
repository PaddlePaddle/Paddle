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
from paddle.distribution import Cauchy
from paddle.distribution.kl import kl_divergence

np.random.seed(2023)
paddle.seed(2023)


def _kstest(samples_a, samples_b):
    """Uses the Kolmogorov-Smirnov test for goodness of fit."""
    _, p_value = scipy.stats.ks_2samp(samples_a, samples_b)
    return not p_value < 0.005


class CauchyNumpy(DistributionNumpy):
    def __init__(self, loc, scale):
        loc = np.array(loc)
        scale = np.array(scale)
        if str(loc.dtype) not in ['float32', 'float64']:
            self.dtype = 'float32'
        else:
            self.dtype = loc.dtype

        self.batch_shape = (loc + scale).shape

        self.loc = loc.astype(self.dtype)
        self.scale = scale.astype(self.dtype)
        self.rv = scipy.stats.cauchy(loc=loc, scale=scale)

    def sample(self, shape):
        shape = np.array(shape, dtype='int')
        if shape.ndim:
            shape = shape.tolist()
        else:
            shape = [shape.tolist()]
        return self.rv.rvs(size=shape + list(self.batch_shape))

    def log_prob(self, value):
        return self.rv.logpdf(value)

    def prob(self, value):
        return self.rv.pdf(value)

    def cdf(self, value):
        return self.rv.cdf(value)

    def entropy(self):
        return self.rv.entropy()

    def kl_divergence(self, other):
        a_loc = self.loc
        b_loc = other.loc

        a_scale = self.scale
        b_scale = other.scale

        t1 = np.log(np.power(a_scale + b_scale, 2) + np.power(a_loc - b_loc, 2))
        t2 = np.log(4 * a_scale * b_scale)
        return t1 - t2


class CauchyTest(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(self.place)
        with paddle.base.dygraph.guard(self.place):
            # just for convenience
            self.dtype = self.expected_dtype

            # init numpy with `dtype`
            self.init_numpy_data(self.loc, self.scale, self.dtype)

            # init paddle and check dtype convert.
            self.init_dynamic_data(
                self.loc, self.scale, self.default_dtype, self.dtype
            )

    def init_numpy_data(self, loc, scale, dtype):
        loc = np.array(loc).astype(dtype)
        scale = np.array(scale).astype(dtype)
        self.rv_np = CauchyNumpy(loc=loc, scale=scale)

    def init_dynamic_data(self, loc, scale, default_dtype, dtype):
        self.rv_paddle = Cauchy(loc=loc, scale=scale)
        self.assertTrue(
            dtype == convert_dtype(self.rv_paddle.loc.dtype),
            (dtype, self.rv_paddle.loc.dtype),
        )
        self.assertTrue(
            dtype == convert_dtype(self.rv_paddle.scale.dtype),
            (dtype, self.rv_paddle.scale.dtype),
        )


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'loc', 'scale', 'default_dtype', 'expected_dtype'),
    [
        # 0-D params
        (
            'params_0d_32_1',
            paddle.full((), 0.1),
            paddle.full((), 1.2),
            'float32',
            'float32',
        ),
        (
            'params_0d_32_2',
            paddle.full((), -1.2),
            paddle.full((), 2.3),
            'float32',
            'float32',
        ),
        (
            'params_0d_64_1',
            paddle.full((), 0.1, dtype='float64'),
            paddle.full((), 1.2, dtype='float64'),
            'float64',
            'float64',
        ),
        (
            'params_0d_64_2',
            paddle.full((), -1.2, dtype='float64'),
            paddle.full((), 2.3, dtype='float64'),
            'float64',
            'float64',
        ),
        # 1-D params
        ('params_float_1', 0.1, 1.2, 'float64', 'float32'),
        ('params_float_2', -1.2, 2.3, 'float64', 'float32'),
        (
            'params_tensor_32_1',
            paddle.to_tensor(0.1),
            paddle.to_tensor(1.2),
            'float32',
            'float32',
        ),
        (
            'params_tensor_32_2',
            paddle.to_tensor(-1.2),
            paddle.to_tensor(2.3),
            'float32',
            'float32',
        ),
        (
            'params_tensor_64_1',
            paddle.to_tensor(0.1, dtype='float64'),
            paddle.to_tensor(1.2, dtype='float64'),
            'float64',
            'float64',
        ),
        (
            'params_tensor_64_2',
            paddle.to_tensor(-1.2, dtype='float64'),
            paddle.to_tensor(2.3, dtype='float64'),
            'float64',
            'float64',
        ),
        (
            'params_tensor_list',
            paddle.to_tensor([0.1]),
            paddle.to_tensor([1.2]),
            'float32',
            'float32',
        ),
        (
            'params_tensor_tuple',
            paddle.to_tensor((0.1,)),
            paddle.to_tensor((1.2,)),
            'float32',
            'float32',
        ),
        # N-D params
        (
            'params_0d_1d_1',
            paddle.full((), 0.1),
            paddle.full((1,), 1.2),
            'float32',
            'float32',
        ),
        (
            'params_0d_1d_2',
            paddle.full((), 0.1),
            paddle.to_tensor(1.2),
            'float32',
            'float32',
        ),
        (
            'params_1d_0d_1',
            paddle.full((1,), 0.1),
            paddle.full((), 1.2),
            'float32',
            'float32',
        ),
        (
            'params_1d_0d_2',
            paddle.to_tensor(0.1),
            paddle.full((), 1.2),
            'float32',
            'float32',
        ),
        (
            'params_0d_3d',
            paddle.full((), 0.1),
            paddle.to_tensor([1.1, 2.2, 3.3]),
            'float32',
            'float32',
        ),
        (
            'params_3d_0d',
            paddle.to_tensor([0.1, -0.2, 0.3]),
            paddle.full((), 1.2),
            'float32',
            'float32',
        ),
        (
            'params_1d_3d',
            paddle.full((1,), 0.1),
            paddle.to_tensor([1.1, 2.2, 3.3]),
            'float32',
            'float32',
        ),
        (
            'params_3d_1d',
            paddle.to_tensor([0.1, -0.2, 0.3]),
            paddle.full((1,), 1.2),
            'float32',
            'float32',
        ),
        (
            'params_3d_3d',
            paddle.to_tensor([0.1, -0.2, 0.3]),
            paddle.to_tensor([1.1, 2.2, 3.3]),
            'float32',
            'float32',
        ),
    ],
)
class CauchyTestFeature(CauchyTest):
    @parameterize_func(
        [
            (paddle.to_tensor([-0.3]),),
            (paddle.to_tensor([0.3]),),
            (paddle.to_tensor([1.3]),),
            (paddle.to_tensor([5.3]),),
            (paddle.to_tensor(0.3, dtype='float64'),),
        ]
    )
    def test_log_prob(self, value):
        with paddle.base.dygraph.guard(self.place):
            if convert_dtype(value.dtype) == convert_dtype(
                self.rv_paddle.loc.dtype
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
            (paddle.to_tensor([-0.3]),),
            (paddle.to_tensor([0.3]),),
            (paddle.to_tensor([1.3]),),
            (paddle.to_tensor([5.3]),),
            (paddle.to_tensor(0.3, dtype='float64'),),
        ]
    )
    def test_prob(self, value):
        with paddle.base.dygraph.guard(self.place):
            if convert_dtype(value.dtype) == convert_dtype(
                self.rv_paddle.loc.dtype
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
            (paddle.to_tensor([-0.3]),),
            (paddle.to_tensor([0.3]),),
            (paddle.to_tensor([1.3]),),
            (paddle.to_tensor([5.3]),),
            (paddle.to_tensor(0.3, dtype='float64'),),
        ]
    )
    def test_cdf(self, value):
        with paddle.base.dygraph.guard(self.place):
            if convert_dtype(value.dtype) == convert_dtype(
                self.rv_paddle.loc.dtype
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

    @parameterize_func(
        [
            (0.6, 5.7),
            (-0.6, 5.7),
        ]
    )
    def test_kl_divergence(self, loc, scale):
        with paddle.base.dygraph.guard(self.place):
            # convert loc/scale to paddle's dtype(float32/float64)
            rv_paddle_other = Cauchy(
                loc=paddle.full((), loc, dtype=self.rv_paddle.loc.dtype),
                scale=paddle.full((), scale, dtype=self.rv_paddle.scale.dtype),
            )
            rv_np_other = CauchyNumpy(loc=loc, scale=scale)

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
        'loc',
        'scale',
        'default_dtype',
        'expected_dtype',
        'shape',
        'expected_shape',
    ),
    [
        # 0-D params
        (
            'params_0d_0d_sample_1d',
            paddle.full((), 0.1),
            paddle.full((), 1.2),
            'float32',
            'float32',
            [100],
            [100],
        ),
        (
            'params_0d_0d_sample_2d',
            paddle.full((), 0.1),
            paddle.full((), 1.2),
            'float32',
            'float32',
            [100, 1],
            [100, 1],
        ),
        (
            'params_0d_0d_sample_3d',
            paddle.full((), 0.1),
            paddle.full((), 1.2),
            'float32',
            'float32',
            [100, 2, 3],
            [100, 2, 3],
        ),
        # 1-D params
        (
            'params_1d_1d_sample_1d_float',
            0.1,
            1.2,
            'float64',
            'float32',
            paddle.to_tensor([100]),
            [100],
        ),
        (
            'params_1d_1d_sample_1d_32',
            paddle.to_tensor([0.1]),
            paddle.to_tensor([1.2]),
            'float32',
            'float32',
            paddle.to_tensor([100]),
            [100, 1],
        ),
        (
            'params_1d_1d_sample_1d_64',
            paddle.to_tensor([0.1], dtype='float64'),
            paddle.to_tensor([1.2], dtype='float64'),
            'float64',
            'float64',
            paddle.to_tensor([100]),
            [100, 1],
        ),
        (
            'params_1d_1d_sample_2d',
            paddle.to_tensor([0.1]),
            paddle.to_tensor([1.2]),
            'float32',
            'float32',
            [100, 2],
            [100, 2, 1],
        ),
        (
            'params_1d_1d_sample_3d',
            paddle.to_tensor([0.1]),
            paddle.to_tensor([1.2]),
            'float32',
            'float32',
            [100, 2, 3],
            [100, 2, 3, 1],
        ),
        # N-D params
        (
            'params_0d_1d_sample_1d',
            paddle.full((), 0.3),
            paddle.to_tensor([1.2]),
            'float32',
            'float32',
            [100],
            [100, 1],
        ),
        (
            'params_1d_0d_sample_1d',
            paddle.to_tensor([0.3]),
            paddle.full((), 1.2),
            'float32',
            'float32',
            [100],
            [100, 1],
        ),
        (
            'params_0d_1d_sample_2d',
            paddle.full((), 0.3),
            paddle.to_tensor([1.2]),
            'float32',
            'float32',
            [100, 2],
            [100, 2, 1],
        ),
        (
            'params_1d_0d_sample_2d',
            paddle.to_tensor([0.3]),
            paddle.full((), 1.2),
            'float32',
            'float32',
            [100, 2],
            [100, 2, 1],
        ),
        (
            'params_1d_2d_sample_1d',
            paddle.to_tensor([0.3]),
            paddle.to_tensor((1.2, 2.3)),
            'float32',
            'float32',
            [100],
            [100, 2],
        ),
        (
            'params_2d_1d_sample_1d',
            paddle.to_tensor((0.3, -0.3)),
            paddle.to_tensor([1.2]),
            'float32',
            'float32',
            [100],
            [100, 2],
        ),
        (
            'params_2d_2d_sample_1d',
            paddle.to_tensor((0.3, -0.3)),
            paddle.to_tensor((1.2, 2.3)),
            'float32',
            'float32',
            [100],
            [100, 2],
        ),
        (
            'params_2d_2d_sample_2d',
            paddle.to_tensor((0.3, -0.3)),
            paddle.to_tensor((1.2, 2.3)),
            'float32',
            'float32',
            [100, 1],
            [100, 1, 2],
        ),
        (
            'params_1d_2d_sample_3d',
            paddle.to_tensor([0.3]),
            paddle.to_tensor((1.2, 2.3)),
            'float32',
            'float32',
            [100, 1, 2],
            [100, 1, 2, 2],
        ),
    ],
)
class CauchyTestSample(CauchyTest):
    def test_sample(self):
        with paddle.base.dygraph.guard(self.place):
            sample_np = self.rv_np.sample(self.shape)
            sample_paddle = self.rv_paddle.sample(self.shape)

            self.assertEqual(list(sample_paddle.shape), self.expected_shape)
            self.assertEqual(sample_paddle.dtype, self.rv_paddle.loc.dtype)

            if len(self.expected_shape) > len(self.shape):
                for i in range(self.expected_shape[-1]):
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

    def test_rsample(self):
        with paddle.base.dygraph.guard(self.place):
            sample_np = self.rv_np.sample(self.shape)
            rsample_paddle = self.rv_paddle.rsample(self.shape)

            self.assertEqual(list(rsample_paddle.shape), self.expected_shape)
            self.assertEqual(rsample_paddle.dtype, self.rv_paddle.loc.dtype)

            if len(self.expected_shape) > len(self.shape):
                for i in range(self.expected_shape[-1]):
                    self.assertTrue(
                        _kstest(
                            sample_np[..., i].reshape(-1),
                            rsample_paddle.numpy()[..., i].reshape(-1),
                        )
                    )
            else:
                self.assertTrue(
                    _kstest(
                        sample_np.reshape(-1),
                        rsample_paddle.numpy().reshape(-1),
                    )
                )

    def test_rsample_backpropagation(self):
        with paddle.base.dygraph.guard(self.place):
            self.rv_paddle.loc.stop_gradient = False
            self.rv_paddle.scale.stop_gradient = False
            rsample_paddle = self.rv_paddle.rsample(self.shape)
            grads = paddle.grad(
                [rsample_paddle], [self.rv_paddle.loc, self.rv_paddle.scale]
            )
            self.assertEqual(len(grads), 2)
            self.assertEqual(grads[0].dtype, self.rv_paddle.loc.dtype)
            self.assertEqual(grads[0].shape, self.rv_paddle.loc.shape)
            self.assertEqual(grads[1].dtype, self.rv_paddle.scale.dtype)
            self.assertEqual(grads[1].shape, self.rv_paddle.scale.shape)


@place(DEVICES)
@parameterize_cls([TEST_CASE_NAME], ['CauchyTestError'])
class CauchyTestError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(self.place)

    def test_bad_property(self):
        """For property like mean/variance/stddev which is undefined in math,
        we should raise `ValueError` instead of `NotImplementedError`.
        """
        with paddle.base.dygraph.guard(self.place):
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
        with paddle.base.dygraph.guard(self.place):
            rv = Cauchy(loc=0.0, scale=1.0)

            with self.assertRaises(TypeError):
                _ = rv.sample(shape)

            with self.assertRaises(TypeError):
                _ = rv.rsample(shape)

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
        with paddle.base.dygraph.guard(self.place):
            rv = Cauchy(loc=0.0, scale=1.0)

            with self.assertRaises(TypeError):
                _ = rv.log_prob(value)

            with self.assertRaises(TypeError):
                _ = rv.prob(value)

            with self.assertRaises(TypeError):
                _ = rv.cdf(value)

    @parameterize_func(
        [
            (np.array(1.0),),  # ndarray or other distribution
        ]
    )
    def test_bad_kl_other_type(self, other):
        with paddle.base.dygraph.guard(self.place):
            rv = Cauchy(loc=0.0, scale=1.0)

            with self.assertRaises(TypeError):
                _ = rv.kl_divergence(other)

    @parameterize_func(
        [
            (
                paddle.to_tensor([0.1, 0.2]),
                paddle.to_tensor([0.3, 0.4]),
                paddle.to_tensor([0.1, 0.2, 0.3]),
            ),
        ]
    )
    def test_bad_broadcast(self, loc, scale, value):
        with paddle.base.dygraph.guard(self.place):
            rv = Cauchy(loc=loc, scale=scale)
            self.assertRaises(ValueError, rv.cdf, value)
            self.assertRaises(ValueError, rv.log_prob, value)
            self.assertRaises(ValueError, rv.prob, value)


if __name__ == '__main__':
    unittest.main()
