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
from distribution.config import ATOL, DEVICES, RTOL
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand

import paddle
from paddle.distribution import geometric, kl
from paddle.nn.functional import log_softmax

np.random.seed(2023)


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'probs'),
    [
        (
            'one-dim',
            xrand(
                (2,),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
                max=1.0,
            ),
        ),
        (
            'multi-dim',
            xrand(
                (2, 3),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
                max=1.0,
            ),
        ),
    ],
)
class TestGeometric(unittest.TestCase):
    def setUp(self):
        probs = self.probs
        if not isinstance(self.probs, numbers.Real):
            probs = paddle.to_tensor(self.probs, dtype=paddle.float32)

        self._paddle_geom = geometric.Geometric(probs)

    def test_mean(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.mean,
                scipy.stats.geom.mean(self.probs, loc=-1),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )

    def test_variance(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.variance,
                scipy.stats.geom.var(self.probs, loc=-1),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )

    def test_stddev(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.stddev,
                scipy.stats.geom.std(self.probs, loc=-1),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )

    def test_entropy(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.entropy(),
                scipy.stats.geom.entropy(self.probs, loc=-1),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )

    def test_init_prob_type_error(self):
        with self.assertRaises(TypeError):
            paddle.distribution.geometric.Geometric([2])

    def test_sample_shape(self):
        cases = [
            {
                'input': (),
                'expect': tuple(paddle.squeeze(self._paddle_geom.probs).shape),
            },
            {
                'input': (4, 2),
                'expect': (
                    4,
                    2,
                    *paddle.squeeze(self._paddle_geom.probs).shape,
                ),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_geom.sample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_sample(self):
        sample_shape = (100000,)
        samples = self._paddle_geom.sample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.probs.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.geom.mean(self.probs, loc=-1),
            rtol=0.1,
            atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.geom.var(self.probs, loc=-1),
            rtol=0.1,
            atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
        )

    def test_rsample_shape(self):
        cases = [
            {
                'input': (),
                'expect': tuple(paddle.squeeze(self._paddle_geom.probs).shape),
            },
            {
                'input': (2, 5),
                'expect': (
                    2,
                    5,
                    *paddle.squeeze(self._paddle_geom.probs).shape,
                ),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_geom.rsample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_rsample(self):
        sample_shape = (100000,)
        samples = self._paddle_geom.rsample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.probs.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.geom.mean(self.probs, loc=-1),
            rtol=0.1,
            atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.geom.var(self.probs, loc=-1),
            rtol=0.1,
            atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
        )

    def test_back_rsample(self):
        sample_shape = (100000,)

        with paddle.base.dygraph.guard(self.place):
            self._paddle_geom.probs.stop_gradient = False

            rs_value = self._paddle_geom.rsample(sample_shape)
            softmax_rs = log_softmax(rs_value)
            grads = paddle.grad([softmax_rs], [self._paddle_geom.probs])

            self.assertEqual(len(grads), 1)
            self.assertEqual(grads[0].dtype, self._paddle_geom.probs.dtype)
            self.assertEqual(grads[0].shape, self._paddle_geom.probs.shape)


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'probs', 'value'),
    [
        (
            'one-dim',
            xrand(
                (2,),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
                max=1.0,
            ),
            5,
        ),
        (
            'mult-dim',
            xrand(
                (2, 2),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
                max=1.0,
            ),
            5,
        ),
        (
            'mult-dim',
            xrand(
                (2, 2, 2),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
                max=1.0,
            ),
            5,
        ),
    ],
)
class TestGeometricPMF(unittest.TestCase):
    def setUp(self):
        self._paddle_geom = geometric.Geometric(
            probs=paddle.to_tensor(self.probs)
        )

    def test_pmf(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.pmf(self.value),
                scipy.stats.geom.pmf(self.value, self.probs, loc=-1),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_log_pmf(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.log_pmf(self.value),
                scipy.stats.geom.logpmf(self.value, self.probs, loc=-1),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_cdf(self):
        with paddle.base.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.cdf(self.value),
                scipy.stats.geom.cdf(self.value, self.probs, loc=-1),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )

    def test_pmf_error(self):
        self.assertRaises(TypeError, self._paddle_geom.pmf, [1, 2])

    def test_log_pmf_error(self):
        self.assertRaises(TypeError, self._paddle_geom.log_pmf, [1, 2])

    def test_cdf_error(self):
        self.assertRaises(TypeError, self._paddle_geom.cdf, [1, 2])


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'probs1', 'probs2'),
    [
        (
            'one-dim',
            xrand(
                (2,),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
                max=1.0,
            ),
            xrand(
                (2,),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
                max=1.0,
            ),
        ),
        (
            'multi-dim',
            xrand(
                (2, 2),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
                max=1.0,
            ),
            xrand(
                (2, 2),
                dtype='float32',
                min=np.finfo(dtype='float32').tiny,
                max=1.0,
            ),
        ),
    ],
)
class TestGeometricKL(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self._geometric1 = geometric.Geometric(
            probs=paddle.to_tensor(self.probs1)
        )
        self._geometric2 = geometric.Geometric(
            probs=paddle.to_tensor(self.probs2)
        )

    def test_kl_divergence(self):
        np.testing.assert_allclose(
            kl.kl_divergence(self._geometric1, self._geometric2),
            self._kl(),
            rtol=RTOL.get(str(self._geometric1.probs.numpy().dtype)),
            atol=ATOL.get(str(self._geometric1.probs.numpy().dtype)),
        )

    def test_kl1_error(self):
        self.assertRaises(
            TypeError,
            self._geometric1.kl_divergence,
            paddle.distribution.beta.Beta,
        )

    def test_kl2_error(self):
        self.assertRaises(
            TypeError,
            self._geometric2.kl_divergence,
            paddle.distribution.beta.Beta,
        )

    def _kl(self):
        return self.probs1 * np.log(self.probs1 / self.probs2) + (
            1.0 - self.probs1
        ) * np.log((1.0 - self.probs1) / (1.0 - self.probs2))


if __name__ == '__main__':
    unittest.main()
