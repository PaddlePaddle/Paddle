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
from config import ATOL, DEVICES, RTOL
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand

import paddle
from paddle.distribution.geometric import Geometric
from paddle.distribution.kl import kl_divergence

np.random.seed(2023)


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'probs'),
    [
        ('one-dim', xrand((2,), dtype='float32', min=0.0, max=1.0)),
        ('multi-dim', xrand((2, 3), dtype='float32', min=0.0, max=1.0)),
    ],
)
class TestGeometric(unittest.TestCase):
    def setUp(self):
        probs = self.probs
        if not isinstance(self.probs, numbers.Real):
            probs = paddle.to_tensor(self.probs, dtype=paddle.float32)

        self._paddle_geom = paddle.distribution.geometric.Geometric(probs)

    def test_mean(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.mean,
                scipy.stats.geom.mean(self.probs),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )

    def test_variance(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.variance,
                scipy.stats.geom.var(self.probs),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )

    def test_stddev(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.stddev,
                scipy.stats.geom.std(self.probs),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )

    def test_entropy(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.entropy(),
                scipy.stats.geom.entropy(self.probs),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )

    def test_sample_shape(self):
        cases = [
            {
                'input': (),
                'expect': ()
                + tuple(paddle.squeeze(self._paddle_geom.probs).shape),
            },
            {
                'input': (4, 2),
                'expect': (4, 2)
                + tuple(paddle.squeeze(self._paddle_geom.probs).shape),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_geom.sample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_sample(self):
        sample_shape = (30000,)
        samples = self._paddle_geom.sample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.probs.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.geom.mean(self.probs),
            rtol=0.7,
            atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.geom.var(self.probs),
            rtol=0.7,
            atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
        )

    def test_rsample_shape(self):
        cases = [
            {
                'input': (),
                'expect': ()
                + tuple(paddle.squeeze(self._paddle_geom.probs).shape),
            },
            {
                'input': (2, 5),
                'expect': (2, 5)
                + tuple(paddle.squeeze(self._paddle_geom.probs).shape),
            },
        ]
        for case in cases:
            self.assertTrue(
                tuple(self._paddle_geom.rsample(case.get('input')).shape)
                == case.get('expect')
            )

    def test_rsample(self):

        sample_shape = (30000,)
        samples = self._paddle_geom.rsample(sample_shape)
        sample_values = samples.numpy()
        self.assertEqual(sample_values.dtype, self.probs.dtype)

        np.testing.assert_allclose(
            sample_values.mean(axis=0),
            scipy.stats.geom.mean(self.probs),
            rtol=0.7,
            atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
        )
        np.testing.assert_allclose(
            sample_values.var(axis=0),
            scipy.stats.geom.var(self.probs),
            rtol=0.7,
            atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
        )


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'probs', 'value'),
    [
        ('one-dim', xrand((2,), dtype='float32', min=0.0, max=1.0), 5),
        ('mult-dim', xrand((2, 2), dtype='float32', min=0.0, max=1.0), 5),
        ('mult-dim', xrand((2, 2, 2), dtype='float32', min=0.0, max=1.0), 5),
    ],
)
class TestGumbelPMF(unittest.TestCase):
    def setUp(self):
        self._paddle_geom = paddle.distribution.geometric.Geometric(
            probs=paddle.to_tensor(self.probs)
        )

    def test_pmf(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.pmf(self.value),
                scipy.stats.geom.pmf(self.value, self.probs),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_log_pmf(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.log_pmf(self.value),
                scipy.stats.geom.logpmf(self.value, self.probs),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_cdf(self):
        with paddle.fluid.dygraph.guard(self.place):
            np.testing.assert_allclose(
                self._paddle_geom.cdf(self.value),
                scipy.stats.geom.cdf(self.value, self.probs),
                rtol=RTOL.get(str(self._paddle_geom.probs.numpy().dtype)),
                atol=ATOL.get(str(self._paddle_geom.probs.numpy().dtype)),
            )


@place(DEVICES)
@parameterize_cls(
    (TEST_CASE_NAME, 'probs1', 'probs2'),
    [
        (
            'one-dim',
            xrand((2,), dtype='float32', min=0.0, max=1.0),
            xrand((2,), dtype='float32', min=0.0, max=1.0),
        ),
        (
            'multi-dim',
            xrand((2, 2), dtype='float32', min=0.0, max=1.0),
            xrand((2, 2), dtype='float32', min=0.0, max=1.0),
            xrand((2, 2, 5), dtype='float32', min=0.0, max=1.0),
            xrand((2, 2, 5, 2), dtype='float32', min=0.0, max=1.0),
        ),
    ],
)
class TestGeometricKL(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self._geometric1 = Geometric(probs=paddle.to_tensor(self.probs1))
        self._geometric2 = Geometric(probs=paddle.to_tensor(self.probs2))

    def test_kl_divergence(self):
        np.testing.assert_allclose(
            kl_divergence(self._geometric1, self._geometric2),
            self._kl(),
            rtol=0.3,
            atol=ATOL.get(str(self._geometric1.probs.numpy().dtype)),
        )

    def _kl(self):
        temp = np.log(self.probs1, self.probs2)
        kl_diff = self.probs1 * np.abs(temp)

        return np.sum(kl_diff, axis=-1)


if __name__ == '__main__':
    unittest.main()
