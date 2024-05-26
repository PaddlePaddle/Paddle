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

import unittest

import numpy as np
import scipy.stats
from distribution.config import ATOL, DEVICES, RTOL
from parameterize import TEST_CASE_NAME, parameterize_cls, place, xrand

import paddle
from paddle.distribution import geometric

np.random.seed(2023)

paddle.enable_static()


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
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            # scale no need convert to tensor for scale input unittest
            probs = paddle.static.data(
                'probs', self.probs.shape, self.probs.dtype
            )

            self._paddle_geometric = geometric.Geometric(probs)
            self.feeds = {'probs': self.probs}

    def test_mean(self):
        with paddle.static.program_guard(self.program):
            [mean] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_geometric.mean],
            )
            np.testing.assert_allclose(
                mean,
                scipy.stats.geom.mean(self.probs, loc=-1),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_variance(self):
        with paddle.static.program_guard(self.program):
            [variance] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_geometric.variance],
            )
            np.testing.assert_allclose(
                variance,
                scipy.stats.geom.var(self.probs, loc=-1),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_stddev(self):
        with paddle.static.program_guard(self.program):
            [stddev] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_geometric.stddev],
            )
            np.testing.assert_allclose(
                stddev,
                scipy.stats.geom.std(self.probs, loc=-1),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_sample(self):
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_geometric.sample(),
            )
            self.assertTrue(
                data.shape == np.broadcast_arrays(self.probs)[0].shape
            )

    def test_rsample(self):
        with paddle.static.program_guard(self.program):
            [data] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=self._paddle_geometric.rsample(),
            )
            self.assertTrue(
                data.shape == np.broadcast_arrays(self.probs)[0].shape
            )

    def test_entropy(self):
        with paddle.static.program_guard(self.program):
            [entropy] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_geometric.entropy()],
            )
            np.testing.assert_allclose(
                entropy,
                scipy.stats.geom.entropy(self.probs, loc=-1),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_init_prob_type_error(self):
        with self.assertRaises(TypeError):
            paddle.distribution.geometric.Geometric([0.5])


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
        self.program = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program):
            probs = paddle.static.data(
                'probs', self.probs.shape, self.probs.dtype
            )

            self._paddle_geometric = geometric.Geometric(probs)
            self.feeds = {'probs': self.probs, 'value': self.value}

    def test_pmf(self):
        with paddle.static.program_guard(self.program):
            [pmf] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_geometric.pmf(self.value)],
            )
            np.testing.assert_allclose(
                pmf,
                scipy.stats.geom.pmf(self.value, self.probs, loc=-1),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_log_pmf(self):
        with paddle.static.program_guard(self.program):
            [log_pmf] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_geometric.log_pmf(self.value)],
            )
            np.testing.assert_allclose(
                log_pmf,
                scipy.stats.geom.logpmf(self.value, self.probs, loc=-1),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_cdf(self):
        with paddle.static.program_guard(self.program):
            [cdf] = self.executor.run(
                self.program,
                feed=self.feeds,
                fetch_list=[self._paddle_geometric.cdf(self.value)],
            )
            np.testing.assert_allclose(
                cdf,
                scipy.stats.geom.cdf(self.value, self.probs, loc=-1),
                rtol=RTOL.get(str(self.probs.dtype)),
                atol=ATOL.get(str(self.probs.dtype)),
            )

    def test_pmf_error(self):
        self.assertRaises(TypeError, self._paddle_geometric.pmf, [1, 2])

    def test_log_pmf_error(self):
        self.assertRaises(TypeError, self._paddle_geometric.log_pmf, [1, 2])

    def test_cdf_error(self):
        self.assertRaises(TypeError, self._paddle_geometric.cdf, [1, 2])


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
        paddle.enable_static()
        self.program_p = paddle.static.Program()
        self.program_q = paddle.static.Program()
        self.executor = paddle.static.Executor(self.place)
        with paddle.static.program_guard(self.program_p, self.program_q):
            probs_p = paddle.static.data(
                'probs1', self.probs1.shape, self.probs1.dtype
            )
            probs_q = paddle.static.data(
                'probs2', self.probs2.shape, self.probs2.dtype
            )

            self._paddle_geomP = geometric.Geometric(probs_p)
            self._paddle_geomQ = geometric.Geometric(probs_q)

            self.feeds = {
                'probs1': self.probs1,
                'probs2': self.probs2,
            }

    def test_kl_divergence(self):
        with paddle.static.program_guard(self.program_p, self.program_q):
            self.executor.run(self.program_q)
            [kl_diver] = self.executor.run(
                self.program_p,
                feed=self.feeds,
                fetch_list=[
                    self._paddle_geomP.kl_divergence(self._paddle_geomQ)
                ],
            )
            np.testing.assert_allclose(
                kl_diver,
                self._kl(),
                rtol=RTOL.get(str(self.probs1.dtype)),
                atol=ATOL.get(str(self.probs1.dtype)),
            )

    def test_kl1_error(self):
        self.assertRaises(
            TypeError,
            self._paddle_geomP.kl_divergence,
            paddle.distribution.beta.Beta,
        )

    def test_kl2_error(self):
        self.assertRaises(
            TypeError,
            self._paddle_geomQ.kl_divergence,
            paddle.distribution.beta.Beta,
        )

    def _kl(self):
        return self.probs1 * np.log(self.probs1 / self.probs2) + (
            1.0 - self.probs1
        ) * np.log((1.0 - self.probs1) / (1.0 - self.probs2))


if __name__ == '__main__':
    unittest.main()
