# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import parameterize
import scipy.stats
from distribution import config

import paddle
from paddle.distribution.student_t import StudentT


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df', 'loc', 'scale'),
    [
        (
            'one-dim',
            np.array(10.0).astype("float32"),
            np.array(1.0).astype("float32"),
            np.array(2.0).astype("float32"),
        ),
        (
            'multi-dim',
            parameterize.xrand((2, 1), dtype='float32', min=2.1, max=30),
            parameterize.xrand((2, 3), dtype='float32', min=-10, max=10),
            parameterize.xrand((2, 3), dtype='float32', min=0.1, max=10),
        ),
        (
            'multi-dim2',
            parameterize.xrand((2, 1), dtype='float64', min=2.1, max=30),
            parameterize.xrand((2, 3), dtype='float64', min=-10, max=10),
            parameterize.xrand((2, 3), dtype='float64', min=0.1, max=10),
        ),
    ],
)
class TestStudentT(unittest.TestCase):
    def setUp(self):
        self._dist = StudentT(
            df=paddle.to_tensor(self.df),
            loc=paddle.to_tensor(self.loc),
            scale=paddle.to_tensor(self.scale),
        )

    def test_mean(self):
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.df.dtype)
        np.testing.assert_allclose(
            mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(self.df.dtype)),
            atol=config.ATOL.get(str(self.df.dtype)),
        )

    def test_variance(self):
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.df.dtype)
        np.testing.assert_allclose(
            var,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.df.dtype)),
            atol=config.ATOL.get(str(self.df.dtype)),
        )

    def test_entropy(self):
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.df.dtype)
        np.testing.assert_allclose(
            entropy,
            self._np_entropy(),
            rtol=config.RTOL.get(str(self.df.dtype)),
            atol=config.ATOL.get(str(self.df.dtype)),
        )

    def test_sample(self):
        sample_shape = ()
        samples = self._dist.sample(sample_shape)
        self.assertEqual(
            tuple(samples.shape),
            sample_shape + self._dist.batch_shape + self._dist.event_shape,
        )

        sample_shape = (10000,)
        samples = self._dist.sample(sample_shape)
        sample_mean = samples.mean(axis=0)
        sample_variance = samples.var(axis=0)

        np.testing.assert_allclose(
            sample_mean, self._dist.mean, atol=0, rtol=0.20
        )
        np.testing.assert_allclose(
            sample_variance, self._dist.variance, atol=0, rtol=0.20
        )

    def _np_variance(self):
        if self.df.dtype == np.float32:
            df = self.df.astype("float64")
        else:
            df = self.df
        if self.loc.dtype == np.float32:
            loc = self.loc.astype("float64")
        else:
            loc = self.loc
        if self.scale.dtype == np.float32:
            scale = self.scale.astype("float64")
        else:
            scale = self.scale
        return scipy.stats.t.var(df, loc, scale)

    def _np_mean(self):
        if self.df.dtype == np.float32:
            df = self.df.astype("float64")
        else:
            df = self.df
        if self.loc.dtype == np.float32:
            loc = self.loc.astype("float64")
        else:
            loc = self.loc
        if self.scale.dtype == np.float32:
            scale = self.scale.astype("float64")
        else:
            scale = self.scale
        return scipy.stats.t.mean(df, loc, scale)

    def _np_entropy(self):
        if self.df.dtype == np.float32:
            df = self.df.astype("float64")
        else:
            df = self.df
        if self.loc.dtype == np.float32:
            loc = self.loc.astype("float64")
        else:
            loc = self.loc
        if self.scale.dtype == np.float32:
            scale = self.scale.astype("float64")
        else:
            scale = self.scale
        return scipy.stats.t.entropy(df, loc, scale)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df', 'loc', 'scale', 'value'),
    [
        (
            'one-dim',
            np.array(10.0).astype("float32"),
            np.array(0.0).astype("float32"),
            np.array(1.0).astype("float32"),
            np.array(3.3).astype("float32"),
        ),
        (
            'multi-dim',
            parameterize.xrand((2, 3), dtype='float32', min=2.1, max=30),
            parameterize.xrand((2, 3), dtype='float32', min=-10, max=10),
            parameterize.xrand((2, 3), dtype='float32', min=0.1, max=5),
            parameterize.xrand((2, 3), dtype='float32', min=-10, max=10),
        ),
        (
            'value-broadcast-shape',
            parameterize.xrand((2, 1), dtype='float64', min=2.1, max=30),
            parameterize.xrand((2, 1), dtype='float64', min=-10, max=10),
            parameterize.xrand((2, 1), dtype='float64', min=0.1, max=5),
            parameterize.xrand((2, 4), dtype='float64', min=-10, max=10),
        ),
    ],
)
class TestStudentTProbs(unittest.TestCase):
    def setUp(self):
        self._dist = StudentT(
            df=paddle.to_tensor(self.df),
            loc=paddle.to_tensor(self.loc),
            scale=paddle.to_tensor(self.scale),
        )

    def test_prob(self):
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
            scipy.stats.t.pdf(self.value, self.df, self.loc, self.scale),
            rtol=config.RTOL.get(str(self.df.dtype)),
            atol=config.ATOL.get(str(self.df.dtype)),
        )

    def test_log_prob(self):
        np.testing.assert_allclose(
            self._dist.log_prob(paddle.to_tensor(self.value)),
            scipy.stats.t.logpdf(self.value, self.df, self.loc, self.scale),
            rtol=config.RTOL.get(str(self.df.dtype)),
            atol=config.ATOL.get(str(self.df.dtype)),
        )


if __name__ == '__main__':
    unittest.main()
