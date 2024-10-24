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
from parameterize import (
    TEST_CASE_NAME,
    parameterize_cls,
    parameterize_func,
)

import paddle
from paddle.distribution.student_t import StudentT


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df', 'loc', 'scale'),
    [
        (
            'one-dim',
            10.0,
            1.0,
            2.0,
        ),
        (
            'multi-dim',
            parameterize.xrand((2, 1), dtype='float32', min=4, max=30),
            parameterize.xrand((2, 3), dtype='float32', min=1, max=10),
            parameterize.xrand((2, 3), dtype='float32', min=0.1, max=3),
        ),
        (
            'multi-dim2',
            parameterize.xrand((2, 1), dtype='float64', min=4, max=30),
            parameterize.xrand((2, 3), dtype='float64', min=-10, max=-1),
            parameterize.xrand((2, 3), dtype='float64', min=0.1, max=3),
        ),
    ],
)
class TestStudentT(unittest.TestCase):
    def setUp(self):
        df = (
            self.df if isinstance(self.df, float) else paddle.to_tensor(self.df)
        )
        loc = (
            self.loc
            if isinstance(self.loc, float)
            else paddle.to_tensor(self.loc)
        )
        scale = (
            self.scale
            if isinstance(self.scale, float)
            else paddle.to_tensor(self.scale)
        )
        self._dist = StudentT(df, loc, scale)

    def test_mean(self):
        mean = self._dist.mean
        target_dtype = (
            "float32" if isinstance(self.df, float) else self.df.dtype
        )
        self.assertEqual(mean.numpy().dtype, target_dtype)
        np.testing.assert_allclose(
            mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(target_dtype)),
            atol=config.ATOL.get(str(target_dtype)),
        )

    def test_variance(self):
        var = self._dist.variance
        target_dtype = (
            "float32" if isinstance(self.df, float) else self.df.dtype
        )
        self.assertEqual(var.numpy().dtype, target_dtype)
        np.testing.assert_allclose(
            var,
            self._np_variance(),
            rtol=config.RTOL.get(str(target_dtype)),
            atol=config.ATOL.get(str(target_dtype)),
        )

    def test_entropy(self):
        entropy = self._dist.entropy()
        target_dtype = (
            "float32" if isinstance(self.df, float) else self.df.dtype
        )
        self.assertEqual(entropy.numpy().dtype, target_dtype)
        np.testing.assert_allclose(
            entropy,
            self._np_entropy(),
            rtol=config.RTOL.get(str(target_dtype)),
            atol=config.ATOL.get(str(target_dtype)),
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

        # Tolerance value 0.1 is empirical value which is consistent with
        # TensorFlow
        np.testing.assert_allclose(
            sample_mean, self._dist.mean, atol=0, rtol=0.10
        )
        # Tolerance value 0.1 is empirical value which is consistent with
        # TensorFlow
        np.testing.assert_allclose(
            sample_variance, self._dist.variance, atol=0, rtol=0.10
        )

    def _np_variance(self):
        if isinstance(self.df, np.ndarray) and self.df.dtype == np.float32:
            df = self.df.astype("float64")
        else:
            df = self.df
        if isinstance(self.loc, np.ndarray) and self.loc.dtype == np.float32:
            loc = self.loc.astype("float64")
        else:
            loc = self.loc
        if (
            isinstance(self.scale, np.ndarray)
            and self.scale.dtype == np.float32
        ):
            scale = self.scale.astype("float64")
        else:
            scale = self.scale
        return scipy.stats.t.var(df, loc, scale)

    def _np_mean(self):
        if isinstance(self.df, np.ndarray) and self.df.dtype == np.float32:
            df = self.df.astype("float64")
        else:
            df = self.df
        if isinstance(self.loc, np.ndarray) and self.loc.dtype == np.float32:
            loc = self.loc.astype("float64")
        else:
            loc = self.loc
        if (
            isinstance(self.scale, np.ndarray)
            and self.scale.dtype == np.float32
        ):
            scale = self.scale.astype("float64")
        else:
            scale = self.scale
        return scipy.stats.t.mean(df, loc, scale)

    def _np_entropy(self):
        if isinstance(self.df, np.ndarray) and self.df.dtype == np.float32:
            df = self.df.astype("float64")
        else:
            df = self.df
        if isinstance(self.loc, np.ndarray) and self.loc.dtype == np.float32:
            loc = self.loc.astype("float64")
        else:
            loc = self.loc
        if (
            isinstance(self.scale, np.ndarray)
            and self.scale.dtype == np.float32
        ):
            scale = self.scale.astype("float64")
        else:
            scale = self.scale
        return scipy.stats.t.entropy(df, loc, scale)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df', 'loc', 'scale'),
    [
        (
            'float-tensor',
            10.0,
            paddle.to_tensor(1.0),
            2.0,
        ),
        (
            'float-tensor1',
            10.0,
            parameterize.xrand((2, 3), dtype='float32', min=1, max=10),
            2.0,
        ),
        (
            'float-tensor2',
            parameterize.xrand((2, 1), dtype='float64', min=4, max=30),
            parameterize.xrand((2, 3), dtype='float64', min=1, max=10),
            2.0,
        ),
        (
            'float-tensor3',
            parameterize.xrand((2, 1), dtype='float64', min=4, max=30),
            1.0,
            parameterize.xrand((2, 1), dtype='float64', min=0.1, max=3),
        ),
        (
            'float-tensor4',
            5.0,
            parameterize.xrand((2, 1), dtype='float32', min=-1, max=-10),
            parameterize.xrand((2, 3), dtype='float32', min=0.1, max=3),
        ),
    ],
)
class TestStudentT2(TestStudentT):
    def setUp(self):
        self._dist = StudentT(self.df, self.loc, self.scale)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df', 'loc', 'scale', 'value'),
    [
        (
            'one-dim',
            10.0,
            0.0,
            1.0,
            np.array(3.3).astype("float32"),
        ),
        (
            'value-broadcast-shape',
            parameterize.xrand((2, 1), dtype='float64', min=4, max=30),
            parameterize.xrand((2, 1), dtype='float64', min=-10, max=10),
            parameterize.xrand((2, 1), dtype='float64', min=0.1, max=5),
            parameterize.xrand((2, 4), dtype='float64', min=-10, max=10),
        ),
    ],
)
class TestStudentTProbs(unittest.TestCase):
    def setUp(self):
        df = (
            self.df if isinstance(self.df, float) else paddle.to_tensor(self.df)
        )
        loc = (
            self.loc
            if isinstance(self.loc, float)
            else paddle.to_tensor(self.loc)
        )
        scale = (
            self.scale
            if isinstance(self.scale, float)
            else paddle.to_tensor(self.scale)
        )
        self._dist = StudentT(df, loc, scale)

    def test_prob(self):
        target_dtype = (
            "float32" if isinstance(self.df, float) else self.df.dtype
        )
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
            scipy.stats.t.pdf(self.value, self.df, self.loc, self.scale),
            rtol=config.RTOL.get(str(target_dtype)),
            atol=config.ATOL.get(str(target_dtype)),
        )

    def test_log_prob(self):
        target_dtype = (
            "float32" if isinstance(self.df, float) else self.df.dtype
        )
        np.testing.assert_allclose(
            self._dist.log_prob(paddle.to_tensor(self.value)),
            scipy.stats.t.logpdf(self.value, self.df, self.loc, self.scale),
            rtol=config.RTOL.get(str(target_dtype)),
            atol=config.ATOL.get(str(target_dtype)),
        )


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'df', 'loc', 'scale', 'value'),
    [
        (
            'float-tensor1',
            10.0,
            parameterize.xrand((2, 1), dtype='float32', min=-10, max=10),
            1.0,
            np.array(3.3).astype("float32"),
        ),
        (
            'float-tensor2',
            parameterize.xrand((2, 1), dtype='float64', min=4, max=30),
            1.0,
            parameterize.xrand((2, 1), dtype='float64', min=0.1, max=5),
            parameterize.xrand((2, 4), dtype='float64', min=-10, max=10),
        ),
    ],
)
class TestStudentTProbs2(TestStudentTProbs):
    def setUp(self):
        self._dist = StudentT(self.df, self.loc, self.scale)


@parameterize.place(config.DEVICES)
@parameterize_cls([TEST_CASE_NAME], ['StudentTTestError'])
class StudentTTestError(unittest.TestCase):
    def setUp(self):
        paddle.disable_static(self.place)

    @parameterize_func(
        [
            (-5.0, 0.0, 1.0, ValueError),  # negative df
            (5.0, 0.0, -1.0, ValueError),  # negative scale
        ]
    )
    def test_bad_parameter(self, df, loc, scale, error):
        with paddle.base.dygraph.guard(self.place):
            self.assertRaises(error, StudentT, df, loc, scale)

    @parameterize_func([(10,)])  # not sequence object sample shape
    def test_bad_sample_shape(self, shape):
        with paddle.base.dygraph.guard(self.place):
            t = StudentT(5.0, 0.0, 1.0)
            self.assertRaises(TypeError, t.sample, shape)


if __name__ == '__main__':
    unittest.main()
