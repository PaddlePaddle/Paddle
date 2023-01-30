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

<<<<<<< HEAD
import config
import numpy as np
import parameterize
import scipy.stats

import paddle
=======
import numpy as np
import paddle
import scipy.stats

import config
import parameterize
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
<<<<<<< HEAD
    (parameterize.TEST_CASE_NAME, 'total_count', 'probs'),
    [
        ('one-dim', 10, parameterize.xrand((3,))),
        ('multi-dim', 9, parameterize.xrand((10, 20))),
        ('prob-sum-one', 10, np.array([0.5, 0.2, 0.3])),
        ('prob-sum-non-one', 10, np.array([2.0, 3.0, 5.0])),
    ],
)
class TestMultinomial(unittest.TestCase):
    def setUp(self):
        self._dist = paddle.distribution.Multinomial(
            total_count=self.total_count, probs=paddle.to_tensor(self.probs)
        )
=======
    (parameterize.TEST_CASE_NAME, 'total_count', 'probs'), [
        ('one-dim', 10, parameterize.xrand((3, ))),
        ('multi-dim', 9, parameterize.xrand((10, 20))),
        ('prob-sum-one', 10, np.array([0.5, 0.2, 0.3])),
        ('prob-sum-non-one', 10, np.array([2., 3., 5.])),
    ])
class TestMultinomial(unittest.TestCase):

    def setUp(self):
        self._dist = paddle.distribution.Multinomial(
            total_count=self.total_count, probs=paddle.to_tensor(self.probs))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_mean(self):
        mean = self._dist.mean
        self.assertEqual(mean.numpy().dtype, self.probs.dtype)
<<<<<<< HEAD
        np.testing.assert_allclose(
            mean,
            self._np_mean(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )
=======
        np.testing.assert_allclose(mean,
                                   self._np_mean(),
                                   rtol=config.RTOL.get(str(self.probs.dtype)),
                                   atol=config.ATOL.get(str(self.probs.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_variance(self):
        var = self._dist.variance
        self.assertEqual(var.numpy().dtype, self.probs.dtype)
<<<<<<< HEAD
        np.testing.assert_allclose(
            var,
            self._np_variance(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )
=======
        np.testing.assert_allclose(var,
                                   self._np_variance(),
                                   rtol=config.RTOL.get(str(self.probs.dtype)),
                                   atol=config.ATOL.get(str(self.probs.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_entropy(self):
        entropy = self._dist.entropy()
        self.assertEqual(entropy.numpy().dtype, self.probs.dtype)
<<<<<<< HEAD
        np.testing.assert_allclose(
            entropy,
            self._np_entropy(),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )
=======
        np.testing.assert_allclose(entropy,
                                   self._np_entropy(),
                                   rtol=config.RTOL.get(str(self.probs.dtype)),
                                   atol=config.ATOL.get(str(self.probs.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_sample(self):
        sample_shape = ()
        samples = self._dist.sample(sample_shape)
        self.assertEqual(samples.numpy().dtype, self.probs.dtype)
        self.assertEqual(
            tuple(samples.shape),
<<<<<<< HEAD
            sample_shape + self._dist.batch_shape + self._dist.event_shape,
        )

        sample_shape = (6,)
=======
            sample_shape + self._dist.batch_shape + self._dist.event_shape)

        sample_shape = (6, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        samples = self._dist.sample(sample_shape)
        self.assertEqual(samples.numpy().dtype, self.probs.dtype)
        self.assertEqual(
            tuple(samples.shape),
<<<<<<< HEAD
            sample_shape + self._dist.batch_shape + self._dist.event_shape,
        )
        self.assertTrue(
            np.all(samples.sum(-1).numpy() == self._dist.total_count)
        )

        sample_shape = (5000,)
=======
            sample_shape + self._dist.batch_shape + self._dist.event_shape)
        self.assertTrue(
            np.all(samples.sum(-1).numpy() == self._dist.total_count))

        sample_shape = (5000, )
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        samples = self._dist.sample(sample_shape)
        sample_mean = samples.mean(axis=0)
        # Tolerance value 0.2 is empirical value which is consistent with
        # TensorFlow
<<<<<<< HEAD
        np.testing.assert_allclose(
            sample_mean, self._dist.mean, atol=0, rtol=0.20
        )
=======
        np.testing.assert_allclose(sample_mean,
                                   self._dist.mean,
                                   atol=0,
                                   rtol=0.20)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def _np_variance(self):
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return self.total_count * probs * (1 - probs)

    def _np_mean(self):
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return self.total_count * probs

    def _np_entropy(self):
        probs = self.probs / self.probs.sum(-1, keepdims=True)
        return scipy.stats.multinomial.entropy(self.total_count, probs)


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'total_count', 'probs', 'value'),
    [
<<<<<<< HEAD
        (
            'value-float',
            10,
            np.array([0.2, 0.3, 0.5]),
            np.array([2.0, 3.0, 5.0]),
        ),
        ('value-int', 10, np.array([0.2, 0.3, 0.5]), np.array([2, 3, 5])),
        (
            'value-multi-dim',
            10,
            np.array([[0.3, 0.7], [0.5, 0.5]]),
            np.array([[4.0, 6], [8, 2]]),
        ),
        # ('value-sum-non-n', 10, np.array([0.5, 0.2, 0.3]), np.array([4,5,2])),
    ],
)
class TestMultinomialPmf(unittest.TestCase):
    def setUp(self):
        self._dist = paddle.distribution.Multinomial(
            total_count=self.total_count, probs=paddle.to_tensor(self.probs)
        )
=======
        ('value-float', 10, np.array([0.2, 0.3, 0.5]), np.array([2., 3., 5.])),
        ('value-int', 10, np.array([0.2, 0.3, 0.5]), np.array([2, 3, 5])),
        ('value-multi-dim', 10, np.array([[0.3, 0.7], [0.5, 0.5]
                                          ]), np.array([[4., 6], [8, 2]])),
        # ('value-sum-non-n', 10, np.array([0.5, 0.2, 0.3]), np.array([4,5,2])),
    ])
class TestMultinomialPmf(unittest.TestCase):

    def setUp(self):
        self._dist = paddle.distribution.Multinomial(
            total_count=self.total_count, probs=paddle.to_tensor(self.probs))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_prob(self):
        np.testing.assert_allclose(
            self._dist.prob(paddle.to_tensor(self.value)),
<<<<<<< HEAD
            scipy.stats.multinomial.pmf(
                self.value, self.total_count, self.probs
            ),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)),
        )
=======
            scipy.stats.multinomial.pmf(self.value, self.total_count,
                                        self.probs),
            rtol=config.RTOL.get(str(self.probs.dtype)),
            atol=config.ATOL.get(str(self.probs.dtype)))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
<<<<<<< HEAD
    (config.TEST_CASE_NAME, 'total_count', 'probs'),
    [
        ('total_count_le_one', 0, np.array([0.3, 0.7])),
        ('total_count_float', np.array([0.3, 0.7])),
        ('probs_zero_dim', np.array(0)),
    ],
)
class TestMultinomialException(unittest.TestCase):
    def TestInit(self):
        with self.assertRaises(ValueError):
            paddle.distribution.Multinomial(
                self.total_count, paddle.to_tensor(self.probs)
            )
=======
    (config.TEST_CASE_NAME, 'total_count', 'probs'), [
        ('total_count_le_one', 0, np.array([0.3, 0.7])),
        ('total_count_float', np.array([0.3, 0.7])),
        ('probs_zero_dim', np.array(0)),
    ])
class TestMultinomialException(unittest.TestCase):

    def TestInit(self):
        with self.assertRaises(ValueError):
            paddle.distribution.Multinomial(self.total_count,
                                            paddle.to_tensor(self.probs))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
