#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import unittest

import numpy as np
import paddle
from paddle import fluid
from paddle.distribution import *
from paddle.fluid import layers

import config
import parameterize

paddle.enable_static()


class DistributionNumpy():
    def sample(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def kl_divergence(self, other):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def probs(self, value):
        raise NotImplementedError


class DistributionTestName(unittest.TestCase):
    def get_prefix(self, string):
        return (string.split('.')[0])

    def test_normal_name(self):
        name = 'test_normal'
        normal1 = Normal(0.0, 1.0, name=name)
        self.assertEqual(normal1.name, name)

        normal2 = Normal(0.0, 1.0)
        self.assertEqual(normal2.name, 'Normal')

        paddle.enable_static()

        sample = normal1.sample([2])
        self.assertEqual(self.get_prefix(sample.name), name + '_sample')

        entropy = normal1.entropy()
        self.assertEqual(self.get_prefix(entropy.name), name + '_entropy')

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = layers.create_tensor(dtype="float32")
        layers.assign(value_npdata, value_tensor)

        lp = normal1.log_prob(value_tensor)
        self.assertEqual(self.get_prefix(lp.name), name + '_log_prob')

        p = normal1.probs(value_tensor)
        self.assertEqual(self.get_prefix(p.name), name + '_probs')

        kl = normal1.kl_divergence(normal2)
        self.assertEqual(self.get_prefix(kl.name), name + '_kl_divergence')

    def test_uniform_name(self):
        name = 'test_uniform'
        uniform1 = Uniform(0.0, 1.0, name=name)
        self.assertEqual(uniform1.name, name)

        uniform2 = Uniform(0.0, 1.0)
        self.assertEqual(uniform2.name, 'Uniform')

        paddle.enable_static()

        sample = uniform1.sample([2])
        self.assertEqual(self.get_prefix(sample.name), name + '_sample')

        entropy = uniform1.entropy()
        self.assertEqual(self.get_prefix(entropy.name), name + '_entropy')

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = layers.create_tensor(dtype="float32")
        layers.assign(value_npdata, value_tensor)

        lp = uniform1.log_prob(value_tensor)
        self.assertEqual(self.get_prefix(lp.name), name + '_log_prob')

        p = uniform1.probs(value_tensor)
        self.assertEqual(self.get_prefix(p.name), name + '_probs')

    def test_categorical_name(self):
        name = 'test_categorical'
        categorical1 = Categorical([0.4, 0.6], name=name)
        self.assertEqual(categorical1.name, name)

        categorical2 = Categorical([0.5, 0.5])
        self.assertEqual(categorical2.name, 'Categorical')

        paddle.enable_static()

        sample = categorical1.sample([2])
        self.assertEqual(self.get_prefix(sample.name), name + '_sample')

        entropy = categorical1.entropy()
        self.assertEqual(self.get_prefix(entropy.name), name + '_entropy')

        kl = categorical1.kl_divergence(categorical2)
        self.assertEqual(self.get_prefix(kl.name), name + '_kl_divergence')

        value_npdata = np.array([0], dtype="int64")
        value_tensor = layers.create_tensor(dtype="int64")
        layers.assign(value_npdata, value_tensor)

        p = categorical1.probs(value_tensor)
        self.assertEqual(self.get_prefix(p.name), name + '_probs')

        lp = categorical1.log_prob(value_tensor)
        self.assertEqual(self.get_prefix(lp.name), name + '_log_prob')


@parameterize.place(config.DEVICES)
@parameterize.parameterize_cls(
    (parameterize.TEST_CASE_NAME, 'batch_shape', 'event_shape'),
    [('test-tuple', (10, 20), (10, 20)),
     ('test-list', [100, 100], [100, 200, 300]), ('test-null-eventshape',
                                                  (100, 100), ())])
class TestDistributionShape(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.dist = paddle.distribution.Distribution(
            batch_shape=self.batch_shape, event_shape=self.event_shape)

    def tearDown(self):
        paddle.enable_static()

    def test_batch_shape(self):
        self.assertTrue(isinstance(self.dist.batch_shape, tuple))
        self.assertTrue(self.dist.batch_shape == tuple(self.batch_shape))

    def test_event_shape(self):
        self.assertTrue(isinstance(self.dist.event_shape, tuple))
        self.assertTrue(self.dist.event_shape == tuple(self.event_shape))

    def test_prob(self):
        with self.assertRaises(NotImplementedError):
            self.dist.prob(paddle.to_tensor(parameterize.xrand()))

    def test_extend_shape(self):
        shapes = [(34, 20), (56, ), ()]
        for shape in shapes:
            self.assertTrue(
                self.dist._extend_shape(shape),
                shape + self.dist.batch_shape + self.dist.event_shape)


class TestDistributionException(unittest.TestCase):
    def setUp(self):
        self._d = paddle.distribution.Distribution()

    def test_mean(self):
        with self.assertRaises(NotImplementedError):
            self._d.mean

    def test_variance(self):
        with self.assertRaises(NotImplementedError):
            self._d.variance

    def test_rsample(self):
        with self.assertRaises(NotImplementedError):
            self._d.rsample(())


if __name__ == '__main__':
    unittest.main()
