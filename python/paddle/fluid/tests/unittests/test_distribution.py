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

import numpy as np
import unittest
import paddle
from paddle import fluid
from paddle.fluid import layers
from paddle.distribution import *
import math


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


class UniformNumpy(DistributionNumpy):
    def __init__(self, low, high):
        self.low = np.array(low).astype('float32')
        self.high = np.array(high).astype('float32')

    def sample(self, shape):
        shape = tuple(shape) + (self.low + self.high).shape
        return self.low + (np.random.uniform(size=shape) *
                           (self.high - self.low))

    def log_prob(self, value):
        lb = np.less(self.low, value).astype('float32')
        ub = np.less(value, self.high).astype('float32')
        return np.log(lb * ub) - np.log(self.high - self.low)

    def probs(self, value):
        lb = np.less(self.low, value).astype('float32')
        ub = np.less(value, self.high).astype('float32')
        return (lb * ub) / (self.high - self.low)

    def entropy(self):
        return np.log(self.high - self.low)


class DistributionTest(unittest.TestCase):
    def setUp(self, use_gpu=False):
        self.use_gpu = use_gpu
        if not use_gpu:
            place = fluid.CPUPlace()
            self.gpu_id = -1
        else:
            place = fluid.CUDAPlace(0)
            self.gpu_id = 0
        self.executor = fluid.Executor(place)

    def build_uniform_common_net(self, batch_size, dims, low_float, high_float,
                                 high_np, low_np, values_np, low, high, values):
        uniform_int = Uniform(int(low_float), int(high_float))
        uniform_float = Uniform(low_float, high_float)
        uniform_float_np_broadcast = Uniform(low_float, high_np)
        uniform_np = Uniform(low_np, high_np)
        uniform_variable = Uniform(low, high)

        sample_int = uniform_int.sample([batch_size, dims])
        sample_float = uniform_float.sample([batch_size, dims])
        sample_float_np_broadcast = uniform_float_np_broadcast.sample(
            [batch_size, dims])
        sample_np = uniform_np.sample([batch_size, dims])
        sample_variable = uniform_variable.sample([batch_size, dims])

        entropy_int = uniform_int.entropy()
        entropy_float = uniform_float.entropy()
        entropy_float_np_broadcast = uniform_float_np_broadcast.entropy()
        entropy_np = uniform_np.entropy()
        entropy_variable = uniform_variable.entropy()

        lp_float_np_broadcast = uniform_float_np_broadcast.log_prob(values)
        lp_np = uniform_np.log_prob(values)
        lp_variable = uniform_variable.log_prob(values)

        p_float_np_broadcast = uniform_float_np_broadcast.probs(values)
        p_np = uniform_np.probs(values)
        p_variable = uniform_variable.probs(values)

        fetch_list = [
            sample_int, sample_float, sample_float_np_broadcast, sample_np,
            sample_variable, entropy_int, entropy_float,
            entropy_float_np_broadcast, entropy_np, entropy_variable,
            lp_float_np_broadcast, lp_np, lp_variable, p_float_np_broadcast,
            p_np, p_variable
        ]
        return fetch_list

    def build_uniform_static(self, test_program, batch_size, dims, low_float,
                             high_float, high_np, low_np, values_np):
        with fluid.program_guard(test_program):
            low = layers.data(name='low', shape=[dims], dtype='float32')
            high = layers.data(name='high', shape=[dims], dtype='float32')

            values = layers.data(name='values', shape=[dims], dtype='float32')

            fetch_list = self.build_uniform_common_net(
                batch_size, dims, low_float, high_float, high_np, low_np,
                values_np, low, high, values)

        feed_vars = {'low': low_np, 'high': high_np, 'values': values_np}
        return feed_vars, fetch_list

    def build_uniform_dygraph(self, batch_size, dims, low_float, high_float,
                              high_np, low_np, values_np):
        low = paddle.to_tensor(low_np)
        high = paddle.to_tensor(high_np)
        values = paddle.to_tensor(values_np)

        fetch_list = self.build_uniform_common_net(batch_size, dims, low_float,
                                                   high_float, high_np, low_np,
                                                   values_np, low, high, values)
        fetch_list_numpy = [t.numpy() for t in fetch_list]
        return fetch_list_numpy

    def compare_uniform_with_numpy(self,
                                   data_list,
                                   output_list,
                                   batch_size=2,
                                   dims=3,
                                   tolerance=1e-6):
        [low_np, low_float, high_float, high_np, values_np] = data_list

        np_uniform_int = UniformNumpy(int(low_float), int(high_float))
        np_uniform_float = UniformNumpy(low_float, high_float)
        np_uniform_float_np_broadcast = UniformNumpy(low_float, high_np)
        np_uniform = UniformNumpy(low_np, high_np)

        gt_sample_int = np_uniform_int.sample([batch_size, dims])
        gt_sample_float = np_uniform_float.sample([batch_size, dims])
        gt_sample_float_np_broadcast = np_uniform_float_np_broadcast.sample(
            [batch_size, dims])
        gt_sample_np = np_uniform.sample([batch_size, dims])
        gt_entropy_int = np_uniform_int.entropy()
        gt_entropy_float = np_uniform_float.entropy()
        gt_entropy_float_np_broadcast = np_uniform_float_np_broadcast.entropy()
        gt_entropy = np_uniform.entropy()
        gt_lp_float_np_broadcast = np_uniform_float_np_broadcast.log_prob(
            values_np)
        gt_lp = np_uniform.log_prob(values_np)
        gt_p_float_np_broadcast = np_uniform_float_np_broadcast.probs(values_np)
        gt_p = np_uniform.probs(values_np)

        [
            output_sample_int, output_sample_float,
            output_sample_float_np_broadcast, output_sample_np,
            output_sample_variable, output_entropy_int, output_entropy_float,
            output_entropy_float_np_broadcast, output_entropy_np,
            output_entropy_variable, output_lp_float_np_broadcast, output_lp_np,
            output_lp_variable, output_p_float_np_broadcast, output_p_np,
            output_p_variable
        ] = output_list

        np.testing.assert_allclose(
            output_sample_int.shape,
            gt_sample_int.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_float.shape,
            gt_sample_float.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_float_np_broadcast.shape,
            gt_sample_float_np_broadcast.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_np.shape,
            gt_sample_np.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_sample_variable.shape,
            gt_sample_np.shape,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_int, gt_entropy_int, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_float,
            gt_entropy_float,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_float_np_broadcast,
            gt_entropy_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_np, gt_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_entropy_variable, gt_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_lp_float_np_broadcast,
            gt_lp_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_lp_np, gt_lp, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_lp_variable, gt_lp, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_p_float_np_broadcast,
            gt_p_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_p_np, gt_p, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_p_variable, gt_p, rtol=tolerance, atol=tolerance)

    def test_uniform_distribution_static(self,
                                         batch_size=2,
                                         dims=3,
                                         tolerance=1e-6):
        test_program = fluid.Program()

        low_np = np.random.randn(batch_size, dims).astype('float32')
        low_float = np.random.uniform(-2, 1)
        high_float = np.random.uniform(1, 3)
        high_np = np.random.uniform(-5.0, 5.0,
                                    (batch_size, dims)).astype('float32')
        values_np = np.random.randn(batch_size, dims).astype('float32')

        data_list = [low_np, low_float, high_float, high_np, values_np]

        feed_vars, fetch_list = self.build_uniform_static(
            test_program, batch_size, dims, low_float, high_float, high_np,
            low_np, values_np)

        self.executor.run(fluid.default_startup_program())

        # result calculated by paddle
        output_list = self.executor.run(program=test_program,
                                        feed=feed_vars,
                                        fetch_list=fetch_list)
        self.compare_uniform_with_numpy(data_list, output_list, batch_size,
                                        dims, tolerance)

    def test_uniform_distribution_dygraph(self,
                                          batch_size=2,
                                          dims=3,
                                          tolerance=1e-6):
        paddle.disable_static()

        low_np = np.random.randn(batch_size, dims).astype('float32')
        low_float = np.random.uniform(-2, 1)
        high_float = np.random.uniform(1, 3)
        high_np = np.random.uniform(-5.0, 5.0,
                                    (batch_size, dims)).astype('float32')
        values_np = np.random.randn(batch_size, dims).astype('float32')

        data_list = [low_np, low_float, high_float, high_np, values_np]
        output_list = self.build_uniform_dygraph(
            batch_size, dims, low_float, high_float, high_np, low_np, values_np)

        self.compare_uniform_with_numpy(data_list, output_list, batch_size,
                                        dims, tolerance)
        paddle.enable_static()


class DistributionTestError(unittest.TestCase):
    def test_distribution_error(self):
        distribution = Distribution()

        self.assertRaises(NotImplementedError, distribution.sample)
        self.assertRaises(NotImplementedError, distribution.entropy)

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = layers.create_tensor(dtype="float32")
        self.assertRaises(NotImplementedError, distribution.log_prob,
                          value_tensor)
        self.assertRaises(NotImplementedError, distribution.probs, value_tensor)

    def test_uniform_error(self):
        uniform = Uniform(0.0, 1.0)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, uniform.log_prob, value)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, uniform.probs, value)

        shape = 1.0
        # type of shape must be list
        self.assertRaises(TypeError, uniform.sample, shape)

        seed = 1.0
        # type of seed must be int
        self.assertRaises(TypeError, uniform.sample, [2, 3], seed)


class DistributionTestName(unittest.TestCase):
    def get_prefix(self, string):
        return (string.split('.')[0])

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


if __name__ == '__main__':
    unittest.main()
