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

<<<<<<< HEAD
import unittest

import numpy as np
from test_distribution import DistributionNumpy

import paddle
from paddle import fluid
from paddle.distribution import Categorical, Distribution, Normal, Uniform
=======
import math
import unittest

import numpy as np
import paddle
from paddle import fluid
from paddle.distribution import *
from paddle.fluid import layers

from test_distribution import DistributionNumpy
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

np.random.seed(2022)


class CategoricalNumpy(DistributionNumpy):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, logits):
        self.logits = np.array(logits).astype('float32')

    def entropy(self):
        logits = self.logits - np.max(self.logits, axis=-1, keepdims=True)
        e_logits = np.exp(logits)
        z = np.sum(e_logits, axis=-1, keepdims=True)
        prob = e_logits / z
<<<<<<< HEAD
        return -1.0 * np.sum(prob * (logits - np.log(z)), axis=-1)
=======
        return -1. * np.sum(prob * (logits - np.log(z)), axis=-1)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def kl_divergence(self, other):
        logits = self.logits - np.max(self.logits, axis=-1, keepdims=True)
        other_logits = other.logits - np.max(
<<<<<<< HEAD
            other.logits, axis=-1, keepdims=True
        )
=======
            other.logits, axis=-1, keepdims=True)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        e_logits = np.exp(logits)
        other_e_logits = np.exp(other_logits)
        z = np.sum(e_logits, axis=-1, keepdims=True)
        other_z = np.sum(other_e_logits, axis=-1, keepdims=True)
        prob = e_logits / z
<<<<<<< HEAD
        return np.sum(
            prob * (logits - np.log(z) - other_logits + np.log(other_z)),
            axis=-1,
            keepdims=True,
        )


class CategoricalTest(unittest.TestCase):
=======
        return np.sum(prob *
                      (logits - np.log(z) - other_logits + np.log(other_z)),
                      axis=-1,
                      keepdims=True)


class CategoricalTest(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self, use_gpu=False, batch_size=3, dims=5):
        self.use_gpu = use_gpu
        if not use_gpu:
            self.place = fluid.CPUPlace()
            self.gpu_id = -1
        else:
            self.place = fluid.CUDAPlace(0)
            self.gpu_id = 0

        self.batch_size = batch_size
        self.dims = dims
        self.init_numpy_data(batch_size, dims)

        paddle.disable_static(self.place)
        self.init_dynamic_data(batch_size, dims)

        paddle.enable_static()
        self.test_program = fluid.Program()
        self.executor = fluid.Executor(self.place)
        self.init_static_data(batch_size, dims)

    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D Tensor
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(batch_size, dims).astype('float32')
<<<<<<< HEAD
        self.other_logits_np = np.random.rand(batch_size, dims).astype(
            'float32'
        )
=======
        self.other_logits_np = np.random.rand(batch_size,
                                              dims).astype('float32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [batch_size, dims]
        # dist_shape = logits_shape[:-1], it represents the number of
        #  different distributions.
        self.dist_shape = [batch_size]
        # sample shape represents the number of samples
        self.sample_shape = [2, 4]
        # value used in probs and log_prob method
        # If value is 1-D and logits is 2-D or higher dimension, value will be
        #  broadcasted to have the same number of distributions with logits.
        # If value is 2-D or higher dimentsion, it should have the same number
        #  of distributions with logtis. ``value[:-1] = logits[:-1]
        self.value_shape = [3]

    def init_dynamic_data(self, batch_size, dims):
        self.logits = paddle.to_tensor(self.logits_np)
        self.other_logits = paddle.to_tensor(self.other_logits_np)
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
<<<<<<< HEAD
            self.logits_static = fluid.data(
                name='logits', shape=self.logits_shape, dtype='float32'
            )
            self.other_logits_static = fluid.data(
                name='other_logits', shape=self.logits_shape, dtype='float32'
            )
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64'
            )
=======
            self.logits_static = fluid.data(name='logits',
                                            shape=self.logits_shape,
                                            dtype='float32')
            self.other_logits_static = fluid.data(name='other_logits',
                                                  shape=self.logits_shape,
                                                  dtype='float32')
            self.value_static = fluid.data(name='value',
                                           shape=self.value_shape,
                                           dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.dist_shape + self.value_shape)
        for i in range(self.batch_size):
            for j in range(3):
                np_probs[i][j] = probability[i][self.value_np[j]]
        return np_probs

    def compare_with_numpy(self, fetch_list, tolerance=1e-6):
        sample, entropy, kl, probs, log_prob = fetch_list
        log_tolerance = 1e-4

<<<<<<< HEAD
        np.testing.assert_equal(
            sample.shape, self.sample_shape + self.dist_shape
        )
=======
        np.testing.assert_equal(sample.shape,
                                self.sample_shape + self.dist_shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        np_categorical = CategoricalNumpy(self.logits_np)
        np_other_categorical = CategoricalNumpy(self.other_logits_np)
        np_entropy = np_categorical.entropy()
        np_kl = np_categorical.kl_divergence(np_other_categorical)

<<<<<<< HEAD
        np.testing.assert_allclose(
            entropy, np_entropy, rtol=log_tolerance, atol=log_tolerance
        )
        np.testing.assert_allclose(
            kl, np_kl, rtol=log_tolerance, atol=log_tolerance
        )
=======
        np.testing.assert_allclose(entropy,
                                   np_entropy,
                                   rtol=log_tolerance,
                                   atol=log_tolerance)
        np.testing.assert_allclose(kl,
                                   np_kl,
                                   rtol=log_tolerance,
                                   atol=log_tolerance)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        sum_dist = np.sum(self.logits_np, axis=-1, keepdims=True)
        probability = self.logits_np / sum_dist
        np_probs = self.get_numpy_selected_probs(probability)
        np_log_prob = np.log(np_probs)

<<<<<<< HEAD
        np.testing.assert_allclose(
            probs, np_probs, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            log_prob, np_log_prob, rtol=tolerance, atol=tolerance
        )
=======
        np.testing.assert_allclose(probs,
                                   np_probs,
                                   rtol=tolerance,
                                   atol=tolerance)
        np.testing.assert_allclose(log_prob,
                                   np_log_prob,
                                   rtol=tolerance,
                                   atol=tolerance)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_categorical_distribution_dygraph(self, tolerance=1e-6):
        paddle.disable_static(self.place)
        categorical = Categorical(self.logits)
        other_categorical = Categorical(self.other_logits)

        sample = categorical.sample(self.sample_shape).numpy()
        entropy = categorical.entropy().numpy()
        kl = categorical.kl_divergence(other_categorical).numpy()
        probs = categorical.probs(self.value).numpy()
        log_prob = categorical.log_prob(self.value).numpy()

        fetch_list = [sample, entropy, kl, probs, log_prob]
        self.compare_with_numpy(fetch_list)

    def test_categorical_distribution_static(self, tolerance=1e-6):
        paddle.enable_static()
        with fluid.program_guard(self.test_program):
            categorical = Categorical(self.logits_static)
            other_categorical = Categorical(self.other_logits_static)

            sample = categorical.sample(self.sample_shape)
            entropy = categorical.entropy()
            kl = categorical.kl_divergence(other_categorical)
            probs = categorical.probs(self.value_static)
            log_prob = categorical.log_prob(self.value_static)

            fetch_list = [sample, entropy, kl, probs, log_prob]

        feed_vars = {
            'logits': self.logits_np,
            'other_logits': self.other_logits_np,
<<<<<<< HEAD
            'value': self.value_np,
        }

        self.executor.run(fluid.default_startup_program())
        fetch_list = self.executor.run(
            program=self.test_program, feed=feed_vars, fetch_list=fetch_list
        )
=======
            'value': self.value_np
        }

        self.executor.run(fluid.default_startup_program())
        fetch_list = self.executor.run(program=self.test_program,
                                       feed=feed_vars,
                                       fetch_list=fetch_list)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.compare_with_numpy(fetch_list)


class CategoricalTest2(CategoricalTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D Tensor with dtype Float64
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(batch_size, dims).astype('float64')
<<<<<<< HEAD
        self.other_logits_np = np.random.rand(batch_size, dims).astype(
            'float64'
        )
=======
        self.other_logits_np = np.random.rand(batch_size,
                                              dims).astype('float64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [batch_size, dims]
        self.dist_shape = [batch_size]
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
<<<<<<< HEAD
            self.logits_static = fluid.data(
                name='logits', shape=self.logits_shape, dtype='float64'
            )
            self.other_logits_static = fluid.data(
                name='other_logits', shape=self.logits_shape, dtype='float64'
            )
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64'
            )


class CategoricalTest3(CategoricalTest):
=======
            self.logits_static = fluid.data(name='logits',
                                            shape=self.logits_shape,
                                            dtype='float64')
            self.other_logits_static = fluid.data(name='other_logits',
                                                  shape=self.logits_shape,
                                                  dtype='float64')
            self.value_static = fluid.data(name='value',
                                           shape=self.value_shape,
                                           dtype='int64')


class CategoricalTest3(CategoricalTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dynamic_data(self, batch_size, dims):
        # input logtis is 2-D numpy.ndarray with dtype Float32
        # value used in probs and log_prob method is 1-D Tensor
        self.logits = self.logits_np
        self.other_logits = self.other_logits_np
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.logits_static = self.logits_np
            self.other_logits_static = self.other_logits_np
<<<<<<< HEAD
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64'
            )


class CategoricalTest4(CategoricalTest):
=======
            self.value_static = fluid.data(name='value',
                                           shape=self.value_shape,
                                           dtype='int64')


class CategoricalTest4(CategoricalTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D numpy.ndarray with dtype Float64
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(batch_size, dims).astype('float64')
<<<<<<< HEAD
        self.other_logits_np = np.random.rand(batch_size, dims).astype(
            'float64'
        )
=======
        self.other_logits_np = np.random.rand(batch_size,
                                              dims).astype('float64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [batch_size, dims]
        self.dist_shape = [batch_size]
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def init_dynamic_data(self, batch_size, dims):
        self.logits = self.logits_np
        self.other_logits = self.other_logits_np
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.logits_static = self.logits_np
            self.other_logits_static = self.other_logits_np
<<<<<<< HEAD
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64'
            )
=======
            self.value_static = fluid.data(name='value',
                                           shape=self.value_shape,
                                           dtype='int64')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


# test shape of logits and value used in probs and log_prob method
class CategoricalTest5(CategoricalTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 1-D Tensor
        # value used in probs and log_prob method is 1-D Tensor
        self.logits_np = np.random.rand(dims).astype('float32')
        self.other_logits_np = np.random.rand(dims).astype('float32')
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [dims]
        self.dist_shape = []
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.value_shape)
        for i in range(3):
            np_probs[i] = probability[self.value_np[i]]
        return np_probs


class CategoricalTest6(CategoricalTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 2-D Tensor
        # value used in probs and log_prob method has the same number of batches with input
        self.logits_np = np.random.rand(3, 5).astype('float32')
        self.other_logits_np = np.random.rand(3, 5).astype('float32')
        self.value_np = np.array([[2, 1], [0, 3], [2, 3]]).astype('int64')

        self.logits_shape = [3, 5]
        self.dist_shape = [3]
        self.sample_shape = [2, 4]
        self.value_shape = [3, 2]

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.value_shape)
        for i in range(3):
            for j in range(2):
                np_probs[i][j] = probability[i][self.value_np[i][j]]
        return np_probs


class CategoricalTest7(CategoricalTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_numpy_data(self, batch_size, dims):
        # input logtis is 3-D Tensor
        # value used in probs and log_prob method has the same number of distribuions with input
        self.logits_np = np.random.rand(3, 2, 5).astype('float32')
        self.other_logits_np = np.random.rand(3, 2, 5).astype('float32')
        self.value_np = np.array([2, 1, 3]).astype('int64')

        self.logits_shape = [3, 2, 5]
        self.dist_shape = [3, 2]
        self.sample_shape = [2, 4]
        self.value_shape = [3]

    def get_numpy_selected_probs(self, probability):
        np_probs = np.zeros(self.dist_shape + self.value_shape)
        for i in range(3):
            for j in range(2):
                for k in range(3):
                    np_probs[i][j][k] = probability[i][j][self.value_np[k]]
        return np_probs


class CategoricalTest8(CategoricalTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dynamic_data(self, batch_size, dims):
        # input logtis is 2-D list
        # value used in probs and log_prob method is 1-D Tensor
        self.logits = self.logits_np.tolist()
        self.other_logits = self.other_logits_np.tolist()
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.logits_static = self.logits_np.tolist()
            self.other_logits_static = self.other_logits_np.tolist()
<<<<<<< HEAD
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64'
            )


class CategoricalTest9(CategoricalTest):
=======
            self.value_static = fluid.data(name='value',
                                           shape=self.value_shape,
                                           dtype='int64')


class CategoricalTest9(CategoricalTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dynamic_data(self, batch_size, dims):
        # input logtis is 2-D tuple
        # value used in probs and log_prob method is 1-D Tensor
        self.logits = tuple(self.logits_np.tolist())
        self.other_logits = tuple(self.other_logits_np.tolist())
        self.value = paddle.to_tensor(self.value_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.logits_static = tuple(self.logits_np.tolist())
            self.other_logits_static = tuple(self.other_logits_np.tolist())
<<<<<<< HEAD
            self.value_static = fluid.data(
                name='value', shape=self.value_shape, dtype='int64'
            )


class DistributionTestError(unittest.TestCase):
=======
            self.value_static = fluid.data(name='value',
                                           shape=self.value_shape,
                                           dtype='int64')


class DistributionTestError(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_distribution_error(self):
        distribution = Distribution()

        self.assertRaises(NotImplementedError, distribution.sample)
        self.assertRaises(NotImplementedError, distribution.entropy)

        normal = Normal(0.0, 1.0)
<<<<<<< HEAD
        self.assertRaises(
            NotImplementedError, distribution.kl_divergence, normal
        )

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = paddle.tensor.create_tensor(dtype="float32")
        self.assertRaises(
            NotImplementedError, distribution.log_prob, value_tensor
        )
=======
        self.assertRaises(NotImplementedError, distribution.kl_divergence,
                          normal)

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = layers.create_tensor(dtype="float32")
        self.assertRaises(NotImplementedError, distribution.log_prob,
                          value_tensor)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertRaises(NotImplementedError, distribution.probs, value_tensor)

    def test_normal_error(self):
        paddle.enable_static()
        normal = Normal(0.0, 1.0)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, normal.log_prob, value)

        value = [1.0, 2.0]
        # type of value must be variable
        self.assertRaises(TypeError, normal.probs, value)

        shape = 1.0
        # type of shape must be list
        self.assertRaises(TypeError, normal.sample, shape)

        seed = 1.0
        # type of seed must be int
        self.assertRaises(TypeError, normal.sample, [2, 3], seed)

        normal_other = Uniform(1.0, 2.0)
        # type of other must be an instance of Normal
        self.assertRaises(TypeError, normal.kl_divergence, normal_other)

    def test_uniform_error(self):
        paddle.enable_static()
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

    def test_categorical_error(self):
        paddle.enable_static()

        categorical = Categorical([0.4, 0.6])

        value = [1, 0]
        # type of value must be variable
        self.assertRaises(AttributeError, categorical.log_prob, value)

        value = [1, 0]
        # type of value must be variable
        self.assertRaises(AttributeError, categorical.probs, value)

        shape = 1.0
        # type of shape must be list
        self.assertRaises(TypeError, categorical.sample, shape)

        categorical_other = Uniform(1.0, 2.0)
        # type of other must be an instance of Categorical
<<<<<<< HEAD
        self.assertRaises(
            TypeError, categorical.kl_divergence, categorical_other
        )
=======
        self.assertRaises(TypeError, categorical.kl_divergence,
                          categorical_other)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_shape_not_match_error():
            # shape of value must match shape of logits
            # value_shape[:-1] == logits_shape[:-1]
            paddle.disable_static()
            logits = paddle.rand([3, 5])
            cat = Categorical(logits)
            value = paddle.to_tensor([[2, 1, 3], [3, 2, 1]], dtype='int64')
            cat.log_prob(value)

        self.assertRaises(ValueError, test_shape_not_match_error)


if __name__ == '__main__':
    unittest.main()
