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
        self.low = np.array(low)
        self.high = np.array(high)
        if str(self.low.dtype) not in ['float32', 'float64']:
            self.low = self.low.astype('float32')
            self.high = self.high.astype('float32')

    def sample(self, shape):
        shape = tuple(shape) + (self.low + self.high).shape
        return self.low + (np.random.uniform(size=shape) *
                           (self.high - self.low))

    def log_prob(self, value):
        lb = np.less(self.low, value).astype(self.low.dtype)
        ub = np.less(value, self.high).astype(self.low.dtype)
        return np.log(lb * ub) - np.log(self.high - self.low)

    def probs(self, value):
        lb = np.less(self.low, value).astype(self.low.dtype)
        ub = np.less(value, self.high).astype(self.low.dtype)
        return (lb * ub) / (self.high - self.low)

    def entropy(self):
        return np.log(self.high - self.low)


class NormalNumpy(DistributionNumpy):
    def __init__(self, loc, scale):
        self.loc = np.array(loc)
        self.scale = np.array(scale)
        if str(self.loc.dtype) not in ['float32', 'float64']:
            self.loc = self.loc.astype('float32')
            self.scale = self.scale.astype('float32')

    def sample(self, shape):
        shape = tuple(shape) + (self.loc + self.scale).shape
        return self.loc + (np.random.randn(*shape) * self.scale)

    def log_prob(self, value):
        var = self.scale * self.scale
        log_scale = np.log(self.scale)
        return -((value - self.loc) * (value - self.loc)) / (
            2. * var) - log_scale - math.log(math.sqrt(2. * math.pi))

    def probs(self, value):
        var = self.scale * self.scale
        return np.exp(-1. * ((value - self.loc) * (value - self.loc)) /
                      (2. * var)) / (math.sqrt(2 * math.pi) * self.scale)

    def entropy(self):
        return 0.5 + 0.5 * np.log(
            np.array(2. * math.pi).astype(self.loc.dtype)) + np.log(self.scale)

    def kl_divergence(self, other):
        var_ratio = (self.scale / other.scale)
        var_ratio = var_ratio * var_ratio
        t1 = ((self.loc - other.loc) / other.scale)
        t1 = (t1 * t1)
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


class UniformTest(unittest.TestCase):
    def setUp(self, use_gpu=False, batch_size=5, dims=6):
        self.use_gpu = use_gpu
        if not use_gpu:
            self.place = fluid.CPUPlace()
            self.gpu_id = -1
        else:
            self.place = fluid.CUDAPlace(0)
            self.gpu_id = 0

        self.init_numpy_data(batch_size, dims)

        paddle.disable_static(self.place)
        self.init_dynamic_data(batch_size, dims)

        paddle.enable_static()
        self.test_program = fluid.Program()
        self.executor = fluid.Executor(self.place)
        self.init_static_data(batch_size, dims)

    def init_numpy_data(self, batch_size, dims):
        # low ans high are 'float'
        self.low_np = np.random.uniform(-2, 1)
        self.high_np = np.random.uniform(1, 3)
        self.values_np = np.array([1.0]).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = self.low_np
        self.dynamic_high = self.high_np
        self.dynamic_values = paddle.to_tensor(self.values_np)

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[], dtype='float32')

    def compare_with_numpy(self, fetch_list, sample_shape=7, tolerance=1e-6):
        sample, entropy, log_prob, probs = fetch_list

        np_uniform = UniformNumpy(self.low_np, self.high_np)
        np_sample = np_uniform.sample([sample_shape])
        np_entropy = np_uniform.entropy()
        np_lp = np_uniform.log_prob(self.values_np)
        np_p = np_uniform.probs(self.values_np)

        np.testing.assert_equal(sample.shape, np_sample.shape)
        np.testing.assert_allclose(
            entropy, np_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            log_prob, np_lp, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(probs, np_p, rtol=tolerance, atol=tolerance)

    def test_uniform_distribution_dygraph(self, sample_shape=7, tolerance=1e-6):
        paddle.disable_static(self.place)
        uniform = Uniform(self.dynamic_low, self.dynamic_high)
        sample = uniform.sample([sample_shape]).numpy()
        entropy = uniform.entropy().numpy()
        log_prob = uniform.log_prob(self.dynamic_values).numpy()
        probs = uniform.probs(self.dynamic_values).numpy()
        fetch_list = [sample, entropy, log_prob, probs]

        self.compare_with_numpy(fetch_list)

    def test_uniform_distribution_static(self, sample_shape=7, tolerance=1e-6):
        paddle.enable_static()
        with fluid.program_guard(self.test_program):
            uniform = Uniform(self.static_low, self.static_high)
            sample = uniform.sample([sample_shape])
            entropy = uniform.entropy()
            log_prob = uniform.log_prob(self.static_values)
            probs = uniform.probs(self.static_values)
            fetch_list = [sample, entropy, log_prob, probs]

        feed_vars = {
            'low': self.low_np,
            'high': self.high_np,
            'values': self.values_np
        }

        self.executor.run(fluid.default_startup_program())
        fetch_list = self.executor.run(program=self.test_program,
                                       feed=feed_vars,
                                       fetch_list=fetch_list)

        self.compare_with_numpy(fetch_list)


class UniformTest2(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low ans high are 'int'
        self.low_np = int(np.random.uniform(-2, 1))
        self.high_np = int(np.random.uniform(1, 3))
        self.values_np = np.array([1.0]).astype('float32')


class UniformTest3(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # test broadcast: low is float, high is numpy.ndarray with dtype 'float32'.
        self.low_np = np.random.uniform(-2, 1)
        self.high_np = np.random.uniform(-5.0, 5.0,
                                         (batch_size, dims)).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class UniformTest4(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are numpy.ndarray with dtype 'float32'.
        self.low_np = np.random.randn(batch_size, dims).astype('float32')
        self.high_np = np.random.uniform(-5.0, 5.0,
                                         (batch_size, dims)).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class UniformTest5(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are numpy.ndarray with dtype 'float64'.
        self.low_np = np.random.randn(batch_size, dims).astype('float64')
        self.high_np = np.random.uniform(-5.0, 5.0,
                                         (batch_size, dims)).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = self.low_np
        self.dynamic_high = self.high_np
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float64')


class UniformTest6(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are Tensor with dtype 'VarType.FP32'.
        self.low_np = np.random.randn(batch_size, dims).astype('float32')
        self.high_np = np.random.uniform(-5.0, 5.0,
                                         (batch_size, dims)).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = paddle.to_tensor(self.low_np)
        self.dynamic_high = paddle.to_tensor(self.high_np)
        self.dynamic_values = paddle.to_tensor(self.values_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_low = layers.data(
                name='low', shape=[dims], dtype='float32')
            self.static_high = layers.data(
                name='high', shape=[dims], dtype='float32')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class UniformTest7(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are Tensor with dtype 'VarType.FP64'.
        self.low_np = np.random.randn(batch_size, dims).astype('float64')
        self.high_np = np.random.uniform(-5.0, 5.0,
                                         (batch_size, dims)).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = paddle.to_tensor(self.low_np, dtype='float64')
        self.dynamic_high = paddle.to_tensor(self.high_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_low = layers.data(
                name='low', shape=[dims], dtype='float64')
            self.static_high = layers.data(
                name='high', shape=[dims], dtype='float64')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float64')


class UniformTest8(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are Tensor with dtype 'VarType.FP64'. value's dtype is 'VarType.FP32'.
        self.low_np = np.random.randn(batch_size, dims).astype('float64')
        self.high_np = np.random.uniform(-5.0, 5.0,
                                         (batch_size, dims)).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = paddle.to_tensor(self.low_np, dtype='float64')
        self.dynamic_high = paddle.to_tensor(self.high_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float32')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_low = layers.data(
                name='low', shape=[dims], dtype='float64')
            self.static_high = layers.data(
                name='high', shape=[dims], dtype='float64')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class NormalTest(unittest.TestCase):
    def setUp(self, use_gpu=False, batch_size=2, dims=3):
        self.use_gpu = use_gpu
        if not use_gpu:
            self.place = fluid.CPUPlace()
            self.gpu_id = -1
        else:
            self.place = fluid.CUDAPlace(0)
            self.gpu_id = 0

        self.init_numpy_data(batch_size, dims)

        paddle.disable_static(self.place)
        self.init_dynamic_data(batch_size, dims)

        paddle.enable_static()
        self.test_program = fluid.Program()
        self.executor = fluid.Executor(self.place)
        self.init_static_data(batch_size, dims)

    def init_numpy_data(self, batch_size, dims):
        # loc ans scale are 'float'
        self.loc_np = (np.random.ranf() - 0.5) * 4
        self.scale_np = (np.random.ranf() - 0.5) * 4
        while self.scale_np < 0:
            self.scale_np = (np.random.ranf() - 0.5) * 4
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = (np.random.ranf() - 0.5) * 4
        self.other_scale_np = (np.random.ranf() - 0.5) * 4
        while self.other_scale_np < 0:
            self.other_scale_np = (np.random.ranf() - 0.5) * 4
        self.values_np = np.random.ranf(1).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = self.loc_np
        self.dynamic_scale = self.scale_np
        self.dynamic_other_loc = self.other_loc_np
        self.dynamic_other_scale = self.other_scale_np
        self.dynamic_values = paddle.to_tensor(self.values_np)

    def init_static_data(self, batch_size, dims):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[], dtype='float32')

    def compare_with_numpy(self, fetch_list, sample_shape=7, tolerance=1e-6):
        sample, entropy, log_prob, probs, kl = fetch_list

        np_normal = NormalNumpy(self.loc_np, self.scale_np)
        np_sample = np_normal.sample([sample_shape])
        np_entropy = np_normal.entropy()
        np_lp = np_normal.log_prob(self.values_np)
        np_p = np_normal.probs(self.values_np)
        np_other_normal = NormalNumpy(self.other_loc_np, self.other_scale_np)
        np_kl = np_normal.kl_divergence(np_other_normal)

        np.testing.assert_equal(sample.shape, np_sample.shape)
        np.testing.assert_allclose(
            entropy, np_entropy, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            log_prob, np_lp, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(probs, np_p, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(kl, np_kl, rtol=tolerance, atol=tolerance)

    def test_normal_distribution_dygraph(self, sample_shape=7, tolerance=1e-6):
        paddle.disable_static(self.place)
        normal = Normal(self.dynamic_loc, self.dynamic_scale)

        sample = normal.sample([sample_shape]).numpy()
        entropy = normal.entropy().numpy()
        log_prob = normal.log_prob(self.dynamic_values).numpy()
        probs = normal.probs(self.dynamic_values).numpy()
        other_normal = Normal(self.dynamic_other_loc, self.dynamic_other_scale)
        kl = normal.kl_divergence(other_normal).numpy()

        fetch_list = [sample, entropy, log_prob, probs, kl]
        self.compare_with_numpy(fetch_list)

    def test_normal_distribution_static(self, sample_shape=7, tolerance=1e-6):
        paddle.enable_static()
        with fluid.program_guard(self.test_program):
            normal = Normal(self.static_loc, self.static_scale)

            sample = normal.sample([sample_shape])
            entropy = normal.entropy()
            log_prob = normal.log_prob(self.static_values)
            probs = normal.probs(self.static_values)
            other_normal = Normal(self.static_other_loc,
                                  self.static_other_scale)
            kl = normal.kl_divergence(other_normal)

            fetch_list = [sample, entropy, log_prob, probs, kl]

        feed_vars = {
            'loc': self.loc_np,
            'scale': self.scale_np,
            'values': self.values_np,
            'other_loc': self.other_loc_np,
            'other_scale': self.other_scale_np
        }

        self.executor.run(fluid.default_startup_program())
        fetch_list = self.executor.run(program=self.test_program,
                                       feed=feed_vars,
                                       fetch_list=fetch_list)

        self.compare_with_numpy(fetch_list)


class NormalTest2(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc ans scale are 'int'
        self.loc_np = int((np.random.ranf() - 0.5) * 8)
        self.scale_np = int((np.random.ranf() - 0.5) * 8)
        while self.scale_np < 0:
            self.scale_np = int((np.random.ranf() - 0.5) * 8)
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = int((np.random.ranf() - 0.5) * 8)
        self.other_scale_np = int((np.random.ranf() - 0.5) * 8)
        while self.other_scale_np < 0:
            self.other_scale_np = int((np.random.ranf() - 0.5) * 8)
        self.values_np = np.random.ranf(1).astype('float32')


class NormalTest3(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # test broadcast: loc is float, scale is numpy.ndarray with dtype 'float32'.
        self.loc_np = (np.random.ranf() - 0.5) * 4
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = (np.random.ranf() - 0.5) * 4
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class NormalTest4(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are numpy.ndarray with dtype 'float32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')


class NormalTest5(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are numpy.ndarray with dtype 'float64'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = self.loc_np
        self.dynamic_scale = self.scale_np
        self.dynamic_other_loc = self.other_loc_np
        self.dynamic_other_scale = self.other_scale_np
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        self.static_loc = self.loc_np
        self.static_scale = self.scale_np
        self.static_other_loc = self.other_loc_np
        self.static_other_scale = self.other_scale_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float64')


class NormalTest6(NormalTest):
    def init_data(self, batch_size=2, dims=3):
        # loc and scale are Tensor with dtype 'VarType.FP32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        self.loc = paddle.to_tensor(self.loc_np)
        self.scale = paddle.to_tensor(self.scale_np)
        self.values = paddle.to_tensor(self.values_np)
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float32')
        self.other_loc = paddle.to_tensor(self.other_loc_np)
        self.other_scale = paddle.to_tensor(self.other_scale_np)

    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.FP32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float32')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float32')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np)
        self.dynamic_scale = paddle.to_tensor(self.scale_np)
        self.dynamic_values = paddle.to_tensor(self.values_np)
        self.dynamic_other_loc = paddle.to_tensor(self.other_loc_np)
        self.dynamic_other_scale = paddle.to_tensor(self.other_scale_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_loc = layers.data(
                name='loc', shape=[dims], dtype='float32')
            self.static_scale = layers.data(
                name='scale', shape=[dims], dtype='float32')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')
            self.static_other_loc = layers.data(
                name='other_loc', shape=[dims], dtype='float32')
            self.static_other_scale = layers.data(
                name='other_scale', shape=[dims], dtype='float32')


class NormalTest7(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.FP64'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float64')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np, dtype='float64')
        self.dynamic_scale = paddle.to_tensor(self.scale_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')
        self.dynamic_other_loc = paddle.to_tensor(
            self.other_loc_np, dtype='float64')
        self.dynamic_other_scale = paddle.to_tensor(
            self.other_scale_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_loc = layers.data(
                name='loc', shape=[dims], dtype='float64')
            self.static_scale = layers.data(
                name='scale', shape=[dims], dtype='float64')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float64')
            self.static_other_loc = layers.data(
                name='other_loc', shape=[dims], dtype='float64')
            self.static_other_scale = layers.data(
                name='other_scale', shape=[dims], dtype='float64')


class NormalTest8(NormalTest):
    def init_numpy_data(self, batch_size, dims):
        # loc and scale are Tensor with dtype 'VarType.FP64'. value's dtype is 'VarType.FP32'.
        self.loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.scale_np = np.random.randn(batch_size, dims).astype('float64')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')
        # used to construct another Normal object to calculate kl_divergence
        self.other_loc_np = np.random.randn(batch_size, dims).astype('float64')
        self.other_scale_np = np.random.randn(batch_size,
                                              dims).astype('float64')
        while not np.all(self.scale_np > 0):
            self.other_scale_np = np.random.randn(batch_size,
                                                  dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_loc = paddle.to_tensor(self.loc_np, dtype='float64')
        self.dynamic_scale = paddle.to_tensor(self.scale_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np)
        self.dynamic_other_loc = paddle.to_tensor(
            self.other_loc_np, dtype='float64')
        self.dynamic_other_scale = paddle.to_tensor(
            self.other_scale_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_loc = layers.data(
                name='loc', shape=[dims], dtype='float64')
            self.static_scale = layers.data(
                name='scale', shape=[dims], dtype='float64')
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32')
            self.static_other_loc = layers.data(
                name='other_loc', shape=[dims], dtype='float64')
            self.static_other_scale = layers.data(
                name='other_scale', shape=[dims], dtype='float64')


class DistributionTestError(unittest.TestCase):
    def test_distribution_error(self):
        distribution = Distribution()

        self.assertRaises(NotImplementedError, distribution.sample)
        self.assertRaises(NotImplementedError, distribution.entropy)

        normal = Normal(0.0, 1.0)
        self.assertRaises(NotImplementedError, distribution.kl_divergence,
                          normal)

        value_npdata = np.array([0.8], dtype="float32")
        value_tensor = layers.create_tensor(dtype="float32")
        self.assertRaises(NotImplementedError, distribution.log_prob,
                          value_tensor)
        self.assertRaises(NotImplementedError, distribution.probs, value_tensor)

    def test_normal_error(self):
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


if __name__ == '__main__':
    unittest.main()
