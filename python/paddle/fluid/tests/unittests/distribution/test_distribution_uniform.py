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

import numpy as np
from test_distribution import DistributionNumpy

import paddle
from paddle import fluid
from paddle.distribution import Uniform
from paddle.fluid import layers

np.random.seed(2022)


class UniformNumpy(DistributionNumpy):
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)
        if str(self.low.dtype) not in ['float32', 'float64']:
            self.low = self.low.astype('float32')
            self.high = self.high.astype('float32')

    def sample(self, shape):
        shape = tuple(shape) + (self.low + self.high).shape
        return self.low + (
            np.random.uniform(size=shape) * (self.high - self.low)
        )

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
        self.high_np = np.random.uniform(2, 4)
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
                name='values', shape=[], dtype='float32'
            )

    def compare_with_numpy(self, fetch_list, sample_shape=7, tolerance=1e-6):
        sample, entropy, log_prob, probs = fetch_list

        np_uniform = UniformNumpy(self.low_np, self.high_np)
        np_sample = np_uniform.sample([sample_shape])
        np_entropy = np_uniform.entropy()
        np_lp = np_uniform.log_prob(self.values_np)
        np_p = np_uniform.probs(self.values_np)

        np.testing.assert_equal(sample.shape, np_sample.shape)
        np.testing.assert_allclose(
            entropy, np_entropy, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            log_prob, np_lp, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(probs, np_p, rtol=tolerance, atol=tolerance)

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
            'values': self.values_np,
        }

        self.executor.run(fluid.default_startup_program())
        fetch_list = self.executor.run(
            program=self.test_program, feed=feed_vars, fetch_list=fetch_list
        )

        self.compare_with_numpy(fetch_list)

    def func_uniform_distribution_dygraph(self, sample_shape=7, tolerance=1e-6):
        paddle.disable_static()
        uniform = Uniform(self.dynamic_low, self.dynamic_high)
        sample = uniform.sample([sample_shape]).numpy()
        entropy = uniform.entropy().numpy()
        log_prob = uniform.log_prob(self.dynamic_values).numpy()
        probs = uniform.probs(self.dynamic_values).numpy()
        fetch_list = [sample, entropy, log_prob, probs]

        self.compare_with_numpy(fetch_list)

    def test_uniform_distribution_dygraph(self):
        self.setUp()
        self.func_uniform_distribution_dygraph()


class UniformTest2(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low ans high are 'int'
        self.low_np = int(np.random.uniform(-2, 1))
        self.high_np = int(np.random.uniform(2, 4))
        self.values_np = np.array([1.0]).astype('float32')


class UniformTest3(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # test broadcast: low is float, high is numpy.ndarray with dtype 'float32'.
        self.low_np = np.random.uniform(-2, 1)
        self.high_np = np.random.uniform(5.0, 15.0, (batch_size, dims)).astype(
            'float32'
        )
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32'
            )


class UniformTest4(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are numpy.ndarray with dtype 'float32'.
        self.low_np = np.random.randn(batch_size, dims).astype('float32')
        self.high_np = np.random.uniform(5.0, 15.0, (batch_size, dims)).astype(
            'float32'
        )
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32'
            )


class UniformTest5(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are numpy.ndarray with dtype 'float64'.
        self.low_np = np.random.randn(batch_size, dims).astype('float64')
        self.high_np = np.random.uniform(5.0, 15.0, (batch_size, dims)).astype(
            'float64'
        )
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
                name='values', shape=[dims], dtype='float64'
            )


class UniformTest6(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are Tensor with dtype 'VarType.FP32'.
        self.low_np = np.random.randn(batch_size, dims).astype('float32')
        self.high_np = np.random.uniform(5.0, 15.0, (batch_size, dims)).astype(
            'float32'
        )
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = paddle.to_tensor(self.low_np)
        self.dynamic_high = paddle.to_tensor(self.high_np)
        self.dynamic_values = paddle.to_tensor(self.values_np)

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_low = layers.data(
                name='low', shape=[dims], dtype='float32'
            )
            self.static_high = layers.data(
                name='high', shape=[dims], dtype='float32'
            )
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32'
            )


class UniformTest7(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are Tensor with dtype 'VarType.FP64'.
        self.low_np = np.random.randn(batch_size, dims).astype('float64')
        self.high_np = np.random.uniform(5.0, 15.0, (batch_size, dims)).astype(
            'float64'
        )
        self.values_np = np.random.randn(batch_size, dims).astype('float64')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = paddle.to_tensor(self.low_np, dtype='float64')
        self.dynamic_high = paddle.to_tensor(self.high_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float64')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_low = layers.data(
                name='low', shape=[dims], dtype='float64'
            )
            self.static_high = layers.data(
                name='high', shape=[dims], dtype='float64'
            )
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float64'
            )


class UniformTest8(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are Tensor with dtype 'VarType.FP64'. value's dtype is 'VarType.FP32'.
        self.low_np = np.random.randn(batch_size, dims).astype('float64')
        self.high_np = np.random.uniform(5.0, 15.0, (batch_size, dims)).astype(
            'float64'
        )
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_dynamic_data(self, batch_size, dims):
        self.dynamic_low = paddle.to_tensor(self.low_np, dtype='float64')
        self.dynamic_high = paddle.to_tensor(self.high_np, dtype='float64')
        self.dynamic_values = paddle.to_tensor(self.values_np, dtype='float32')

    def init_static_data(self, batch_size, dims):
        with fluid.program_guard(self.test_program):
            self.static_low = layers.data(
                name='low', shape=[dims], dtype='float64'
            )
            self.static_high = layers.data(
                name='high', shape=[dims], dtype='float64'
            )
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32'
            )


class UniformTest9(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are numpy.ndarray with dtype 'float32'.
        # high < low.
        self.low_np = np.random.randn(batch_size, dims).astype('float32')
        self.high_np = np.random.uniform(
            -10.0, -5.0, (batch_size, dims)
        ).astype('float32')
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32'
            )


class UniformTest10(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are list.
        self.low_np = (
            np.random.randn(batch_size, dims).astype('float32').tolist()
        )
        self.high_np = (
            np.random.uniform(5.0, 15.0, (batch_size, dims))
            .astype('float32')
            .tolist()
        )
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32'
            )


class UniformTest11(UniformTest):
    def init_numpy_data(self, batch_size, dims):
        # low and high are tuple.
        self.low_np = tuple(
            np.random.randn(batch_size, dims).astype('float32').tolist()
        )
        self.high_np = tuple(
            np.random.uniform(5.0, 15.0, (batch_size, dims))
            .astype('float32')
            .tolist()
        )
        self.values_np = np.random.randn(batch_size, dims).astype('float32')

    def init_static_data(self, batch_size, dims):
        self.static_low = self.low_np
        self.static_high = self.high_np
        with fluid.program_guard(self.test_program):
            self.static_values = layers.data(
                name='values', shape=[dims], dtype='float32'
            )


class UniformTestSample(unittest.TestCase):
    def setUp(self):
        self.init_param()

    def init_param(self):
        self.low = 3.0
        self.high = 4.0

    def test_uniform_sample(self):
        paddle.disable_static()
        uniform = Uniform(low=self.low, high=self.high)
        s = uniform.sample([100])
        self.assertTrue((s >= self.low).all())
        self.assertTrue((s < self.high).all())
        paddle.enable_static()


class UniformTestSample2(UniformTestSample):
    def init_param(self):
        self.low = -5.0
        self.high = 2.0


if __name__ == '__main__':
    unittest.main()
