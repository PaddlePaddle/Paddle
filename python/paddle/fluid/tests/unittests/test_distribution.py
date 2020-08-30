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
        lb = np.less(self.low, value).astype(value.dtype)
        ub = np.less(value, self.high).astype(value.dtype)
        return np.log(lb * ub) - np.log(self.high - self.low)

    def probs(self, value):
        lb = np.less(self.low, value).astype(value.dtype)
        ub = np.less(value, self.high).astype(value.dtype)
        return (lb * ub) / (self.high - self.low)

    def entropy(self):
        return np.log(self.high - self.low)


class NormalNumpy(DistributionNumpy):
    def __init__(self, loc, scale):
        self.loc = np.array(loc).astype('float32')
        self.scale = np.array(scale).astype('float32')

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
        return 0.5 + 0.5 * np.log(np.array(2. * math.pi).astype(
            'float32')) + np.log(self.scale)

    def kl_divergence(self, other):
        var_ratio = (self.scale / other.scale)
        var_ratio = var_ratio * var_ratio
        t1 = ((self.loc - other.loc) / other.scale)
        t1 = (t1 * t1)
        return 0.5 * (var_ratio + t1 - 1 - np.log(var_ratio))


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

    def build_normal_common_net(self, batch_size, dims, sample_shape, loc_float,
                                scale_float, other_loc_float, other_scale_float,
                                scale_np, scale_np_fp64, other_scale_np, loc_np,
                                loc_np_fp64, other_loc_np, loc, scale,
                                other_loc, other_scale, values, values_fp64):
        """Generate Normal object and get the output of its methods including
        ``sample``, ``entropy``, ``log_prob``, ``probs`` and ``kl_divergence``.
        Parameters ``loc`` and ``scale`` have different data types to test different situations.

        Args:
          batch_size(int): The first dimension of the shape of parameters(loc and scale).
          dims(int): The second dimension of the shape of parameters.
          sample_shape(int): The sample value used in ``sample`` method.
          loc_float(float): Generated in function ``get_normal_random_input``, loc is a float number.
          scale_float(float): Generated in function ``get_normal_random_input``, scale is a float number.
          other_loc_float(float): Generated in function ``get_normal_random_input``, other_loc is a
            float number. It is the first parameter in another Normal object used in ``kl_divergence``
            method.
          other_scale_float(float): Generated in function ``get_normal_random_input``, other_scale is a
            float number. It is the second parameter in another Normal object used in ``kl_divergence``
            method.
          scale_np(numpy.ndarray): Generated in function ``get_normal_random_input``, An numpy array
            whose shape is [batch_size, dims].
          scale_np_fp64(numpy.ndarray): Generated in function ``get_normal_random_input``, An numpy array
            whose shape is [batch_size, dims], and dtype is ``float64``.
          other_scale_np(numpy.ndarray): Generated in function ``get_normal_random_input``, other_scale_np
            is an numpy array. It is the second parameter in another Normal object used in ``kl_divergence``
            method.
          loc_np(numpy.ndarray): Generated in function ``get_normal_random_input``, An numpy array
            whose shape is [batch_size, dims].
          loc_np_fp64(numpy.ndarray): Generated in function ``get_normal_random_input``, An numpy array
            whose shape is [batch_size, dims], and dtype is ``float64``.
          other_loc_np(numpy.ndarray): Generated in function ``get_normal_random_input``, other_loc_np
            is an numpy array. It is the first parameter in another Normal object used in ``kl_divergence``
            method.
          loc(Tensor): In dynamic mode, loc is generated in ``build_normal_dygraph``, it's a Tensor filled
            with ``loc_np`` data. In static mode, loc is generated in ``build_normal_static``, ``layers.data``
             method is used to get a Placeholder whose shape is [dims].
          scale(Tensor): In dynamic mode, scale is generated in ``build_normal_dygraph``, it's a Tensor filled
            with ``scale_np`` data. In static mode, scale is generated in ``build_normal_static``, ``layers.data``
             method is used to get a Placeholder whose shape is [dims].
          other_loc(Tensor): In dynamic mode, other_loc is generated in ``build_normal_dygraph``, it's a Tensor
            filled with ``other_loc_np`` data. In static mode, other_loc is generated in ``build_normal_static``,
             ``layers.data`` method is used to get a Placeholder whose shape is [dims]. It is the first parameter
              in another Normal object used in ``kl_divergence`` method.
          other_scale(Tensor): In dynamic mode, other_scale is generated in ``build_normal_dygraph``, it's a Tensor
            filled with ``other_scale_np`` data. In static mode, other_scale is generated in ``build_normal_static``,
             ``layers.data`` method is used to get a Placeholder whose shape is [dims]. It is the second parameter
              in another Normal object used in ``kl_divergence`` method.
          values(Tensor): In dynamic mode, values is generated in ``build_normal_dygraph``, it's a Tensor filled with
             ``values_np`` data. In static mode, values is generated in ``build_normal_static``, ``layers.data``
             method is used to get a Placeholder whose shape is [dims].
          values_fp64(Tensor): In dynamic mode, values is generated in ``build_normal_dygraph``, it's a Tensor filled 
             with ``values_np`` data. In static mode, values is generated in ``build_normal_static``, ``layers.data``
             method is used to get a Placeholder whose shape is [dims]. The dtype is VarDesc.VarType.FP64.

        Returns:
          List: The elements of the list are the output of sample, entropy, log_prob, probs, kl_divergence methods.
          The inputs' type of these methods can be float, np.ndarray and Tensor. And broadcast will be considered.

        """
        normal_int = Normal(int(loc_float), int(scale_float))
        normal_float = Normal(loc_float, scale_float)
        other_normal_float = Normal(other_loc_float, other_scale_float)

        normal_float_np_broadcast = Normal(loc_float, scale_np)
        other_normal_float_np_broadcast = Normal(other_loc_float,
                                                 other_scale_np)

        normal_np = Normal(loc_np, scale_np)
        normal_np_fp64 = Normal(loc_np_fp64, scale_np_fp64)
        other_normal_np = Normal(other_loc_np, other_scale_np)

        normal_variable = Normal(loc, scale)
        other_normal_variable = Normal(other_loc, other_scale)

        sample_int = normal_int.sample([batch_size, dims])
        sample_float = normal_float.sample([batch_size, dims])
        sample_float_np_broadcast = normal_float_np_broadcast.sample(
            [batch_size, dims])
        sample_np = normal_np.sample([batch_size, dims])
        sample_variable = normal_variable.sample([batch_size, dims])

        sample_int_diff = normal_int.sample([sample_shape])
        sample_float_diff = normal_float.sample([sample_shape])
        sample_float_np_broadcast_diff = normal_float_np_broadcast.sample(
            [sample_shape])
        sample_np_diff = normal_np.sample([sample_shape])
        sample_variable_diff = normal_variable.sample([sample_shape])

        entropy_int = normal_int.entropy()
        entropy_float = normal_float.entropy()
        entropy_float_np_broadcast = normal_float_np_broadcast.entropy()
        entropy_np = normal_np.entropy()
        entropy_variable = normal_variable.entropy()

        lp_float_np_broadcast = normal_float_np_broadcast.log_prob(values)
        lp_np = normal_np.log_prob(values)
        lp_np_fp64 = normal_np_fp64.log_prob(values_fp64)
        lp_variable = normal_variable.log_prob(values)

        p_float_np_broadcast = normal_float_np_broadcast.probs(values)
        p_np = normal_np.probs(values)
        p_np_fp64 = normal_np_fp64.probs(values_fp64)
        p_variable = normal_variable.probs(values)

        kl_float = normal_float.kl_divergence(other_normal_float)
        kl_float_np_broadcast = normal_float_np_broadcast.kl_divergence(
            other_normal_float_np_broadcast)
        kl_np = normal_np.kl_divergence(other_normal_np)
        kl_variable = normal_variable.kl_divergence(other_normal_variable)

        fetch_list = [
            sample_int, sample_float, sample_float_np_broadcast, sample_np,
            sample_variable, sample_int_diff, sample_float_diff,
            sample_float_np_broadcast_diff, sample_np_diff,
            sample_variable_diff, entropy_int, entropy_float,
            entropy_float_np_broadcast, entropy_np, entropy_variable,
            lp_float_np_broadcast, lp_np, lp_np_fp64, lp_variable,
            p_float_np_broadcast, p_np, p_np_fp64, p_variable, kl_float,
            kl_float_np_broadcast, kl_np, kl_variable
        ]
        return fetch_list

    def build_normal_static(self, test_program, batch_size, dims, sample_shape,
                            loc_float, scale_float, other_loc_float,
                            other_scale_float, scale_np, scale_np_fp64,
                            other_scale_np, loc_np, loc_np_fp64, other_loc_np,
                            values_np, values_np_fp64):
        """
        In static mode, generate feed data of Normal network, and get output fetch_list using
        ``build_normal_common_net``.

        Args:
          test_program: In static mode, the Program object.
          other args can refer to function ``build_normal_common_net``.

        Returns:
          feed_vars: The feed data of Normal network in static mode.
          fetch_list: The output is generated by function ``build_normal_common_net``.
        """
        with fluid.program_guard(test_program):
            loc = layers.data(name='loc', shape=[dims], dtype='float32')
            scale = layers.data(name='scale', shape=[dims], dtype='float32')

            other_loc = layers.data(
                name='other_loc', shape=[dims], dtype='float32')
            other_scale = layers.data(
                name='other_scale', shape=[dims], dtype='float32')

            values = layers.data(name='values', shape=[dims], dtype='float32')
            values_fp64 = layers.data(
                name='values_fp64', shape=[dims], dtype='float64')

            fetch_list = self.build_normal_common_net(
                batch_size, dims, sample_shape, loc_float, scale_float,
                other_loc_float, other_scale_float, scale_np, scale_np_fp64,
                other_scale_np, loc_np, loc_np_fp64, other_loc_np, loc, scale,
                other_loc, other_scale, values, values_fp64)

        feed_vars = {
            'loc': loc_np,
            'scale': scale_np,
            'other_loc': other_loc_np,
            'other_scale': other_scale_np,
            'values': values_np,
            'values_fp64': values_np_fp64
        }
        return feed_vars, fetch_list

    def build_normal_dygraph(self, batch_size, dims, sample_shape, loc_float,
                             scale_float, other_loc_float, other_scale_float,
                             scale_np, scale_np_fp64, other_scale_np, loc_np,
                             loc_np_fp64, other_loc_np, values_np,
                             values_np_fp64):
        """
        In dynamic mode, generate input data of Normal network, and get output fetch_list using
        ``build_normal_common_net``.

        Args:
          refer to function ``build_normal_common_net``.

        Returns:
          fetch_list_numpy: The output is generated by function ``build_normal_common_net``. Transform
          these tensor to numpy.ndarray.
        """
        loc = paddle.to_tensor(loc_np)
        scale = paddle.to_tensor(scale_np)
        other_loc = paddle.to_tensor(other_loc_np)
        other_scale = paddle.to_tensor(other_scale_np)
        values = paddle.to_tensor(values_np)
        values_fp64 = paddle.to_tensor(values_np_fp64)

        fetch_list = self.build_normal_common_net(
            batch_size, dims, sample_shape, loc_float, scale_float,
            other_loc_float, other_scale_float, scale_np, scale_np_fp64,
            other_scale_np, loc_np, loc_np_fp64, other_loc_np, loc, scale,
            other_loc, other_scale, values, values_fp64)
        fetch_list_numpy = [t.numpy() for t in fetch_list]
        return fetch_list_numpy

    def get_normal_random_input(self, batch_size, dims):
        """
        Generate input data ``loc`` and ``scale`` used in Normal network.

        Args:
          refer to function ``build_normal_common_net``.

        Returns:
          List: Different data type of ``loc`` and ``scale``, including float, numpy.ndarray.
          By the way, ``other_loc`` and ``other_scale`` are used in ``kl_divergence`` method.
          refer to ``args`` in function ``build_normal_common_net``.
        """
        loc_np = np.random.randn(batch_size, dims).astype('float32')
        loc_np_fp64 = np.random.randn(batch_size, dims).astype('float64')
        other_loc_np = np.random.randn(batch_size, dims).astype('float32')

        loc_float = (np.random.ranf() - 0.5) * 4
        scale_float = (np.random.ranf() - 0.5) * 4
        while scale_float < 0:
            scale_float = (np.random.ranf() - 0.5) * 4

        other_loc_float = (np.random.ranf() - 0.5) * 4
        other_scale_float = (np.random.ranf() - 0.5) * 4
        while other_scale_float < 0:
            other_scale_float = (np.random.ranf() - 0.5) * 4

        scale_np = np.random.randn(batch_size, dims).astype('float32')
        scale_np_fp64 = np.random.randn(batch_size, dims).astype('float64')
        other_scale_np = np.random.randn(batch_size, dims).astype('float32')
        values_np = np.random.randn(batch_size, dims).astype('float32')
        values_np_fp64 = np.random.randn(batch_size, dims).astype('float64')

        while not np.all(scale_np > 0):
            scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(scale_np_fp64 > 0):
            scale_np_fp64 = np.random.randn(batch_size, dims).astype('float64')
        while not np.all(other_scale_np > 0):
            other_scale_np = np.random.randn(batch_size, dims).astype('float32')
        return [
            loc_np, loc_np_fp64, other_loc_np, loc_float, scale_float,
            other_loc_float, other_scale_float, scale_np, scale_np_fp64,
            other_scale_np, values_np, values_np_fp64
        ]

    def compare_normal_with_numpy(self,
                                  data_list,
                                  output_list,
                                  batch_size=2,
                                  dims=3,
                                  sample_shape=7,
                                  tolerance=1e-6):
        """
        Compare the outputs of Normal's methods in paddle and numpy. If the outputs are not consistent,
        raise errors.

        Args:
          data_list: Input data generated by function ``get_normal_random_input``.
          output_list: The outputs of Normal's methods in static or dynamic mode.
          batch_size(int): The first dimension of the shape of parameters(loc and scale).
          dims(int): The second dimension of the shape of parameters.
          sample_shape(int): The sample value used in ``sample`` method.
          tolerance(float): The tolerance of the error.
        """
        loc_np, loc_np_fp64, other_loc_np, loc_float, scale_float, other_loc_float, other_scale_float, scale_np, scale_np_fp64, other_scale_np, values_np, values_np_fp64 = data_list

        np_normal_int = NormalNumpy(int(loc_float), int(scale_float))
        np_normal_float = NormalNumpy(loc_float, scale_float)
        np_other_normal_float = NormalNumpy(other_loc_float, other_scale_float)
        np_normal_float_np_broadcast = NormalNumpy(loc_float, scale_np)
        np_other_normal_float_np_broadcast = NormalNumpy(other_loc_float,
                                                         other_scale_np)
        np_normal = NormalNumpy(loc_np, scale_np)
        np_normal_fp64 = NormalNumpy(loc_np_fp64, scale_np_fp64)
        np_other_normal = NormalNumpy(other_loc_np, other_scale_np)

        gt_sample_int = np_normal_int.sample([batch_size, dims])
        gt_sample_float = np_normal_float.sample([batch_size, dims])
        gt_sample_float_np_broadcast = np_normal_float_np_broadcast.sample(
            [batch_size, dims])
        gt_sample_np = np_normal.sample([batch_size, dims])

        gt_sample_int_diff = np_normal_int.sample([sample_shape])
        gt_sample_float_diff = np_normal_float.sample([sample_shape])
        gt_sample_float_np_broadcast_diff = np_normal_float_np_broadcast.sample(
            [sample_shape])
        gt_sample_np_diff = np_normal.sample([sample_shape])

        gt_entropy_int = np_normal_int.entropy()
        gt_entropy_float = np_normal_float.entropy()
        gt_entropy_float_np_broadcast = np_normal_float_np_broadcast.entropy()
        gt_entropy = np_normal.entropy()
        gt_lp_float_np_broadcast = np_normal_float_np_broadcast.log_prob(
            values_np)
        gt_lp = np_normal.log_prob(values_np)
        gt_lp_fp64 = np_normal_fp64.log_prob(values_np_fp64)
        gt_p_float_np_broadcast = np_normal_float_np_broadcast.probs(values_np)
        gt_p = np_normal.probs(values_np)
        gt_p_fp64 = np_normal_fp64.probs(values_np_fp64)
        gt_kl_float = np_normal_float.kl_divergence(np_other_normal_float)
        gt_kl_float_np_broadcast = np_normal_float_np_broadcast.kl_divergence(
            np_other_normal_float_np_broadcast)
        gt_kl = np_normal.kl_divergence(np_other_normal)

        [
            output_sample_int, output_sample_float,
            output_sample_float_np_broadcast, output_sample_np,
            output_sample_variable, output_sample_int_diff,
            output_sample_float_diff, output_sample_float_np_broadcast_diff,
            output_sample_np_diff, output_sample_variable_diff,
            output_entropy_int, output_entropy_float,
            output_entropy_float_np_broadcast, output_entropy_np,
            output_entropy_variable, output_lp_float_np_broadcast, output_lp_np,
            output_lp_np_fp64, output_lp_variable, output_p_float_np_broadcast,
            output_p_np, output_p_np_fp64, output_p_variable, output_kl_float,
            output_kl_float_np_broadcast, output_kl_np, output_kl_variable
        ] = output_list

        np.testing.assert_equal(output_sample_int.shape, gt_sample_int.shape)
        np.testing.assert_equal(output_sample_float.shape,
                                gt_sample_float.shape)
        np.testing.assert_equal(output_sample_float_np_broadcast.shape,
                                gt_sample_float_np_broadcast.shape)
        np.testing.assert_equal(output_sample_np.shape, gt_sample_np.shape)
        np.testing.assert_equal(output_sample_variable.shape,
                                gt_sample_np.shape)
        np.testing.assert_equal(output_sample_int_diff.shape,
                                gt_sample_int_diff.shape)
        np.testing.assert_equal(output_sample_float_diff.shape,
                                gt_sample_float_diff.shape)
        np.testing.assert_equal(output_sample_float_np_broadcast_diff.shape,
                                gt_sample_float_np_broadcast_diff.shape)
        np.testing.assert_equal(output_sample_np_diff.shape,
                                gt_sample_np_diff.shape)
        np.testing.assert_equal(output_sample_variable_diff.shape,
                                gt_sample_np_diff.shape)
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
            output_lp_np_fp64, gt_lp_fp64, rtol=tolerance, atol=tolerance)
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
            output_p_np_fp64, gt_p_fp64, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_p_variable, gt_p, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_kl_float, gt_kl_float, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_kl_float_np_broadcast,
            gt_kl_float_np_broadcast,
            rtol=tolerance,
            atol=tolerance)
        np.testing.assert_allclose(
            output_kl_np, gt_kl, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_kl_variable, gt_kl, rtol=tolerance, atol=tolerance)

    def test_normal_distribution_static(self,
                                        batch_size=2,
                                        dims=3,
                                        sample_shape=7,
                                        tolerance=1e-6):
        """
        Test Normal's methods in static mode.

        Args:
          refer to ``compare_normal_with_numpy`` function.
        """
        test_program = fluid.Program()
        data_list = self.get_normal_random_input(batch_size, dims)
        loc_np, loc_np_fp64, other_loc_np, loc_float, scale_float, other_loc_float, other_scale_float, scale_np, scale_np_fp64, other_scale_np, values_np, values_np_fp64 = data_list

        feed_vars, fetch_list = self.build_normal_static(
            test_program, batch_size, dims, sample_shape, loc_float,
            scale_float, other_loc_float, other_scale_float, scale_np,
            scale_np_fp64, other_scale_np, loc_np, loc_np_fp64, other_loc_np,
            values_np, values_np_fp64)
        self.executor.run(fluid.default_startup_program())

        output_list = self.executor.run(program=test_program,
                                        feed=feed_vars,
                                        fetch_list=fetch_list)

        self.compare_normal_with_numpy(data_list, output_list, batch_size, dims,
                                       sample_shape, tolerance)

    def test_normal_distribution_dygraph(self,
                                         batch_size=2,
                                         dims=3,
                                         sample_shape=7,
                                         tolerance=1e-6):
        """
        Test Normal's methods in dynamic mode.

        Args:
          refer to ``compare_normal_with_numpy`` function.
        """
        paddle.disable_static()
        data_list = self.get_normal_random_input(batch_size, dims)
        loc_np, loc_np_fp64, other_loc_np, loc_float, scale_float, other_loc_float, other_scale_float, scale_np, scale_np_fp64, other_scale_np, values_np, values_np_fp64 = data_list

        output_list = self.build_normal_dygraph(
            batch_size, dims, sample_shape, loc_float, scale_float,
            other_loc_float, other_scale_float, scale_np, scale_np_fp64,
            other_scale_np, loc_np, loc_np_fp64, other_loc_np, values_np,
            values_np_fp64)

        self.compare_normal_with_numpy(data_list, output_list, batch_size, dims,
                                       sample_shape, tolerance)
        paddle.enable_static()

    def build_uniform_common_net(self, batch_size, dims, sample_shape,
                                 low_float, high_float, high_np, high_np_fp64,
                                 low_np, low_np_fp64, values_np, values_np_fp64,
                                 low, high, values, values_fp64):
        """Generate Uniform object and get the output of its methods including ``sample``, ``entropy``,
         ``log_prob`` and ``probs``.
        Parameters ``low`` and ``high`` have different data types to test different situations.

        Args:
          batch_size(int): The first dimension of the shape of parameters(low and high).
          dims(int): The second dimension of the shape of parameters.
          sample_shape(int): The sample value used in ``sample`` method.
          low_float(float): Parameter ``low`` is a float number.
          high_float(float): Parameter ``high`` is a float number.
          high_np(numpy.ndarray): An numpy array whose shape is [batch_size, dims].
          low_np(numpy.ndarray): An numpy array whose shape is [batch_size, dims].
          values_np(numpy.ndarray): The input of ``log_prob`` and ``probs`` methods. An numpy array whose
            shape is [batch_size, dims].
          low(Tensor): In dynamic mode, low is generated in ``build_uniform_dygraph``, it's a Tensor filled
            with ``low_np`` data. In static mode, low is generated in ``build_uniform_static``.
          high(Tensor): In dynamic mode, high is generated in ``build_uniform_dygraph``, it's a Tensor filled
            with ``high_np`` data. In static mode, high is generated in ``build_uniform_static``.
          values(Tensor): In dynamic mode, values is generated in ``build_uniform_dygraph``, it's a Tensor
            filled with ``values_np`` data. In static mode, values is generated in ``build_uniform_static``.

        Returns:
          List: The elements of the list are the output of sample, entropy, log_prob, probs methods.
          The inputs' type of these methods can be float, np.ndarray and Tensor. And broadcast will be
           considered.

        """
        uniform_int = Uniform(int(low_float), int(high_float))
        uniform_float = Uniform(low_float, high_float)
        uniform_float_np_broadcast = Uniform(low_float, high_np)
        uniform_np = Uniform(low_np, high_np)
        uniform_np_fp64 = Uniform(low_np_fp64, high_np_fp64)
        uniform_variable = Uniform(low, high)

        sample_int = uniform_int.sample([batch_size, dims])
        sample_float = uniform_float.sample([batch_size, dims])
        sample_float_np_broadcast = uniform_float_np_broadcast.sample(
            [batch_size, dims])
        sample_np = uniform_np.sample([batch_size, dims])
        sample_variable = uniform_variable.sample([batch_size, dims])

        sample_int_diff = uniform_int.sample([sample_shape])
        sample_float_diff = uniform_float.sample([sample_shape])
        sample_float_np_broadcast_diff = uniform_float_np_broadcast.sample(
            [sample_shape])
        sample_np_diff = uniform_np.sample([sample_shape])
        sample_variable_diff = uniform_variable.sample([sample_shape])

        entropy_int = uniform_int.entropy()
        entropy_float = uniform_float.entropy()
        entropy_float_np_broadcast = uniform_float_np_broadcast.entropy()
        entropy_np = uniform_np.entropy()
        entropy_variable = uniform_variable.entropy()

        lp_float_np_broadcast = uniform_float_np_broadcast.log_prob(values)
        lp_np = uniform_np.log_prob(values)
        lp_np_fp64 = uniform_np_fp64.log_prob(values_fp64)
        lp_variable = uniform_variable.log_prob(values)

        p_float_np_broadcast = uniform_float_np_broadcast.probs(values)
        p_np = uniform_np.probs(values)
        p_np_fp64 = uniform_np_fp64.probs(values_fp64)
        p_variable = uniform_variable.probs(values)

        fetch_list = [
            sample_int, sample_float, sample_float_np_broadcast, sample_np,
            sample_variable, sample_int_diff, sample_float_diff,
            sample_float_np_broadcast_diff, sample_np_diff,
            sample_variable_diff, entropy_int, entropy_float,
            entropy_float_np_broadcast, entropy_np, entropy_variable,
            lp_float_np_broadcast, lp_np, lp_np_fp64, lp_variable,
            p_float_np_broadcast, p_np, p_np_fp64, p_variable
        ]
        return fetch_list

    def build_uniform_static(self, test_program, batch_size, dims, sample_shape,
                             low_float, high_float, high_np, high_np_fp64,
                             low_np, low_np_fp64, values_np, values_np_fp64):
        """
        In static mode, generate feed data of Uniform network, and get output fetch_list using
        ``build_uniform_common_net``.

        Args:
          test_program: In static mode, the Program object.
          other args can refer to function ``build_uniform_common_net``.

        Returns:
          feed_vars: The feed data of Uniform network in static mode.
          fetch_list: The output is generated by function ``build_uniform_common_net``.
        """
        with fluid.program_guard(test_program):
            low = layers.data(name='low', shape=[dims], dtype='float32')
            high = layers.data(name='high', shape=[dims], dtype='float32')

            values = layers.data(name='values', shape=[dims], dtype='float32')
            values_fp64 = layers.data(
                name='values_fp64', shape=[dims], dtype='float64')

            fetch_list = self.build_uniform_common_net(
                batch_size, dims, sample_shape, low_float, high_float, high_np,
                high_np_fp64, low_np, low_np_fp64, values_np, values_np_fp64,
                low, high, values, values_fp64)

        feed_vars = {
            'low': low_np,
            'high': high_np,
            'values': values_np,
            'values_fp64': values_np_fp64
        }
        return feed_vars, fetch_list

    def build_uniform_dygraph(self, batch_size, dims, sample_shape, low_float,
                              high_float, high_np, high_np_fp64, low_np,
                              low_np_fp64, values_np, values_np_fp64):
        """
        In dynamic mode, generate input data of Uniform network, and get output fetch_list using
        ``build_uniform_common_net``.

        Args:
          refer to function ``build_uniform_common_net``.

        Returns:
          fetch_list_numpy: The output is generated by function ``build_uniform_common_net``. Transform
          these tensor to numpy.ndarray.
        """
        low = paddle.to_tensor(low_np)
        high = paddle.to_tensor(high_np)
        values = paddle.to_tensor(values_np)
        values_fp64 = paddle.to_tensor(values_np_fp64)

        fetch_list = self.build_uniform_common_net(
            batch_size, dims, sample_shape, low_float, high_float, high_np,
            high_np_fp64, low_np, low_np_fp64, values_np, values_np_fp64, low,
            high, values, values_fp64)
        fetch_list_numpy = [t.numpy() for t in fetch_list]
        return fetch_list_numpy

    def compare_uniform_with_numpy(self,
                                   data_list,
                                   output_list,
                                   batch_size=2,
                                   dims=3,
                                   sample_shape=7,
                                   tolerance=1e-6):
        """
        Compare the outputs of Uniform's methods in paddle and numpy. If the outputs are not consistent,
        raise errors.

        Args:
          data_list: Input data including float and numpy.ndarray type of ``low`` and ``high`` parameters.
          output_list: The outputs of Uniform's methods in static or dynamic mode.
          batch_size(int): The first dimension of the shape of parameters(low and high).
          dims(int): The second dimension of the shape of parameters.
          sample_shape(int): The sample value used in ``sample`` method.
          tolerance(float): The tolerance of the error.
        """
        [
            low_np, low_np_fp64, low_float, high_float, high_np, high_np_fp64,
            values_np, values_np_fp64
        ] = data_list

        np_uniform_int = UniformNumpy(int(low_float), int(high_float))
        np_uniform_float = UniformNumpy(low_float, high_float)
        np_uniform_float_np_broadcast = UniformNumpy(low_float, high_np)
        np_uniform = UniformNumpy(low_np, high_np)
        np_uniform_fp64 = UniformNumpy(low_np_fp64, high_np_fp64)

        gt_sample_int = np_uniform_int.sample([batch_size, dims])
        gt_sample_float = np_uniform_float.sample([batch_size, dims])
        gt_sample_float_np_broadcast = np_uniform_float_np_broadcast.sample(
            [batch_size, dims])
        gt_sample_np = np_uniform.sample([batch_size, dims])
        gt_sample_int_diff = np_uniform_int.sample([sample_shape])
        gt_sample_float_diff = np_uniform_float.sample([sample_shape])
        gt_sample_float_np_broadcast_diff = np_uniform_float_np_broadcast.sample(
            [sample_shape])
        gt_sample_np_diff = np_uniform.sample([sample_shape])
        gt_entropy_int = np_uniform_int.entropy()
        gt_entropy_float = np_uniform_float.entropy()
        gt_entropy_float_np_broadcast = np_uniform_float_np_broadcast.entropy()
        gt_entropy = np_uniform.entropy()
        gt_lp_float_np_broadcast = np_uniform_float_np_broadcast.log_prob(
            values_np)
        gt_lp = np_uniform.log_prob(values_np)
        gt_lp_fp64 = np_uniform_fp64.log_prob(values_np_fp64)
        gt_p_float_np_broadcast = np_uniform_float_np_broadcast.probs(values_np)
        gt_p = np_uniform.probs(values_np)
        gt_p_fp64 = np_uniform_fp64.probs(values_np_fp64)

        [
            output_sample_int, output_sample_float,
            output_sample_float_np_broadcast, output_sample_np,
            output_sample_variable, output_sample_int_diff,
            output_sample_float_diff, output_sample_float_np_broadcast_diff,
            output_sample_np_diff, output_sample_variable_diff,
            output_entropy_int, output_entropy_float,
            output_entropy_float_np_broadcast, output_entropy_np,
            output_entropy_variable, output_lp_float_np_broadcast, output_lp_np,
            output_lp_np_fp64, output_lp_variable, output_p_float_np_broadcast,
            output_p_np, output_p_np_fp64, output_p_variable
        ] = output_list

        np.testing.assert_equal(output_sample_int.shape, gt_sample_int.shape)
        np.testing.assert_equal(output_sample_float.shape,
                                gt_sample_float.shape)
        np.testing.assert_equal(output_sample_float_np_broadcast.shape,
                                gt_sample_float_np_broadcast.shape)
        np.testing.assert_equal(output_sample_np.shape, gt_sample_np.shape)
        np.testing.assert_equal(output_sample_variable.shape,
                                gt_sample_np.shape)
        np.testing.assert_equal(output_sample_int_diff.shape,
                                gt_sample_int_diff.shape)
        np.testing.assert_equal(output_sample_float_diff.shape,
                                gt_sample_float_diff.shape)
        np.testing.assert_equal(output_sample_float_np_broadcast_diff.shape,
                                gt_sample_float_np_broadcast_diff.shape)
        np.testing.assert_equal(output_sample_np_diff.shape,
                                gt_sample_np_diff.shape)
        np.testing.assert_equal(output_sample_variable_diff.shape,
                                gt_sample_np_diff.shape)
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
            output_lp_np_fp64, gt_lp_fp64, rtol=tolerance, atol=tolerance)
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
            output_p_np_fp64, gt_p_fp64, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(
            output_p_variable, gt_p, rtol=tolerance, atol=tolerance)

    def test_uniform_distribution_static(self,
                                         batch_size=2,
                                         dims=3,
                                         sample_shape=7,
                                         tolerance=1e-6):
        """
        Test Uniform's methods in static mode.

        Args:
          refer to ``compare_uniform_with_numpy`` function.
        """
        test_program = fluid.Program()

        low_np = np.random.randn(batch_size, dims).astype('float32')
        low_np_fp64 = np.random.randn(batch_size, dims).astype('float64')
        low_float = np.random.uniform(-2, 1)
        high_float = np.random.uniform(1, 3)
        high_np = np.random.uniform(-5.0, 5.0,
                                    (batch_size, dims)).astype('float32')
        high_np_fp64 = np.random.uniform(-5.0, 5.0,
                                         (batch_size, dims)).astype('float64')
        values_np = np.random.randn(batch_size, dims).astype('float32')
        values_np_fp64 = np.random.randn(batch_size, dims).astype('float64')

        data_list = [
            low_np, low_np_fp64, low_float, high_float, high_np, high_np_fp64,
            values_np, values_np_fp64
        ]

        feed_vars, fetch_list = self.build_uniform_static(
            test_program, batch_size, dims, sample_shape, low_float, high_float,
            high_np, high_np_fp64, low_np, low_np_fp64, values_np,
            values_np_fp64)

        self.executor.run(fluid.default_startup_program())

        # result calculated by paddle
        output_list = self.executor.run(program=test_program,
                                        feed=feed_vars,
                                        fetch_list=fetch_list)
        self.compare_uniform_with_numpy(data_list, output_list, batch_size,
                                        dims, sample_shape, tolerance)

    def test_uniform_distribution_dygraph(self,
                                          batch_size=2,
                                          dims=3,
                                          sample_shape=7,
                                          tolerance=1e-6):
        """
        Test Uniform's methods in dynamic mode.

        Args:
          refer to ``compare_uniform_with_numpy`` function.
        """
        paddle.disable_static()

        low_np = np.random.randn(batch_size, dims).astype('float32')
        low_np_fp64 = np.random.randn(batch_size, dims).astype('float64')
        low_float = np.random.uniform(-2, 1)
        high_float = np.random.uniform(1, 3)
        high_np = np.random.uniform(-5.0, 5.0,
                                    (batch_size, dims)).astype('float32')
        high_np_fp64 = np.random.uniform(-5.0, 5.0,
                                         (batch_size, dims)).astype('float64')
        values_np = np.random.randn(batch_size, dims).astype('float32')
        values_np_fp64 = np.random.randn(batch_size, dims).astype('float64')

        data_list = [
            low_np, low_np_fp64, low_float, high_float, high_np, high_np_fp64,
            values_np, values_np_fp64
        ]
        output_list = self.build_uniform_dygraph(
            batch_size, dims, sample_shape, low_float, high_float, high_np,
            high_np_fp64, low_np, low_np_fp64, values_np, values_np_fp64)

        self.compare_uniform_with_numpy(data_list, output_list, batch_size,
                                        dims, sample_shape, tolerance)
        paddle.enable_static()


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
