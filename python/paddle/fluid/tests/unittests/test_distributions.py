#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.distribution import *
import torch
import torch.distributions as td

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

    def test_normal_distribution(self, batch_size=1, dims=3, decimal=5):
        test_program = fluid.Program()
        loc_np = np.random.randn(batch_size, dims).astype('float32')
        other_loc_np = np.random.randn(batch_size,
                                       dims).astype('float32')

        loc_float = (np.random.ranf() - 0.5) * 4
        scale_float = (np.random.ranf() - 0.5) * 4
        while scale_float < 0:
            scale_float = (np.random.ranf() - 0.5) * 4

        other_loc_float = (np.random.ranf() - 0.5) * 4
        other_scale_float = (np.random.ranf() - 0.5) * 4
        while other_scale_float < 0:
            other_scale_float = (np.random.ranf() - 0.5) * 4

        scale_np = np.random.randn(batch_size, dims).astype('float32')
        other_scale_np = np.random.randn(batch_size,
                                         dims).astype('float32')

        values_np = np.random.randn(batch_size, dims).astype('float32')

        while not np.all(scale_np > 0):
            scale_np = np.random.randn(batch_size, dims).astype('float32')
        while not np.all(other_scale_np > 0):
            other_scale_np = np.random.randn(batch_size,
                                             dims).astype('float32')

        with fluid.program_guard(test_program):
            loc = layers.data(
                name='loc', shape=[dims], dtype='float32')
            scale = layers.data(
                name='scale', shape=[dims], dtype='float32')

            other_loc = layers.data(
                name='other_loc', shape=[dims], dtype='float32')
            other_scale = layers.data(
                name='other_scale', shape=[dims], dtype='float32')

            values = layers.data(name='values', shape=[dims], dtype='float32')

            normal_float = Normal(loc_float, scale_float)
            other_normal_float = Normal(other_loc_float, other_scale_float)

            normal_float_np_broadcast = Normal(loc_float, scale_np)
            other_normal_float_np_broadcast = Normal(other_loc_float, other_scale_np)

            normal_np = Normal(loc_np, scale_np)
            other_normal_np = Normal(other_loc_np, other_scale_np)

            normal_variable = Normal(loc, scale)
            other_normal_variable = Normal(other_loc, other_scale)

            sample_float = normal_float.sample([batch_size, dims])
            sample_float_np_broadcast = normal_float_np_broadcast.sample([batch_size, dims])
            sample_np = normal_np.sample([batch_size, dims])
            sample_variable = normal_variable.sample([batch_size, dims])

            entropy_float = normal_float.entropy()
            entropy_float_np_broadcast = normal_float_np_broadcast.entropy()
            entropy_np = normal_np.entropy()
            entropy_variable = normal_variable.entropy()

            # lp_float = normal_float.log_prob(values)
            lp_float_np_broadcast = normal_float_np_broadcast.log_prob(values)
            lp_np = normal_np.log_prob(values)
            lp_variable = normal_variable.log_prob(values)

            kl_float = normal_float.kl_divergence(other_normal_float)
            kl_float_np_broadcast = normal_float_np_broadcast.kl_divergence(other_normal_float_np_broadcast)
            kl_np = normal_np.kl_divergence(other_normal_np)
            kl_variable = normal_variable.kl_divergence(other_normal_variable)

        self.executor.run(fluid.default_startup_program())
        if self.use_gpu:
            normal_float_torch = td.Normal(torch.FloatTensor(np.array(loc_float)).cuda(),
                                           torch.FloatTensor(np.array(scale_float)).cuda())
            other_normal_float_torch = td.Normal(torch.FloatTensor(np.array(other_loc_float)).cuda(),
                                                 torch.FloatTensor(np.array(other_scale_float)).cuda())

            normal_float_np_broadcast_torch = td.Normal(torch.FloatTensor(np.array(loc_float)).cuda(),
                                                        torch.FloatTensor(np.array(scale_np)).cuda())
            other_normal_float_np_broadcast_torch = td.Normal(torch.FloatTensor(np.array(other_loc_float)).cuda(),
                                                              torch.FloatTensor(np.array(other_scale_np)).cuda())

            normal_np_torch = td.Normal(torch.FloatTensor(np.array(loc_np)).cuda(),
                                        torch.FloatTensor(np.array(scale_np)).cuda())
            other_normal_np_torch = td.Normal(torch.FloatTensor(np.array(other_loc_np)).cuda(),
                                              torch.FloatTensor(np.array(other_scale_np)).cuda())

            gt_lp_float_np_broadcast = normal_float_np_broadcast_torch.log_prob(
                torch.FloatTensor(np.array(values_np)).cuda()).cpu().numpy()
            gt_lp_np = gt_lp_variable = normal_np_torch.log_prob(
                torch.FloatTensor(np.array(values_np)).cuda()).cpu().numpy()
        else:
            normal_float_torch = td.Normal(torch.FloatTensor(np.array(loc_float)),
                                           torch.FloatTensor(np.array(scale_float)))
            other_normal_float_torch = td.Normal(torch.FloatTensor(np.array(other_loc_float)),
                                                 torch.FloatTensor(np.array(other_scale_float)))

            normal_float_np_broadcast_torch = td.Normal(torch.FloatTensor(np.array(loc_float)),
                                                        torch.FloatTensor(np.array(scale_np)))
            other_normal_float_np_broadcast_torch = td.Normal(torch.FloatTensor(np.array(other_loc_float)),
                                                              torch.FloatTensor(np.array(other_scale_np)))

            normal_np_torch = td.Normal(torch.FloatTensor(np.array(loc_np)), torch.FloatTensor(np.array(scale_np)))
            other_normal_np_torch = td.Normal(torch.FloatTensor(np.array(other_loc_np)),
                                              torch.FloatTensor(np.array(other_scale_np)))

            gt_lp_float_np_broadcast = normal_float_np_broadcast_torch.log_prob(
                torch.FloatTensor(np.array(values_np))).cpu().numpy()
            gt_lp_np = gt_lp_variable = normal_np_torch.log_prob(torch.FloatTensor(np.array(values_np))).cpu().numpy()

        gt_sample_float = normal_float_torch.sample([batch_size, dims]).cpu().numpy()
        gt_sample_float_np_broadcast = normal_float_np_broadcast_torch.sample([batch_size, dims]).cpu().numpy()
        gt_sample_np = gt_sample_variable = normal_np_torch.sample([batch_size, dims]).cpu().numpy()

        gt_entropy_float = normal_float_torch.entropy().cpu().numpy()
        gt_entropy_float_np_broadcast = normal_float_np_broadcast_torch.entropy().cpu().numpy()
        gt_entropy_np = gt_entropy_variable = normal_np_torch.entropy().cpu().numpy()

        gt_kl_float = td.kl_divergence(normal_float_torch, other_normal_float_torch).cpu().numpy()
        gt_kl_float_np_broadcast = td.kl_divergence(normal_float_np_broadcast_torch,
                                                    other_normal_float_np_broadcast_torch).cpu().numpy()
        gt_kl_np = gt_kl_variable = td.kl_divergence(normal_np_torch, other_normal_np_torch).cpu().numpy()

        # result calculated by CategoricalDistribution
        [output_sample_float,
         output_sample_float_np_broadcast,
         output_sample_np,
         output_sample_variable,
         output_entropy_float,
         output_entropy_float_np_broadcast,
         output_entropy_np,
         output_entropy_variable,
         output_lp_float_np_broadcast,
         output_lp_np,
         output_lp_variable,
         output_kl_float,
         output_kl_float_np_broadcast,
         output_kl_np,
         output_kl_variable
         ] = self.executor.run(
            program=test_program,
            feed={
                'loc': loc_np,
                'scale': scale_np,
                'other_loc': other_loc_np,
                'other_scale': other_scale_np,
                'values': values_np
            },
            fetch_list=[
                sample_float,
                sample_float_np_broadcast,
                sample_np,
                sample_variable,
                entropy_float,
                entropy_float_np_broadcast,
                entropy_np,
                entropy_variable,
                lp_float_np_broadcast,
                lp_np,
                lp_variable,
                kl_float,
                kl_float_np_broadcast,
                kl_np,
                kl_variable
            ])

        # test entropy
        np.testing.assert_almost_equal(output_sample_float.shape, gt_sample_float.shape, decimal)
        np.testing.assert_almost_equal(output_entropy_float, gt_entropy_float, decimal)
        np.testing.assert_almost_equal(output_entropy_float_np_broadcast, gt_entropy_float_np_broadcast, decimal)
        np.testing.assert_almost_equal(output_entropy_np, gt_entropy_np, decimal)
        np.testing.assert_almost_equal(output_entropy_variable, gt_entropy_variable, decimal)
        np.testing.assert_almost_equal(output_lp_float_np_broadcast, gt_lp_float_np_broadcast, decimal)
        np.testing.assert_almost_equal(output_lp_np, gt_lp_np, decimal)
        np.testing.assert_almost_equal(output_lp_variable, gt_lp_variable, decimal)
        np.testing.assert_almost_equal(output_kl_float, gt_kl_float, decimal)
        np.testing.assert_almost_equal(output_kl_float_np_broadcast, gt_kl_float_np_broadcast, decimal)
        np.testing.assert_almost_equal(output_kl_np, gt_kl_np, decimal)
        np.testing.assert_almost_equal(output_kl_variable, gt_kl_variable, decimal)

    def test_uniform_distribution(self, batch_size=1, dims=3, decimal=5):
        test_program = fluid.Program()
        low_np = np.random.randn(batch_size, dims).astype('float32')
        other_low_np = np.random.randn(batch_size,
                                       dims).astype('float32')

        low_float = np.random.uniform(-2, 1)
        high_float = np.random.uniform(1, 3)

        other_low_float = np.random.uniform(-2, 1)
        other_high_float = np.random.uniform(1, 3)

        high_np = np.random.uniform(-5.0, 5.0, (batch_size, dims)).astype('float32')
        other_high_np = np.random.uniform(5.0, 15.0, (batch_size,
                                                      dims)).astype('float32')

        values_np = np.random.randn(batch_size, dims).astype('float32')

        with fluid.program_guard(test_program):
            low = layers.data(
                name='low', shape=[dims], dtype='float32')
            high = layers.data(
                name='high', shape=[dims], dtype='float32')

            other_low = layers.data(
                name='other_low', shape=[dims], dtype='float32')
            other_high = layers.data(
                name='other_high', shape=[dims], dtype='float32')

            values = layers.data(name='values', shape=[dims], dtype='float32')

            uniform_float = Uniform(low_float, high_float)
            #other_uniform_float = Uniform(other_low_float, other_high_float)

            uniform_float_np_broadcast = Uniform(low_float, high_np)
            #other_uniform_float_np_broadcast = Uniform(other_low_float, other_high_np)

            uniform_np = Uniform(low_np, high_np)
            #other_uniform_np = Uniform(other_low_np, other_high_np)

            uniform_variable = Uniform(low, high)
            #other_uniform_variable = Uniform(other_low, other_high)

            sample_float = uniform_float.sample([batch_size, dims])
            sample_float_np_broadcast = uniform_float_np_broadcast.sample([batch_size, dims])
            sample_np = uniform_np.sample([batch_size, dims])
            sample_variable = uniform_variable.sample([batch_size, dims])

            entropy_float = uniform_float.entropy()
            entropy_float_np_broadcast = uniform_float_np_broadcast.entropy()
            entropy_np = uniform_np.entropy()
            entropy_variable = uniform_variable.entropy()

            # lp_float = uniform_float.log_prob(values)
            lp_float_np_broadcast = uniform_float_np_broadcast.log_prob(values)
            lp_np = uniform_np.log_prob(values)
            lp_variable = uniform_variable.log_prob(values)

        self.executor.run(fluid.default_startup_program())
        if self.use_gpu:
            uniform_float_torch = td.Uniform(torch.FloatTensor(np.array(low_float)).cuda(),
                                             torch.FloatTensor(np.array(high_float)).cuda())
            other_uniform_float_torch = td.Uniform(torch.FloatTensor(np.array(other_low_float)).cuda(),
                                                   torch.FloatTensor(np.array(other_high_float)).cuda())

            uniform_float_np_broadcast_torch = td.Uniform(torch.FloatTensor(np.array(low_float)).cuda(),
                                                          torch.FloatTensor(np.array(high_np)).cuda())
            other_uniform_float_np_broadcast_torch = td.Uniform(torch.FloatTensor(np.array(other_low_float)).cuda(),
                                                                torch.FloatTensor(np.array(other_high_np)).cuda())

            uniform_np_torch = td.Uniform(torch.FloatTensor(np.array(low_np)).cuda(),
                                          torch.FloatTensor(np.array(high_np)).cuda())
            other_uniform_np_torch = td.Uniform(torch.FloatTensor(np.array(other_low_np)).cuda(),
                                                torch.FloatTensor(np.array(other_high_np)).cuda())

            gt_lp_float_np_broadcast = uniform_float_np_broadcast_torch.log_prob(
                torch.FloatTensor(np.array(values_np)).cuda()).cpu().numpy()
            gt_lp_np = gt_lp_variable = uniform_np_torch.log_prob(
                torch.FloatTensor(np.array(values_np)).cuda()).cpu().numpy()
        else:
            uniform_float_torch = td.Uniform(torch.FloatTensor(np.array(low_float)),
                                             torch.FloatTensor(np.array(high_float)))
            other_uniform_float_torch = td.Uniform(torch.FloatTensor(np.array(other_low_float)),
                                                   torch.FloatTensor(np.array(other_high_float)))

            uniform_float_np_broadcast_torch = td.Uniform(torch.FloatTensor(np.array(low_float)),
                                                          torch.FloatTensor(np.array(high_np)))
            other_uniform_float_np_broadcast_torch = td.Uniform(torch.FloatTensor(np.array(other_low_float)),
                                                                torch.FloatTensor(np.array(other_high_np)))

            uniform_np_torch = td.Uniform(torch.FloatTensor(np.array(low_np)), torch.FloatTensor(np.array(high_np)))
            other_uniform_np_torch = td.Uniform(torch.FloatTensor(np.array(other_low_np)),
                                                torch.FloatTensor(np.array(other_high_np)))

            gt_lp_float_np_broadcast = uniform_float_np_broadcast_torch.log_prob(
                torch.FloatTensor(np.array(values_np))).cpu().numpy()
            gt_lp_np = gt_lp_variable = uniform_np_torch.log_prob(torch.FloatTensor(np.array(values_np))).cpu().numpy()

        gt_sample_float = uniform_float_torch.sample([batch_size, dims]).cpu().numpy()
        gt_sample_float_np_broadcast = uniform_float_np_broadcast_torch.sample([batch_size, dims]).cpu().numpy()
        gt_sample_np = gt_sample_variable = uniform_np_torch.sample([batch_size, dims]).cpu().numpy()

        gt_entropy_float = uniform_float_torch.entropy().cpu().numpy()
        gt_entropy_float_np_broadcast = uniform_float_np_broadcast_torch.entropy().cpu().numpy()
        gt_entropy_np = gt_entropy_variable = uniform_np_torch.entropy().cpu().numpy()

        # result calculated by CategoricalDistribution
        [output_sample_float,
         output_sample_float_np_broadcast,
         output_sample_np,
         output_sample_variable,
         output_entropy_float,
         output_entropy_float_np_broadcast,
         output_entropy_np,
         output_entropy_variable,
         output_lp_float_np_broadcast,
         output_lp_np,
         output_lp_variable
         ] = self.executor.run(
            program=test_program,
            feed={
                'low': low_np,
                'high': high_np,
                'other_low': other_low_np,
                'other_high': other_high_np,
                'values': values_np
            },
            fetch_list=[
                sample_float,
                sample_float_np_broadcast,
                sample_np,
                sample_variable,
                entropy_float,
                entropy_float_np_broadcast,
                entropy_np,
                entropy_variable,
                lp_float_np_broadcast,
                lp_np,
                lp_variable
            ])

        # test entropy
        np.testing.assert_almost_equal(output_sample_float.shape, gt_sample_float.shape, decimal)
        np.testing.assert_almost_equal(output_entropy_float, gt_entropy_float, decimal)
        np.testing.assert_almost_equal(output_entropy_float_np_broadcast, gt_entropy_float_np_broadcast, decimal)
        np.testing.assert_almost_equal(output_entropy_np, gt_entropy_np, decimal)
        np.testing.assert_almost_equal(output_entropy_variable, gt_entropy_variable, decimal)
        np.testing.assert_almost_equal(output_lp_float_np_broadcast, gt_lp_float_np_broadcast, decimal)
        np.testing.assert_almost_equal(output_lp_np, gt_lp_np, decimal)
        np.testing.assert_almost_equal(output_lp_variable, gt_lp_variable, decimal)


if __name__ == '__main__':
    unittest.main()