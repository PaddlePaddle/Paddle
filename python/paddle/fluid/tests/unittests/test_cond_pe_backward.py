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

from __future__ import print_function

import numpy as np
import os
import unittest

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Program, program_guard
from simple_nets import simple_fc_net_with_inputs, batchnorm_fc_with_inputs

np.random.seed(123)


class TestCondBackward(unittest.TestCase):
    def backward_value_helper(self, cond_func, use_cuda, use_parallel_exe):
        """
        Helper function that compares calculated backward value is close to dy/dx
        """
        main_program = Program()
        main_program.random_seed = 123
        startup_program = Program()
        startup_program.random_seed = 123
        with program_guard(main_program, startup_program):
            img = fluid.data(name='image', shape=[-1, 9], dtype='float32')
            img.stop_gradient = False
            label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
            i = fluid.data(name="i", shape=[1], dtype='int32')
            loss = cond_func(i, img, label)
            append_backward(loss)
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        num_devices = 1
        if use_parallel_exe:
            os.environ['CPU_NUM'] = str(2)
            exe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                main_program=main_program,
                loss_name=loss.name)
            num_devices = exe.device_count

        delta = 0.005
        for feed_i in range(0, 10):
            feed_img = np.random.random(size=[1, 9]).astype(np.float32)
            feed_label = np.random.randint(
                low=0, high=10, size=[1, 1], dtype=np.int64)
            if use_parallel_exe:
                img_grad, loss_value = exe.run(
                    feed={
                        'i': np.full((num_devices), feed_i, np.int32),
                        'image': np.repeat(
                            feed_img, num_devices, axis=0),
                        'label': np.repeat(
                            feed_label, num_devices, axis=0)
                    },
                    fetch_list=[img.grad_name, loss.name])
            else:
                img_grad, loss_value = exe.run(
                    main_program,
                    feed={
                        'i': np.full((1), feed_i, np.int32),
                        'image': feed_img,
                        'label': feed_label
                    },
                    fetch_list=[img.grad_name, loss.name])

            numerical_grad = np.zeros(shape=[num_devices, 9], dtype=np.float32)
            feed_img_delta = np.copy(feed_img)
            for j in range(9):
                feed_img_delta[0][j] = feed_img[0][j] + delta
                if use_parallel_exe:
                    loss_delta = exe.run(feed={
                        'i': np.full((num_devices), feed_i, np.int32),
                        'image': np.repeat(
                            feed_img_delta, num_devices, axis=0),
                        'label': np.repeat(
                            feed_label, num_devices, axis=0)
                    },
                                         fetch_list=[loss.name])
                    multi_device_grad = (
                        loss_delta[0] - loss_value[0]) / delta / num_devices
                    for d in range(num_devices):
                        numerical_grad[d][j] = multi_device_grad[d]
                else:
                    loss_delta = exe.run(main_program,
                                         feed={
                                             'i': np.full((1), feed_i,
                                                          np.int32),
                                             'image': feed_img_delta,
                                             'label': feed_label
                                         },
                                         fetch_list=[loss.name])
                    numerical_grad[0][j] = (
                        loss_delta[0] - loss_value[0]) / delta
                feed_img_delta[0][j] = feed_img[0][j]
            self.assertTrue(
                np.isclose(
                    img_grad, numerical_grad, atol=0.05, rtol=0.05).all())

    def add_optimizer_helper(self, cond_func, use_cuda, use_parallel_exe):
        """
        Test that program is runnable when add optimizer
        """
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            img = fluid.data(name='image', shape=[-1, 784], dtype='float32')
            label = fluid.data(name='label', shape=[-1, 1], dtype='int64')
            i = fluid.data(name="i", shape=[1], dtype='int32')
            loss = cond_func(i, img, label)
            optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss)

        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        if use_parallel_exe:
            os.environ['CPU_NUM'] = str(2)
            exe = fluid.ParallelExecutor(
                use_cuda=use_cuda,
                main_program=main_program,
                loss_name=loss.name)
            num_devices = exe.device_count

        for feed_i in range(0, 10):
            feed_img = np.random.random(size=[16, 784]).astype(np.float32)
            feed_label = np.random.randint(
                low=0, high=10, size=[16, 1], dtype=np.int64)
            if use_parallel_exe:
                exe.run(feed={
                    'i': np.full((num_devices), feed_i, np.int32),
                    'image': np.repeat(
                        feed_img, num_devices, axis=0),
                    'label': np.repeat(
                        feed_label, num_devices, axis=0)
                },
                        fetch_list=[loss.name])
            else:
                exe.run(main_program,
                        feed={
                            'i': np.full((1), feed_i, np.int32),
                            'image': feed_img,
                            'label': feed_label
                        },
                        fetch_list=[loss])

    def test_cond_backward(self):
        def cond_func(i, img, label):
            predicate = ((i % 2) == 0)
            return layers.cond(predicate,
                               lambda: simple_fc_net_with_inputs(img, label, class_num=10),
                               lambda: batchnorm_fc_with_inputs(img, label, class_num=10))

        for use_parallel_exe in [False, True]:
            self.backward_value_helper(cond_func,
                                       core.is_compiled_with_cuda(),
                                       use_parallel_exe)
            self.add_optimizer_helper(cond_func,
                                      core.is_compiled_with_cuda(),
                                      use_parallel_exe)


if __name__ == '__main__':
    unittest.main()
