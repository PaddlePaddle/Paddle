# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
import paddle.fluid.core as core
import paddle
import os
import paddle.fluid as fluid
from parallel_executor_test_base import TestParallelExecutorBase, DeviceType
from parallel_executor_test_base import DeviceType


def simple_fc_net(use_feed):
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    hidden = img
    for _ in range(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def fc_with_batchnorm(use_feed):
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    hidden = img
    for _ in range(1):
        with fluid.name_scope("hidden"):
            hidden = fluid.layers.fc(
                hidden,
                size=200,
                act='tanh',
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=1.0)))

            hidden = fluid.layers.batch_norm(input=hidden)
    with fluid.name_scope("fc_layer"):
        prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    with fluid.name_scope("loss"):
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        loss = fluid.layers.mean(loss)
    return loss


class TestMNIST(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def _init_data(self):
        np.random.seed(5)
        img = np.random.random(size=[32, 784]).astype(np.float32)
        label = np.ones(shape=[32, 1], dtype='int64')
        return img, label

    def _compare_reduce_and_allreduce(self,
                                      model,
                                      use_device,
                                      delta1=1e-6,
                                      delta2=1e-4):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        if use_device == DeviceType.XPU and not core.is_compiled_with_xpu():
            return

        img, label = self._init_data()

        all_reduce_first_loss, all_reduce_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            use_device=use_device,
            use_reduce=False)

        reduce_first_loss, reduce_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            use_device=use_device,
            use_reduce=True)

        for loss in zip(all_reduce_first_loss, reduce_first_loss):
            self.assertAlmostEqual(loss[0], loss[1], delta=delta1)
        for loss in zip(all_reduce_last_loss, reduce_last_loss):
            self.assertAlmostEqual(loss[0], loss[1], delta=delta2)

    # simple_fc
    def check_simple_fc_convergence(self, use_device, use_reduce=False):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        if use_device == DeviceType.XPU and not core.is_compiled_with_xpu():
            return

        img, label = self._init_data()

        self.check_network_convergence(
            simple_fc_net,
            feed_dict={"image": img,
                       "label": label},
            use_device=use_device,
            use_reduce=use_reduce)

    def test_simple_fc(self):
        # use_device
        self.check_simple_fc_convergence(DeviceType.CUDA)
        self.check_simple_fc_convergence(DeviceType.CPU)
        self.check_simple_fc_convergence(DeviceType.XPU)

    def test_simple_fc_with_new_strategy(self):
        # use_device, use_reduce
        # NOTE: the computation result of nccl_reduce is non-deterministic,
        # related issue: https://github.com/NVIDIA/nccl/issues/157
        self._compare_reduce_and_allreduce(simple_fc_net, DeviceType.CUDA, 1e-5,
                                           1e-2)
        self._compare_reduce_and_allreduce(simple_fc_net, DeviceType.CPU, 1e-5,
                                           1e-2)

    def check_simple_fc_parallel_accuracy(self, use_device):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        img, label = self._init_data()

        single_first_loss, single_last_loss = self.check_network_convergence(
            method=simple_fc_net,
            feed_dict={"image": img,
                       "label": label},
            use_device=use_device,
            use_parallel_executor=False)
        parallel_first_loss, parallel_last_loss = self.check_network_convergence(
            method=simple_fc_net,
            feed_dict={"image": img,
                       "label": label},
            use_device=use_device,
            use_parallel_executor=True)

        self.assertAlmostEquals(
            np.mean(parallel_first_loss),
            single_first_loss,
            delta=1e-6, )
        self.assertAlmostEquals(
            np.mean(parallel_last_loss), single_last_loss, delta=1e-6)

    def test_simple_fc_parallel_accuracy(self):
        self.check_simple_fc_parallel_accuracy(DeviceType.CUDA)
        self.check_simple_fc_parallel_accuracy(DeviceType.CPU)

    def check_batchnorm_fc_convergence(self, use_device, use_fast_executor):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return
        if use_device == DeviceType.XPU and not core.is_compiled_with_xpu():
            return
        img, label = self._init_data()

        self.check_network_convergence(
            fc_with_batchnorm,
            feed_dict={"image": img,
                       "label": label},
            use_device=use_device,
            use_fast_executor=use_fast_executor)

    def test_batchnorm_fc(self):
        for use_device in (DeviceType.CPU, DeviceType.CUDA):
            for use_fast_executor in (False, True):
                self.check_batchnorm_fc_convergence(use_device,
                                                    use_fast_executor)

    def test_batchnorm_fc_with_new_strategy(self):
        # NOTE: the computation result of nccl_reduce is non-deterministic,
        # related issue: https://github.com/NVIDIA/nccl/issues/157
        self._compare_reduce_and_allreduce(fc_with_batchnorm, DeviceType.CUDA,
                                           1e-5, 1e-2)
        self._compare_reduce_and_allreduce(fc_with_batchnorm, DeviceType.CPU,
                                           1e-5, 1e-2)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
