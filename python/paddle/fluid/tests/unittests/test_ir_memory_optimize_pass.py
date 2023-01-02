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

import unittest

import numpy as np
from parallel_executor_test_base import DeviceType, TestParallelExecutorBase

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core


def _feed_data_helper():
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    return img, label


def simple_fc_net(use_feed):
    assert use_feed
    x, y = _feed_data_helper()
    hidden_layer = 4
    for _ in range(hidden_layer):
        x = fluid.layers.fc(input=x, size=20, act='relu')
    y_predict = fluid.layers.fc(input=x, size=10, act='softmax')
    cost = paddle.nn.functional.cross_entropy(
        input=y_predict, label=y, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(cost)
    return avg_cost


def fc_with_inplace_net(use_feed):
    assert use_feed
    x, y = _feed_data_helper()
    fc = fluid.layers.fc(input=x, size=20, act='relu')
    fc = fluid.layers.fc(input=fc, size=10, act='relu')
    reshape = paddle.reshape(x=fc, shape=[-1, 2, 5])
    reshape = paddle.reshape(x=reshape, shape=[-1, 5, 2])
    y_predict = fluid.layers.fc(input=reshape, size=10, act='softmax')
    cost = paddle.nn.functional.cross_entropy(
        input=y_predict, label=y, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(cost)
    return avg_cost


class TestMNIST(TestParallelExecutorBase):
    def _dummy_data(self):
        np.random.seed(5)
        img = np.random.random(size=[32, 784]).astype(np.float32)
        label = np.ones(shape=[32, 1], dtype='int64')
        return img, label

    def _compare_ir_memory_optimize(self, model, use_device):
        if use_device == DeviceType.CUDA and not core.is_compiled_with_cuda():
            return

        img, label = self._dummy_data()
        first_loss0, last_loss0, _ = self.check_network_convergence(
            model,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_ir_memory_optimize=False,
        )
        first_loss1, last_loss1, _ = self.check_network_convergence(
            model,
            feed_dict={"image": img, "label": label},
            use_device=use_device,
            use_ir_memory_optimize=True,
        )
        for loss in zip(first_loss0, first_loss1):
            self.assertAlmostEqual(loss[0], loss[1], delta=1e-6)
        for loss in zip(last_loss0, last_loss1):
            self.assertAlmostEqual(loss[0], loss[1], delta=1e-6)

    def test_simple_fc_net(self):
        self._compare_ir_memory_optimize(simple_fc_net, DeviceType.CPU)
        self._compare_ir_memory_optimize(simple_fc_net, DeviceType.CUDA)

    def test_fc_with_reshape_net(self):
        self._compare_ir_memory_optimize(fc_with_inplace_net, DeviceType.CPU)
        self._compare_ir_memory_optimize(fc_with_inplace_net, DeviceType.CUDA)


if __name__ == '__main__':
    unittest.main()
