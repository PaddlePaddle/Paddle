#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from simple_nets import init_data

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

batch_size = 12
img_shape = [1, 28, 28]


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)
    return avg_loss


def conv_net(use_feed):
    img = fluid.layers.data(name='image', shape=img_shape, dtype='float16')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    conv_pool_1 = paddle.static.nn.batch_norm(conv_pool_1)

    conv_pool_1 = fluid.layers.cast(conv_pool_1, np.float32)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu",
    )
    hidden = fluid.layers.cast(conv_pool_2, np.float32)
    return loss_net(hidden, label)


def _optimizer(learning_rate=1e-6):
    optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
    return optimizer


class TestResnet(TestParallelExecutorBase):
    def check_model(self, use_device):
        img, label = init_data(
            batch_size=batch_size, img_shape=img_shape, label_range=9
        )
        img = np.float16(img)
        feed_dict = {"image": img, "label": label}

        TestParallelExecutorBase.check_network_convergence(
            conv_net,
            feed_dict=feed_dict,
            iter=10,
            use_device=use_device,
            fuse_all_reduce_ops=True,
            optimizer=_optimizer,
        )

    def test_model(self):
        if core.is_compiled_with_cuda():
            self.check_model(DeviceType.CUDA)


if __name__ == '__main__':
    unittest.main()
