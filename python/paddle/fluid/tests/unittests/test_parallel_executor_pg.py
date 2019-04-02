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
import os
os.environ['FLAGS_enable_parallel_graph'] = str(1)
import paddle.fluid.core as core
import os
import paddle.fluid as fluid
from parallel_executor_test_base import TestParallelExecutorBase


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


class TestMNIST(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def _init_data(self):
        np.random.seed(5)
        img = np.random.random(size=[32, 784]).astype(np.float32)
        label = np.ones(shape=[32, 1], dtype='int64')
        return img, label

    # simple_fc
    def check_simple_fc_convergence(self, use_cuda, use_reduce=False):
        if use_cuda and not core.is_compiled_with_cuda():
            return

        img, label = self._init_data()

        self.check_network_convergence(
            simple_fc_net,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            use_reduce=use_reduce)

    def test_simple_fc(self):
        # use_cuda
        self.check_simple_fc_convergence(True)

    def check_simple_fc_parallel_accuracy(self, use_cuda):
        if use_cuda and not core.is_compiled_with_cuda():
            return

        img, label = self._init_data()

        single_first_loss, single_last_loss = self.check_network_convergence(
            method=simple_fc_net,
            seed=1,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            use_parallel_executor=False)
        parallel_first_loss, parallel_last_loss = self.check_network_convergence(
            method=simple_fc_net,
            seed=1,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            use_parallel_executor=True)

        self.assertAlmostEquals(
            np.mean(parallel_first_loss),
            single_first_loss,
            delta=1e-6, )
        self.assertAlmostEquals(
            np.mean(parallel_last_loss), single_last_loss, delta=1e-6)

    def test_simple_fc_parallel_accuracy(self):
        self.check_simple_fc_parallel_accuracy(True)


if __name__ == '__main__':
    unittest.main()
