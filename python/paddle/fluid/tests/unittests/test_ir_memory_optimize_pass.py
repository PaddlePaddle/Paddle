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

from parallel_executor_test_base import TestParallelExecutorBase
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np
import paddle
import paddle.dataset.mnist as mnist
import unittest
import os

MNIST_RECORDIO_FILE = "./mnist_test_pe.recordio"


def _feed_data_helper(use_feed):
    if use_feed:
        img = fluid.layers.data(name='image', shape=[784], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    else:
        reader = fluid.layers.open_files(
            filenames=[MNIST_RECORDIO_FILE],
            shapes=[[-1, 784], [-1, 1]],
            lod_levels=[0, 0],
            dtypes=['float32', 'int64'])
        reader = fluid.layers.io.double_buffer(reader)
        img, label = fluid.layers.read_file(reader)
    return img, label


def simple_fc_net(use_feed):
    x, y = _feed_data_helper(use_feed)
    hidden_layer = 4
    for _ in range(hidden_layer):
        x = fluid.layers.fc(input=x, size=20, act='relu')
    y_predict = fluid.layers.fc(input=x, size=10, act='softmax')
    cost = fluid.layers.cross_entropy(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


def fc_with_inplace_net(use_feed):
    x, y = _feed_data_helper(use_feed)
    fc = fluid.layers.fc(input=x, size=20, act='relu')
    fc = fluid.layers.fc(input=fc, size=10, act='relu')
    reshape = fluid.layers.reshape(x=fc, shape=[-1, 2, 5])
    reshape = fluid.layers.reshape(x=reshape, shape=[-1, 5, 2])
    y_predict = fluid.layers.fc(input=reshape, size=10, act='softmax')
    cost = fluid.layers.cross_entropy(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)
    return avg_cost


class TestMNIST(TestParallelExecutorBase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)
        # Convert mnist to recordio file
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            reader = paddle.batch(mnist.train(), batch_size=4)
            feeder = fluid.DataFeeder(
                feed_list=[  # order is image and label
                    fluid.layers.data(
                        name='image', shape=[784]),
                    fluid.layers.data(
                        name='label', shape=[1], dtype='int64'),
                ],
                place=fluid.CPUPlace())
            fluid.recordio_writer.convert_reader_to_recordio_file(
                MNIST_RECORDIO_FILE, reader, feeder)

    def _dummy_data(self):
        np.random.seed(5)
        img = np.random.random(size=[32, 784]).astype(np.float32)
        label = np.ones(shape=[32, 1], dtype='int64')
        return img, label

    def _compare_ir_and_python_memory_optimize(self, model, use_cuda):
        if use_cuda and not core.is_compiled_with_cuda():
            return

        img, label = self._dummy_data()
        first_loss0, last_loss0 = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            memory_opt=False,
            use_ir_memory_optimize=False)
        first_loss1, last_loss1 = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            memory_opt=False,
            use_ir_memory_optimize=True)
        for loss in zip(first_loss0, first_loss1):
            self.assertAlmostEqual(loss[0], loss[1], delta=1e-6)
        for loss in zip(last_loss0, last_loss1):
            self.assertAlmostEqual(loss[0], loss[1], delta=1e-6)

    def test_simple_fc_net(self):
        self._compare_ir_and_python_memory_optimize(simple_fc_net, False)
        self._compare_ir_and_python_memory_optimize(simple_fc_net, True)

    def test_fc_with_reshape_net(self):
        self._compare_ir_and_python_memory_optimize(fc_with_inplace_net, False)
        self._compare_ir_and_python_memory_optimize(fc_with_inplace_net, True)


if __name__ == '__main__':
    unittest.main()
