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
import numpy as np
import paddle
import paddle.dataset.mnist as mnist
import unittest
import os

MNIST_RECORDIO_FILE = "./mnist_test_pe.recordio"


def simple_fc_net(use_feed):
    if use_feed:
        img = fluid.layers.data(name='image', shape=[784], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    else:
        reader = fluid.layers.open_files(
            filenames=[MNIST_RECORDIO_FILE],
            shapes=[[-1, 784], [-1, 1]],
            lod_levels=[0, 0],
            dtypes=['float32', 'int64'],
            thread_num=1,
            for_parallel=True)
        reader = fluid.layers.io.double_buffer(reader)
        img, label = fluid.layers.read_file(reader)
    hidden = img
    for _ in xrange(4):
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
    if use_feed:
        img = fluid.layers.data(name='image', shape=[784], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    else:
        reader = fluid.layers.open_files(
            filenames=[MNIST_RECORDIO_FILE],
            shapes=[[-1, 784], [-1, 1]],
            lod_levels=[0, 0],
            dtypes=['float32', 'int64'],
            thread_num=1,
            for_parallel=True)
        reader = fluid.layers.io.double_buffer(reader)
        img, label = fluid.layers.read_file(reader)

    hidden = img
    for _ in xrange(1):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))

        hidden = fluid.layers.batch_norm(input=hidden)

    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


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

    def check_simple_fc_convergence(self,
                                    balance_parameter_opt_between_cards,
                                    use_cuda=True):
        self.check_network_convergence(simple_fc_net, use_cuda=use_cuda)
        self.check_network_convergence(
            simple_fc_net, use_cuda=use_cuda, allow_op_delay=True)

        img = np.zeros(shape=[32, 784], dtype='float32')
        label = np.ones(shape=[32, 1], dtype='int64')
        self.check_network_convergence(
            simple_fc_net,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
        )

    def test_simple_fc(self):
        self.check_simple_fc_convergence(False, use_cuda=True)
        self.check_simple_fc_convergence(False, use_cuda=False)

    def test_simple_fc_with_new_strategy(self):
        self.check_simple_fc_convergence(True, use_cuda=True)
        self.check_simple_fc_convergence(True, use_cuda=False)

    def check_simple_fc_parallel_accuracy(self,
                                          balance_parameter_opt_between_cards,
                                          use_cuda=True):
        img = np.zeros(shape=[32, 784], dtype='float32')
        label = np.ones(shape=[32, 1], dtype='int64')
        single_first_loss, single_last_loss = self.check_network_convergence(
            method=simple_fc_net,
            seed=1000,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            use_parallel_executor=False)
        parallel_first_loss, parallel_last_loss = self.check_network_convergence(
            method=simple_fc_net,
            seed=1000,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            use_parallel_executor=True,
            balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
        )

        for p_f in parallel_first_loss:
            self.assertAlmostEquals(p_f, single_first_loss[0], delta=1e-6)
        for p_l in parallel_last_loss:
            self.assertAlmostEquals(p_l, single_last_loss[0], delta=1e-6)

    def test_simple_fc_parallel_accuracy(self):
        self.check_simple_fc_parallel_accuracy(False, use_cuda=True)
        self.check_simple_fc_parallel_accuracy(False, use_cuda=False)

    def test_simple_fc_parallel_accuracy_with_new_strategy(self):
        self.check_simple_fc_parallel_accuracy(True, use_cuda=True)
        self.check_simple_fc_parallel_accuracy(True, use_cuda=False)

    def check_batchnorm_fc_convergence(
            self, balance_parameter_opt_between_cards, use_cuda):
        self.check_network_convergence(fc_with_batchnorm, use_cuda=use_cuda)
        img = np.zeros(shape=[32, 784], dtype='float32')
        label = np.ones(shape=[32, 1], dtype='int64')
        self.check_network_convergence(
            fc_with_batchnorm,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            balance_parameter_opt_between_cards=balance_parameter_opt_between_cards
        )

    def test_batchnorm_fc(self):
        self.check_batchnorm_fc_convergence(False, use_cuda=True)
        self.check_batchnorm_fc_convergence(False, use_cuda=False)

    def test_batchnorm_fc_with_new_strategy(self):
        self.check_batchnorm_fc_convergence(True, use_cuda=True)
        self.check_batchnorm_fc_convergence(True, use_cuda=False)


if __name__ == '__main__':
    unittest.main()
