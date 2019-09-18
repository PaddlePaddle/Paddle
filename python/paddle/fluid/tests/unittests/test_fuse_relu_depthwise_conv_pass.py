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


def norm(*args, **kargs):
    return fluid.layers.batch_norm(*args, **kargs)


def sep_conv(input, channel, stride, filter, dilation=1, act=None):
    # with scope('depthwise'):
    input = fluid.layers.conv2d(
        input,
        input.shape[1],
        filter,
        stride,
        groups=input.shape[1],
        padding=(filter // 2) * dilation,
        dilation=dilation,
        use_cudnn=False,
        bias_attr=False)
    input = norm(input)
    if act: input = act(input)
    # with scope('pointwise'):
    input = fluid.layers.conv2d(
        input, channel, 1, 1, groups=1, padding=0, bias_attr=False)
    input = norm(input)
    if act: input = act(input)
    return input


def simple_depthwise_net(use_feed):
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
    hidden = fluid.layers.reshape(img, (-1, 1, 28, 28))
    for _ in range(4):
        hidden = sep_conv(hidden, channel=200, stride=2, filter=5)
        hidden = fluid.layers.relu(hidden)
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

    def _init_data(self, random=True):
        np.random.seed(5)
        if random:
            img = np.random.random(size=[32, 784]).astype(np.float32)
        else:
            img = np.ones(shape=[32, 784], dtype='float32')
        label = np.ones(shape=[32, 1], dtype='int64')
        return img, label

    def _compare(self, model, use_cuda, random_data=True, only_forward=False):
        if use_cuda and not core.is_compiled_with_cuda():
            return
        img, label = self._init_data(random_data)

        def _optimizer(learning_rate=1e-6):
            optimizer = fluid.optimizer.SGD(
                learning_rate=learning_rate,
                regularization=fluid.regularizer.L2Decay(1e-6))
            return optimizer

        if only_forward:
            _optimizer = None

        fuse_op_first_loss, fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            fuse_relu_depthwise_conv=True,
            use_ir_memory_optimize=True,
            memory_opt=False,
            optimizer=_optimizer)
        not_fuse_op_first_loss, not_fuse_op_last_loss = self.check_network_convergence(
            model,
            feed_dict={"image": img,
                       "label": label},
            use_cuda=use_cuda,
            fuse_relu_depthwise_conv=False,
            memory_opt=False,
            optimizer=_optimizer)

        for loss in zip(not_fuse_op_first_loss, fuse_op_first_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)
        for loss in zip(not_fuse_op_last_loss, fuse_op_last_loss):
            self.assertAlmostEquals(loss[0], loss[1], delta=1e-6)

    def test_simple_depthwise_with_fuse_op(self):
        self._compare(simple_depthwise_net, True)
        self._compare(simple_depthwise_net, False)

    def test_simple_depthwise_with_fuse_op_only_forward(self):
        self._compare(simple_depthwise_net, True, only_forward=True)
        self._compare(simple_depthwise_net, False, only_forward=True)


if __name__ == '__main__':
    unittest.main()
