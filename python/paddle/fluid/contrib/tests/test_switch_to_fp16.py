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

from __future__ import print_function

import paddle.fluid.core as core
import math
import os
import sys
import unittest

import numpy as np

import paddle
import paddle.fluid as fluid


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def mlp(img, label):
    hidden = fluid.layers.fc(input=img, size=200, act='tanh')
    hidden = fluid.layers.fc(input=hidden, size=200, act='tanh')
    return hidden, label


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    return conv_pool_2, label


def train(nn_type):

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net_conf = mlp if nn_type == 'mlp' else conv_net

    with fluid.contrib.switch_to_fp16(fluid.default_main_program()):
        hidden, label = net_conf(img, label)

    hidden = fluid.layers.cast(hidden, np.float32)
    prediction, avg_loss, acc = loss_net(hidden, label)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=64)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())
        for batch_id, data in enumerate(train_reader()):
            acc_np, avg_loss_np = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=[acc, avg_loss])
            print("acc: {}, avg_loss: {}".format(
                float(acc_np), float(avg_loss_np)))

            return

    train_loop(fluid.default_main_program())


def main(nn_type):
    train(nn_type=nn_type)


class TestRecognizeDigits(unittest.TestCase):
    pass


def inject_test_method(nn_type):
    def __impl__(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                main(nn_type)

    fn = 'test_{0}'.format(nn_type)

    setattr(TestRecognizeDigits, fn, __impl__)


def inject_all_tests():
    if not core.is_compiled_with_cuda():
        return
    for nn_type in ('mlp', 'conv'):
        inject_test_method(nn_type)


inject_all_tests()

if __name__ == '__main__':
    unittest.main()
