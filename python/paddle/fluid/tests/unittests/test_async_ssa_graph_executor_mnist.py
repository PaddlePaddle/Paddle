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

import os
import unittest

import numpy
import paddle
import paddle.fluid as fluid

BATCH_SIZE = 64


def loss_net(hidden, label):
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc


def convolutional_neural_network(img, label):
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
    return loss_net(conv_pool_2, label)


def train(use_cuda, thread_num, cpu_num):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        print("paddle is not compiled with cuda, exit!")
        return

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    prediction, avg_loss, acc = convolutional_neural_network(img, label)

    test_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    def train_test(train_test_program, train_test_feed, train_test_reader):
        acc_set = []
        avg_loss_set = []
        for test_data in train_test_reader():
            acc_np, avg_loss_np = exe.run(program=train_test_program,
                                          feed=train_test_feed.feed(test_data),
                                          fetch_list=[acc, avg_loss])
            acc_set.append(float(acc_np))
            avg_loss_set.append(float(avg_loss_np))
        # get test acc and loss
        acc_val_mean = numpy.array(acc_set).mean()
        avg_loss_val_mean = numpy.array(avg_loss_set).mean()
        return avg_loss_val_mean, acc_val_mean

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.mnist.train(), buf_size=500),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    os.environ['CPU_NUM'] = str(cpu_num)

    print("cpu_num:" + str(cpu_num))
    print("thread_num:" + str(thread_num))

    build_strategy = fluid.BuildStrategy()
    build_strategy.async_mode = True  # enable async mode

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = thread_num
    exec_strategy.num_iteration_per_run = 2

    main_program = fluid.default_main_program()
    pe = fluid.ParallelExecutor(
        use_cuda=False,
        loss_name=avg_loss.name,
        main_program=main_program,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    step = 0
    for step_id, data in enumerate(train_reader()):
        loss_val = pe.run(feed=feeder.feed(data), fetch_list=[avg_loss.name])
        loss_val = numpy.mean(loss_val)
        if step % 100 == 0:
            print("Batch %d, Cost %f" % (step, loss_val))
        step += 1
    # test for epoch
    avg_loss_val, acc_val = train_test(
        train_test_program=test_program,
        train_test_reader=test_reader,
        train_test_feed=feeder)

    print("Test: avg_cost: %s, acc: %s" % (avg_loss_val, acc_val))


class TestAsyncSSAGraphExecutor(unittest.TestCase):
    def test_check_async_ssa_exe_train(self):
        train(use_cuda=False, thread_num=2, cpu_num=2)


if __name__ == "__main__":
    unittest.main()
