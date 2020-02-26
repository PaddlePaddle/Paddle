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
from time import time
import numpy as np
from collections import Counter
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.base import to_variable

from dygraph_to_static.program import FuncProgram

SEED = 2020

from mnist import MNIST

func_program = FuncProgram('func_program')


def dygraph_to_static_output(dygraph_func):
    def __impl__(*args, **kwargs):
        static_func = func_program.add_layers(dygraph_func)

        return func_program(*args, **kwargs)

    return __impl__


def test_mnist(reader, model, batch_size):
    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array([x[0].reshape(1, 28, 28)
                              for x in data]).astype('float32')
        y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape(batch_size, 1)

        img = to_variable(dy_x_data)
        label = to_variable(y_data)
        label.stop_gradient = True
        prediction, acc, avg_loss = model(img, label)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    return avg_loss_val_mean, acc_val_mean


def train_mnist_in_dygraph_mode():
    """
    Tests model if doesn't change the layers while decorated
    by `dygraph_to_static_output`. In this case, everything should
    still works if model is trained in dygraph mode.
    """
    epoch_num = 1
    BATCH_SIZE = 64
    use_data_parallel = False

    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):

        if use_data_parallel:
            strategy = fluid.dygraph.parallel.prepare_context()
        mnist = MNIST()
        adam = AdamOptimizer(
            learning_rate=0.001, parameter_list=mnist.parameters())
        if use_data_parallel:
            mnist = fluid.dygraph.parallel.DataParallel(mnist, strategy)

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
        if use_data_parallel:
            train_reader = fluid.contrib.reader.distributed_batch_reader(
                train_reader)

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)
        start = time()
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                      for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = to_variable(dy_x_data)
                label = to_variable(y_data)

                label.stop_gradient = True

                cost, acc, avg_loss = mnist(img, label)

                if use_data_parallel:
                    avg_loss = mnist.scale_loss(avg_loss)
                    avg_loss.backward()
                    mnist.apply_collective_grads()
                else:
                    avg_loss.backward()

                adam.minimize(avg_loss)
                # save checkpoint
                mnist.clear_gradients()
                if batch_id % 100 == 0:
                    print(
                        "Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}"
                        .format(epoch, batch_id,
                                avg_loss.numpy(), acc.numpy(), time() - start))
                    if batch_id == 300:
                        break


def train_mnist_in_static_mode():
    """
    Tests model when using `dygraph_to_static_output` to convert dygraph into static
    model. It allows user to add customized code to train static model, such as `with`
    and `Executor` statement.
    """
    epoch_num = 1
    BATCH_SIZE = 64
    use_data_parallel = False

    prev_ops, cur_ops = Counter(), Counter()

    place = fluid.CPUPlace()
    main_prog = fluid.default_main_program()
    with fluid.program_guard(main_prog):
        adam_flag = True

        mnist = MNIST()
        adam = AdamOptimizer(
            learning_rate=0.001, parameter_list=mnist.parameters())
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)

        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

        exe = fluid.Executor(place)
        start = time()

        img = fluid.data(name='img', shape=[None, 1, 28, 28], dtype='float32')
        label = fluid.data(name='label', shape=[None, 1], dtype='int64')
        label.stop_gradient = True

        cost, acc, avg_loss = mnist(img, label)
        adam.minimize(avg_loss)
        exe.run(fluid.default_startup_program())

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                      for x in data]).astype('float32')
                y_data = np.array(
                    [x[1] for x in data]).astype('int64').reshape(-1, 1)

                prev_ops = cur_ops
                cur_ops = Counter([
                    op.type for op in fluid.default_main_program().block(0).ops
                ])
                # print(len(fluid.default_main_program().block(0).ops))
                # print(cur_ops - prev_ops)

                out = exe.run(fetch_list=[avg_loss, acc],
                              feed={'img': dy_x_data,
                                    'label': y_data})
                # save checkpoint
                if batch_id % 100 == 0:
                    print(
                        "Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}"
                        .format(epoch, batch_id,
                                np.array(out[0]),
                                np.array(out[1]), time() - start))
                    if batch_id == 300:
                        break


def train_mnist_with_cache_program():
    """
    Tests model with no code related to static model training, like `with`,
    `Executor` or `feed/fetch` statement. It allows user to get outputs easily
    and train model with less code.
    """
    epoch_num = 1
    BATCH_SIZE = 64

    prev_ops, cur_ops = Counter(), Counter()

    mnist = MNIST()
    adam = AdamOptimizer(learning_rate=0.001)
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)

    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

    start = time()

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_reader()):
            dy_x_data = np.array([x[0].reshape(1, 28, 28)
                                  for x in data]).astype('float32')
            y_data = np.array([x[1]
                               for x in data]).astype('int64').reshape(-1, 1)

            prediction, acc, avg_loss = mnist(dy_x_data, y_data)

            prev_ops = cur_ops
            cur_ops = Counter(
                [op.type for op in fluid.default_main_program().block(0).ops])
            # print(len(fluid.default_main_program().block(0).ops))
            # print(cur_ops - prev_ops)

            # save checkpoint
            if batch_id % 100 == 0:
                print("Loss at epoch {} step {}: loss: {:}, acc: {}, cost: {}"
                      .format(epoch, batch_id,
                              np.array(avg_loss[0]),
                              np.array(acc[0]), time() - start))
                if batch_id == 300:
                    break


if __name__ == "__main__":
    # train_mnist_in_dygraph_mode()  # 11.41 s / 300 batch
    # train_mnist_in_static_mode()  # 11.21 s / 300 batch
    train_mnist_with_cache_program()  # 11.84s / 300 batch
