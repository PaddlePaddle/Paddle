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

import numpy
import six

import paddle
import paddle.dataset.mnist as mnist
import paddle.fluid as fluid


def network(is_train):
    reader = fluid.layers.py_reader(
        capacity=10,
        shapes=((-1, 784), (-1, 1)),
        dtypes=('float32', 'int64'),
        name="train_reader" if is_train else "test_reader",
        use_double_buffer=True)
    img, label = fluid.layers.read_file(reader)

    hidden = img

    for i in six.moves.xrange(2):
        hidden = fluid.layers.fc(input=hidden, size=100, act='tanh')
        hidden = fluid.layers.dropout(
            hidden, dropout_prob=0.5, is_test=not is_train)

    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    return fluid.layers.mean(loss), reader


def main():
    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            loss, train_reader = network(True)
            adam = fluid.optimizer.Adam(learning_rate=0.01)
            adam.minimize(loss)

    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        with fluid.unique_name.guard():
            test_loss, test_reader = network(False)

    use_cuda = fluid.core.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    fluid.Executor(place).run(startup_prog)
    fluid.Executor(place).run(test_startup)

    trainer = fluid.ParallelExecutor(
        use_cuda=use_cuda, loss_name=loss.name, main_program=train_prog)

    tester = fluid.ParallelExecutor(
        use_cuda=use_cuda, share_vars_from=trainer, main_program=test_prog)

    train_reader.decorate_paddle_reader(
        paddle.reader.shuffle(
            paddle.batch(mnist.train(), 512), buf_size=8192))

    test_reader.decorate_paddle_reader(paddle.batch(mnist.test(), 512))

    for epoch_id in six.moves.xrange(10):
        train_reader.start()
        try:
            while True:
                print(
                    'train_loss',
                    numpy.array(trainer.run(fetch_list=[loss.name])))
        except fluid.core.EOFException:
            print('End of epoch', epoch_id)
            train_reader.reset()

        test_reader.start()
        try:
            while True:
                print(
                    'test loss',
                    numpy.array(tester.run(fetch_list=[test_loss.name])))
        except fluid.core.EOFException:
            print('End of testing')
            test_reader.reset()


if __name__ == '__main__':
    main()
