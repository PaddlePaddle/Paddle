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

import paddle.fluid as fluid
import paddle.dataset.mnist as mnist
import paddle
import paddle.v2
import threading
import numpy


def network(is_train):
    reader, queue = fluid.layers.py_reader(
        capacity=10,
        shapes=((-1, 784), (-1, 1)),
        dtypes=('float32', 'int64'),
        name="train_reader" if is_train else "test_reader")
    img, label = fluid.layers.read_file(reader)

    hidden = img

    for i in xrange(2):
        hidden = fluid.layers.fc(input=hidden, size=100, act='tanh')
        hidden = fluid.layers.dropout(
            hidden, dropout_prob=0.5, is_test=not is_train)

    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    return fluid.layers.mean(loss), queue, reader


def pipe_reader_to_queue(reader_creator, queue):
    with fluid.program_guard(fluid.Program(), fluid.Program()):
        feeder = fluid.DataFeeder(
            feed_list=[
                fluid.layers.data(
                    name='img', dtype='float32', shape=[784]),
                fluid.layers.data(
                    name='label', dtype='int64', shape=[1])
            ],
            place=fluid.CPUPlace())

    def __thread_main__():
        for data in feeder.decorate_reader(
                reader_creator, multi_devices=False)():
            tmp = fluid.core.LoDTensorArray()
            tmp.append(data['img'])
            tmp.append(data['label'])
            queue.push(tmp)
        queue.close()

    th = threading.Thread(target=__thread_main__)
    th.start()
    return th


def main():
    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            loss, train_queue, train_reader = network(True)
            adam = fluid.optimizer.Adam(learning_rate=0.01)
            adam.minimize(loss)

    test_prog = fluid.Program()
    test_startup = fluid.Program()
    with fluid.program_guard(test_prog, test_startup):
        with fluid.unique_name.guard():
            test_loss, test_queue, test_reader = network(False)

    fluid.Executor(fluid.CUDAPlace(0)).run(startup_prog)
    fluid.Executor(fluid.CUDAPlace(0)).run(test_startup)

    trainer = fluid.ParallelExecutor(
        use_cuda=True, loss_name=loss.name, main_program=train_prog)

    tester = fluid.ParallelExecutor(
        use_cuda=True, share_vars_from=trainer, main_program=test_prog)

    for epoch_id in xrange(10):
        train_data_thread = pipe_reader_to_queue(
            paddle.batch(paddle.v2.reader.firstn(mnist.train(), 32), 64),
            train_queue)
        try:
            while True:
                print 'train_loss', numpy.array(
                    trainer.run(fetch_list=[loss.name]))
        except fluid.core.EOFException:
            print 'End of epoch', epoch_id
            train_reader.reset()
        train_data_thread.join()

        test_data_thread = pipe_reader_to_queue(
            paddle.batch(mnist.test(), 32), test_queue)
        try:
            while True:
                print 'test loss', numpy.array(
                    tester.run(fetch_list=[test_loss.name]))
        except fluid.core.EOFException:
            print 'End of testing'
            test_reader.reset()

        test_data_thread.join()
        break
    del trainer
    del tester


if __name__ == '__main__':
    main()
