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

import os
import unittest

import numpy
import time
import paddle
import paddle.fluid as fluid

BATCH_SIZE = 64


def convolutional_neural_network(use_py_reader):
    with fluid.unique_name.guard():
        img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        py_reader = None
        if use_py_reader:
            py_reader = fluid.io.DataLoader.from_generator(
                capacity=64,
                feed_list=[img, label],
                iterable=False,
                use_double_buffer=False)

        conv_pool_1 = fluid.nets.simple_img_conv_pool(input=img,
                                                      filter_size=5,
                                                      num_filters=20,
                                                      pool_size=2,
                                                      pool_stride=2,
                                                      act="relu")
        conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
        conv_pool_2 = fluid.nets.simple_img_conv_pool(input=conv_pool_1,
                                                      filter_size=5,
                                                      num_filters=50,
                                                      pool_size=2,
                                                      pool_stride=2,
                                                      act="relu")

        prediction = fluid.layers.fc(input=conv_pool_2, size=10, act='softmax')
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_loss = paddle.mean(loss)
        acc = fluid.layers.accuracy(input=prediction, label=label)
        i = fluid.layers.zeros(shape=[1], dtype='int64')
        array = fluid.layers.array_write(x=prediction, i=i)
        fluid.layers.increment(i)
        fluid.layers.array_write(x=acc, i=i, array=array)
        return array, img, label, prediction, avg_loss, acc, py_reader


def test():
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    test_reader = paddle.batch(paddle.dataset.mnist.test(),
                               batch_size=BATCH_SIZE)

    array, img, label, prediction, avg_loss, acc, py_reader = convolutional_neural_network(
        use_py_reader=False)
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

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

    # test for epoch
    avg_loss_val, acc_val = train_test(
        train_test_program=fluid.default_main_program(),
        train_test_reader=test_reader,
        train_test_feed=feeder)

    print("Test: avg_cost: %s, acc: %s" % (avg_loss_val, acc_val))
    assert acc_val > 0.96


def train(use_cuda, thread_num, cpu_num):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        print("paddle is not compiled with cuda, exit!")
        return

    array, img, label, prediction, avg_loss, acc, py_reader = convolutional_neural_network(
        use_py_reader=True)
    print("build convolutional neural network done.")

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)
    print("Adam optimizer minimize done.")

    train_reader = paddle.batch(paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
                                batch_size=BATCH_SIZE)
    print("declared train reader done.")

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    print("going to run startup program")
    exe.run(fluid.default_startup_program())
    print("run startup program done.")

    os.environ['CPU_NUM'] = str(cpu_num)

    print("cpu_num:" + str(cpu_num))
    print("thread_num:" + str(thread_num))

    build_strategy = fluid.BuildStrategy()
    build_strategy.async_mode = True

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = thread_num
    exec_strategy.num_iteration_per_run = 10

    main_program = fluid.default_main_program()
    pe = fluid.ParallelExecutor(use_cuda=False,
                                loss_name=avg_loss.name,
                                main_program=main_program,
                                build_strategy=build_strategy,
                                exec_strategy=exec_strategy)
    print("declare parallel executor done.")

    py_reader.set_sample_list_generator(train_reader)

    for pass_id in range(2):
        step = 0
        py_reader.start()
        try:
            while True:
                array_v, acc_v, prediction_v, loss_val = pe.run(
                    fetch_list=[array, acc, prediction, avg_loss.name])

                assert numpy.allclose(array_v[0], prediction_v) == True
                assert numpy.allclose(array_v[1], acc_v) == True

                loss_val = numpy.mean(loss_val)
                if step % 10 == 0:
                    print("Pass %d, Batch %d, Cost %f, queue size %d" %
                          (pass_id, step, loss_val, py_reader.queue.size()))
                step += 1
        except fluid.core.EOFException:
            print("train end pass = " + str(pass_id))
            py_reader.reset()

    return step


class TestAsyncSSAGraphExecutor(unittest.TestCase):

    def test_check_async_ssa_exe_train(self):
        step_list = []
        for cpu_num in [1, 2, 4]:
            print("run cpu_num -> " + str(cpu_num))
            with fluid.scope_guard(fluid.core.Scope()):
                with fluid.program_guard(main_program=fluid.Program(),
                                         startup_program=fluid.Program()):
                    start_time = time.time()
                    step = train(use_cuda=False,
                                 thread_num=cpu_num,
                                 cpu_num=cpu_num)
                    end_time = time.time()
                    step_list.append(step)
                print("cpu_num -> " + str(cpu_num) + " step -> " + str(step) +
                      " time -> " + str(end_time - start_time))
                with fluid.program_guard(main_program=fluid.Program(),
                                         startup_program=fluid.Program()):
                    test()
        assert abs(int(step_list[0] / 2) - int(step_list[1])) < 5
        assert abs(int(step_list[1] / 2) - int(step_list[2])) < 5


if __name__ == "__main__":
    unittest.main()
