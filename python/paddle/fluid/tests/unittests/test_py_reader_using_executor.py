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
import paddle.fluid as fluid
from paddle.fluid import compiler
import paddle.fluid.core as core
import numpy as np
import threading
import multiprocessing
import os


def as_tensor(np_array_or_tensor, place=None):
    if isinstance(np_array_or_tensor, fluid.LoDTensor):
        return np_array_or_tensor

    if place is None:
        place = fluid.CPUPlace()

    tensor = fluid.LoDTensor()
    tensor.set(np_array_or_tensor, place)
    return tensor


def as_numpy(tensor_or_numpy):
    return tensor_or_numpy if isinstance(
        tensor_or_numpy, np.ndarray) else np.array(tensor_or_numpy)


def feed_data(feed_queue, reader):
    data_generator = reader()
    while True:
        data = next(data_generator, None)
        if data is None or not feed_queue.push(data):
            break


def simple_fc_net(in_size,
                  class_num,
                  hidden_sizes,
                  batch_size,
                  queue_capacity,
                  use_double_buffer=False,
                  use_feed_list=True):
    if use_feed_list:
        data = fluid.layers.data(name="data", dtype='float32', shape=[in_size])
        label = fluid.layers.data(name='label', dtype='int64', shape=[1])
        py_reader = fluid.layers.create_py_reader_by_data(
            capacity=queue_capacity,
            use_double_buffer=False,
            feed_list=[data, label])
    else:
        py_reader = fluid.layers.py_reader(
            capacity=queue_capacity,
            shapes=[[-1, in_size], [-1, 1]],
            lod_levels=[0, 0],
            dtypes=['float32', 'int64'],
            use_double_buffer=False)
    feed_queue = py_reader.queue
    reader = fluid.layers.batch(py_reader, batch_size=batch_size)
    if use_double_buffer:
        reader = fluid.layers.double_buffer(reader)

    in_data, label = fluid.layers.read_file(reader)

    hidden = in_data
    for hidden_size in hidden_sizes:
        hidden = fluid.layers.fc(
            hidden,
            size=hidden_size,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))

    predict_label = fluid.layers.fc(hidden, size=class_num, act='softmax')
    loss = fluid.layers.mean(
        fluid.layers.cross_entropy(
            input=predict_label, label=label))

    optimizer = fluid.optimizer.Adam()
    optimizer.minimize(loss)
    return in_data, label, loss, optimizer, feed_queue, py_reader


class TestPyReaderUsingExecutor(unittest.TestCase):
    def setUp(self):
        self.in_size = 1000
        self.hidden_sizes = [50, 30, 20]
        self.class_num = 10
        self.batch_size = 32
        self.iterations = 10
        self.queue_capacity = 50

    def test(self):
        for use_cuda in ([False, True]
                         if core.is_compiled_with_cuda() else [False]):
            for use_parallel_executor in [False, True]:
                for use_double_buffer in [False, True]:
                    for use_feed_list in [False, True]:
                        for use_decorate_paddle_reader in [False, True]:
                            print('Test Parameters:'),
                            print({
                                'use_cuda': use_cuda,
                                'use_parallel_executor': use_parallel_executor,
                                'use_double_buffer': use_double_buffer,
                                'use_feed_list': use_feed_list,
                                'use_decorate_paddle_reader':
                                use_decorate_paddle_reader
                            })
                            self.main(use_cuda, use_parallel_executor,
                                      use_double_buffer, use_feed_list,
                                      use_decorate_paddle_reader)

    def tensor_reader(self, use_decorate_paddle_reader):
        def reader():
            self.inputs = []
            cnt = 0
            while True:
                tensors = fluid.LoDTensorArray()
                in_data = np.random.uniform(
                    low=0, high=1, size=(1, self.in_size)).astype('float32')
                tensors.append(as_tensor(in_data))
                label = np.random.random_integers(
                    low=0, high=self.class_num - 1, size=(1, 1)).astype('int64')
                tensors.append(as_tensor(label))

                if cnt < self.iterations * self.batch_size * self.batch_size_times:
                    if cnt % (self.batch_size * self.batch_size_times) == 0:
                        self.inputs.append([in_data, label])
                    else:
                        self.inputs[-1][0] = np.concatenate(
                            (self.inputs[-1][0], in_data), axis=0)
                        self.inputs[-1][1] = np.concatenate(
                            (self.inputs[-1][1], label), axis=0)
                elif not self.use_double_buffer:
                    break

                if use_decorate_paddle_reader:
                    yield [(in_data, label)]
                else:
                    yield tensors
                cnt += 1

            if not use_decorate_paddle_reader:
                yield None

        return reader

    def main(self,
             use_cuda=True,
             use_parallel_executor=False,
             use_double_buffer=False,
             use_feed_list=False,
             use_decorate_paddle_reader=False):
        assert not use_cuda or use_cuda and core.is_compiled_with_cuda()

        self.use_cuda = use_cuda
        self.use_parallel_executor = use_parallel_executor
        self.use_double_buffer = use_double_buffer
        self.use_feed_list = use_feed_list
        self.use_decorate_paddle_reader = use_decorate_paddle_reader

        startup_program = fluid.Program()
        main_program = fluid.Program()

        with fluid.program_guard(main_program, startup_program):
            in_data, label, loss, optimizer, feed_queue, py_reader = simple_fc_net(
                in_size=self.in_size,
                class_num=self.class_num,
                hidden_sizes=self.hidden_sizes,
                batch_size=self.batch_size,
                queue_capacity=self.queue_capacity,
                use_double_buffer=self.use_double_buffer,
                use_feed_list=self.use_feed_list)

            place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

            exe = fluid.Executor(place)
            exe.run(startup_program)

            train_cp = compiler.CompiledProgram(main_program)
            if use_parallel_executor:
                train_cp = train_cp.with_data_parallel(loss_name=loss.name)
                if use_cuda:
                    self.batch_size_times = core.get_cuda_device_count()
                else:
                    self.batch_size_times = int(
                        os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
            else:
                self.batch_size_times = 1

            reader = self.tensor_reader(use_decorate_paddle_reader)
            if use_decorate_paddle_reader:
                py_reader.decorate_paddle_reader(reader)
                py_reader.start()
            else:
                thread = threading.Thread(
                    target=feed_data, args=(feed_queue, reader))
                thread.daemon = True
                thread.start()

            self.outputs = []
            for _ in range(self.iterations):
                fetches = exe.run(train_cp,
                                  fetch_list=[in_data.name, label.name])
                fetches = [as_numpy(fetch) for fetch in fetches]
                self.outputs.append(fetches)

            feed_queue.close()
            self.validate()
            if use_decorate_paddle_reader:
                py_reader.exited = True
                py_reader.thread.join()
            else:
                thread.join()

    def validate(self):
        self.assertEqual(len(self.inputs), len(self.outputs))
        for batch_in, batch_out in zip(self.inputs, self.outputs):
            self.assertEqual(len(batch_in), len(batch_out))
            if self.use_parallel_executor and not self.use_double_buffer:
                self.validate_unordered_batch(batch_in, batch_out)
            else:
                for in_data, out_data in zip(batch_in, batch_out):
                    self.assertEqual(in_data.shape, out_data.shape)
                    if not self.use_parallel_executor:
                        self.assertTrue((in_data == out_data).all())

    def validate_unordered_batch(self, batch_in, batch_out):
        out_index_left_set = set(range(self.batch_size * self.batch_size_times))
        mapping_num = 0
        for i in range(self.batch_size * self.batch_size_times):
            for j in out_index_left_set:
                flag = True
                for k in range(len(batch_in)):
                    in_data = batch_in[k][i]
                    out_data = batch_out[k][j]
                    if (in_data != out_data).any():
                        flag = False
                        break

                if flag:
                    out_index_left_set.remove(j)
                    mapping_num += 1
                    break

        self.assertEqual(mapping_num, self.batch_size * self.batch_size_times)


if __name__ == '__main__':
    unittest.main()
