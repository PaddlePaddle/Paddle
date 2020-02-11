# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import unittest
import six


class TestInferencePartialFeedBase(unittest.TestCase):
    def setUp(self):
        self.epoch_num = 3
        self.batch_num = 100
        self.batch_size = 32

    def create_reader(self):
        def __impl__():
            for _ in six.moves.range(self.batch_num):
                yield np.random.random([self.batch_size, 1]).astype('float32'),

        return __impl__

    def run_network(self, iterable, use_cuda):
        x = fluid.data(shape=[None, 1], name='x', dtype='float32')
        places = fluid.cuda_places([0, 1]) if use_cuda else fluid.cpu_places(4)
        loader = fluid.io.DataLoader.from_generator(
            feed_list=[x], capacity=16, iterable=iterable)
        y = fluid.layers.fc(x, size=10)
        loss = fluid.layers.reduce_mean(y)

        exe = fluid.Executor(places[0])
        exe.run(fluid.default_startup_program())

        prog = fluid.CompiledProgram(fluid.default_main_program(
        )).with_data_parallel(
            places=places, loss_name=loss.name)

        loader.set_batch_generator(
            self.create_reader(), places=places if iterable else None)

        for _ in six.moves.range(self.epoch_num):
            actual_batch_num = 0
            if loader.iterable:
                for feed_data in loader():
                    x_data, = exe.run(prog, feed=feed_data, fetch_list=[x])
                    self.assertEqual(x_data.shape[0] % self.batch_size, 0)
                    self.assertTrue(x_data.shape[0] != 0)
                    actual_batch_num += int(x_data.shape[0] / self.batch_size)
            else:
                loader.start()
                try:
                    while True:
                        x_data, = exe.run(prog, fetch_list=[x])
                        self.assertEqual(x_data.shape[0] % self.batch_size, 0)
                        self.assertTrue(x_data.shape[0] != 0)
                        actual_batch_num += int(x_data.shape[0] /
                                                self.batch_size)
                except fluid.core.EOFException:
                    loader.reset()

            self.assertEqual(self.batch_num, actual_batch_num)

    def test_main(self):
        use_cuda_list = [False, True] if fluid.is_compiled_with_cuda(
        ) else [False]
        iterable_list = [False, True]
        for iterable in iterable_list:
            for use_cuda in use_cuda_list:
                with fluid.program_guard(fluid.Program(), fluid.Program()):
                    with fluid.scope_guard(fluid.Scope()):
                        self.run_network(iterable, use_cuda)


if __name__ == '__main__':
    unittest.main()
