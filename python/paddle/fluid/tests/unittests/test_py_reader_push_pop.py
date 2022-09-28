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

import unittest
import paddle.fluid as fluid
import numpy as np
from threading import Thread


def feed_data(feed_queue, inputs):
    for in_data in inputs:
        feed_queue.push(in_data)


class TestPyReader(unittest.TestCase):

    def setUp(self):
        self.capacity = 10
        self.batch_size_min = 10
        self.batch_size_max = 20
        self.shapes = [(-1, 3, 2, 1), (-1, 1)]
        self.lod_levels = [0, 0]
        self.dtypes = ['float32', 'int64']
        self.iterations = 20

    def test_single_thread_main(self):
        self.main(use_thread=False)

    def test_multiple_thread_main(self):
        self.main(use_thread=True)

    def main(self, use_thread=False):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            place = fluid.CUDAPlace(
                0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
            executor = fluid.Executor(place)

            data_file = fluid.layers.py_reader(capacity=self.capacity,
                                               dtypes=self.dtypes,
                                               lod_levels=self.lod_levels,
                                               shapes=self.shapes)
            feed_queue = data_file.queue
            read_out_data = fluid.layers.read_file(data_file)
            self.inputs = []

            for i in range(self.iterations):
                in_data = fluid.LoDTensorArray()
                batch_size = np.random.random_integers(self.batch_size_min,
                                                       self.batch_size_max)
                for shape, dtype in zip(self.shapes, self.dtypes):
                    next_data = np.random.uniform(low=0,
                                                  high=1000,
                                                  size=(batch_size, ) +
                                                  shape[1:]).astype(dtype)
                    in_data.append(
                        fluid.executor._as_lodtensor(next_data, place))

                self.inputs.append(in_data)

            executor.run(fluid.default_startup_program())
            self.outputs = []
            if use_thread:
                thread = Thread(target=feed_data,
                                args=(feed_queue, self.inputs))
                thread.start()
                for in_data in self.inputs:
                    self.outputs.append(
                        executor.run(fetch_list=list(read_out_data)))
            else:
                for in_data in self.inputs:
                    feed_queue.push(in_data)
                    self.outputs.append(
                        executor.run(fetch_list=list(read_out_data)))

            feed_queue.close()
            self.validate()

    def validate(self):
        self.assertEqual(len(self.inputs), len(self.outputs))
        for in_data_list, out_data_list in zip(self.inputs, self.outputs):
            self.assertEqual(len(in_data_list), len(out_data_list))
            in_data_list_np = [
                np.array(in_lod_tensor) for in_lod_tensor in in_data_list
            ]
            for in_data, out_data in zip(in_data_list_np, out_data_list):
                self.assertTrue((in_data == out_data).all())


if __name__ == '__main__':
    unittest.main()
