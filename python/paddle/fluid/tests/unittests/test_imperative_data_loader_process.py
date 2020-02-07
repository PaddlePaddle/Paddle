# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
import numpy as np
import paddle.fluid as fluid

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label


def batch_generator_creator(batch_size, batch_num):
    def __reader__():
        for _ in range(batch_num):
            batch_image, batch_label = get_random_images_and_labels(
                [batch_size, 784], [batch_size, 1])
            yield batch_image, batch_label

    return __reader__


# NOTE: coverage CI can't cover child process code, so need these test.
# Here test child process loop function in main process
class TestDygraphhDataLoaderProcess(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.batch_num = 4
        self.epoch_num = 2
        self.capacity = 2

    def test_reader_process_loop(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(
                capacity=self.batch_num + 1, use_multiprocess=True)
            loader.set_batch_generator(
                batch_generator_creator(self.batch_size, self.batch_num),
                places=fluid.CPUPlace())
            loader._data_queue = queue.Queue(self.batch_num + 1)
            loader._reader_process_loop()
            for _ in range(self.batch_num):
                loader._data_queue.get(timeout=10)

    def test_reader_process_loop_simple_none(self):
        def none_sample_genarator(batch_num):
            def __reader__():
                for _ in range(batch_num):
                    yield None

            return __reader__

        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(
                capacity=self.batch_num + 1, use_multiprocess=True)
            loader.set_batch_generator(
                none_sample_genarator(self.batch_num), places=fluid.CPUPlace())
            loader._data_queue = queue.Queue(self.batch_num + 1)
            exception = None
            try:
                loader._reader_process_loop()
            except AttributeError as ex:
                exception = ex
            self.assertIsNotNone(exception)


if __name__ == '__main__':
    unittest.main()
