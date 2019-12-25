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
import paddle.fluid as fluid
from paddle.fluid import core
import unittest
import numpy as np

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


def get_random_images_and_labels(image_shape, label_shape):
    image = np.random.random(size=image_shape).astype('float32')
    label = np.random.random(size=label_shape).astype('int64')
    return image, label


def sample_generator_creator(batch_size, batch_num):
    def __reader__():
        for _ in range(batch_num * batch_size):
            image, label = get_random_images_and_labels([784], [1])
            yield image, label

    return __reader__


def sample_list_generator_creator(batch_size, batch_num):
    def __reader__():
        for _ in range(batch_num):
            sample_list = []
            for _ in range(batch_size):
                image, label = get_random_images_and_labels([784], [1])
                sample_list.append([image, label])

            yield sample_list

    return __reader__


def batch_generator_creator(batch_size, batch_num):
    def __reader__():
        for _ in range(batch_num):
            batch_image, batch_label = get_random_images_and_labels(
                [batch_size, 784], [batch_size, 1])
            yield batch_image, batch_label

    return __reader__


class TestdygraphhDataLoader(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.batch_num = 4
        self.epoch_num = 2

    def test_not_capacity(self):
        with fluid.dygraph.guard():
            with self.assertRaisesRegexp(ValueError,
                                         "Please give value to capacity."):
                fluid.io.DataLoader.from_generator()

    def test_single_process(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(
                capacity=2, iterable=False, use_multiprocess=False)
            loader.set_sample_generator(
                sample_generator_creator(self.batch_size, self.batch_num),
                batch_size=self.batch_size,
                places=fluid.CPUPlace())
            for _ in range(self.epoch_num):
                for image, label in loader():
                    fluid.layers.relu(image)

    def test_sample_genarator(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(capacity=2)
            loader.set_sample_generator(
                sample_generator_creator(self.batch_size, self.batch_num),
                batch_size=self.batch_size,
                places=fluid.CPUPlace())
            for _ in range(self.epoch_num):
                for image, label in loader():
                    fluid.layers.relu(image)

    def test_sample_list_generator(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(capacity=2)
            loader.set_sample_list_generator(
                sample_list_generator_creator(self.batch_size, self.batch_num),
                places=fluid.CPUPlace())
            for _ in range(self.epoch_num):
                for image, label in loader():
                    fluid.layers.relu(image)

    def test_batch_genarator(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(capacity=2)
            loader.set_batch_generator(
                batch_generator_creator(self.batch_size, self.batch_num),
                places=fluid.CPUPlace())
            for _ in range(self.epoch_num):
                for image, label in loader():
                    fluid.layers.relu(image)

    # NOTE: coverage CI can't cover child process code, so need this test
    def test_reader_process_loop(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(
                capacity=self.batch_num + 1)
            loader.set_batch_generator(
                batch_generator_creator(self.batch_size, self.batch_num),
                places=fluid.CPUPlace())
        loader._data_queue = queue.Queue(self.batch_num + 1)
        loader._reader_process_loop()
        for _ in range(self.batch_num):
            sample = loader._data_queue.get(timeout=10)


if __name__ == '__main__':
    unittest.main()
