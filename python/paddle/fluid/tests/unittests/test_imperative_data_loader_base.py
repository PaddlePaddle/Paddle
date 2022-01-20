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
from paddle.fluid import core
from paddle.fluid.reader import use_pinned_memory
from paddle.fluid.framework import _test_eager_guard


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


class TestDygraphDataLoader(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.batch_num = 4
        self.epoch_num = 1
        self.capacity = 5

    def iter_loader_data(self, loader):
        for _ in range(self.epoch_num):
            for image, label in loader():
                relu = fluid.layers.relu(image)
                self.assertEqual(image.shape, [self.batch_size, 784])
                self.assertEqual(label.shape, [self.batch_size, 1])
                self.assertEqual(relu.shape, [self.batch_size, 784])

    def func_test_single_process_loader(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(
                capacity=self.capacity, iterable=False, use_multiprocess=False)
            loader.set_sample_generator(
                sample_generator_creator(self.batch_size, self.batch_num),
                batch_size=self.batch_size,
                places=fluid.CPUPlace())
            self.iter_loader_data(loader)

    def test_single_process_loader(self):
        with _test_eager_guard():
            self.func_test_single_process_loader()
        self.func_test_single_process_loader()

    def func_test_multi_process_loader(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(
                capacity=self.capacity, use_multiprocess=True)
            loader.set_sample_generator(
                sample_generator_creator(self.batch_size, self.batch_num),
                batch_size=self.batch_size,
                places=fluid.CPUPlace())
            self.iter_loader_data(loader)

    def test_multi_process_loader(self):
        with _test_eager_guard():
            self.func_test_multi_process_loader()
        self.func_test_multi_process_loader()

    def func_test_generator_no_places(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(capacity=self.capacity)
            loader.set_sample_generator(
                sample_generator_creator(self.batch_size, self.batch_num),
                batch_size=self.batch_size)
            self.iter_loader_data(loader)

    def test_generator_no_places(self):
        with _test_eager_guard():
            self.func_test_generator_no_places()
        self.func_test_generator_no_places()

    def func_test_set_pin_memory(self):
        with fluid.dygraph.guard():
            use_pinned_memory(False)
            loader = fluid.io.DataLoader.from_generator(
                capacity=self.capacity, iterable=False, use_multiprocess=False)
            loader.set_sample_generator(
                sample_generator_creator(self.batch_size, self.batch_num),
                batch_size=self.batch_size,
                places=fluid.CPUPlace())
            self.iter_loader_data(loader)
            use_pinned_memory(True)

    def test_set_pin_memory(self):
        with _test_eager_guard():
            self.func_test_set_pin_memory()
        self.func_test_set_pin_memory()


if __name__ == '__main__':
    unittest.main()
