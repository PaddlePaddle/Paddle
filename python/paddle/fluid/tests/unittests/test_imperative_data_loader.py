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

import os
import sys
import signal
import unittest
import numpy as np

import paddle.fluid as fluid
from paddle.fluid import core
import paddle.compat as cpt

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


def set_signal_handler(sig):
    current_handler = signal.getsignal(sig)
    if not callable(current_handler):
        current_handler = None

    def __handler__(signum, frame):
        core._throw_error_if_process_failed()
        if current_handler is not None:
            current_handler(signum, frame)

    signal.signal(sig, __handler__)


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
                    relu = fluid.layers.relu(image)
                    self.assertEqual(image.shape, [self.batch_size, 784])
                    self.assertEqual(label.shape, [self.batch_size, 1])
                    self.assertEqual(relu.shape, [self.batch_size, 784])

    def test_sample_genarator(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(capacity=2)
            loader.set_sample_generator(
                sample_generator_creator(self.batch_size, self.batch_num),
                batch_size=self.batch_size,
                places=fluid.CPUPlace())
            for _ in range(self.epoch_num):
                for image, label in loader():
                    relu = fluid.layers.relu(image)
                    self.assertEqual(image.shape, [self.batch_size, 784])
                    self.assertEqual(label.shape, [self.batch_size, 1])
                    self.assertEqual(relu.shape, [self.batch_size, 784])

    def test_sample_list_generator(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(capacity=2)
            loader.set_sample_list_generator(
                sample_list_generator_creator(self.batch_size, self.batch_num),
                places=fluid.CPUPlace())
            for _ in range(self.epoch_num):
                for image, label in loader():
                    relu = fluid.layers.relu(image)
                    self.assertEqual(image.shape, [self.batch_size, 784])
                    self.assertEqual(label.shape, [self.batch_size, 1])
                    self.assertEqual(relu.shape, [self.batch_size, 784])

    def test_batch_genarator(self):
        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(capacity=2)
            loader.set_batch_generator(
                batch_generator_creator(self.batch_size, self.batch_num),
                places=fluid.CPUPlace())
            for _ in range(self.epoch_num):
                for image, label in loader():
                    relu = fluid.layers.relu(image)
                    self.assertEqual(image.shape, [self.batch_size, 784])
                    self.assertEqual(label.shape, [self.batch_size, 1])
                    self.assertEqual(relu.shape, [self.batch_size, 784])

    # NOTE: coverage CI can't cover child process code, so need this test.
    # It test child process loop function in main process
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
                loader._data_queue.get(timeout=10)

    # NOTE: exception tests
    def test_single_process_with_thread_expection(self):
        def error_sample_genarator(batch_num):
            def __reader__():
                for _ in range(batch_num):
                    yield [[[1, 2], [1]]]

            return __reader__

        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(
                capacity=2, iterable=False, use_multiprocess=False)
            loader.set_batch_generator(
                error_sample_genarator(self.batch_num), places=fluid.CPUPlace())
            exception = None
            try:
                for _ in loader():
                    print("test_single_process_with_thread_expection")
            except core.EnforceNotMet as ex:
                self.assertIn("Blocking queue is killed",
                              cpt.get_exception_message(ex))
                exception = ex
            self.assertIsNotNone(exception)

    def test_multi_process_with_thread_expection(self):
        def error_sample_genarator(batch_num):
            def __reader__():
                for _ in range(batch_num):
                    yield [[[1, 2], [1]]]

            return __reader__

        with fluid.dygraph.guard():
            loader = fluid.io.DataLoader.from_generator(capacity=2)
            loader.set_batch_generator(
                error_sample_genarator(self.batch_num), places=fluid.CPUPlace())
            exception = None
            try:
                for _ in loader():
                    print("test_multi_process_with_thread_expection")
            except core.EnforceNotMet as ex:
                self.assertIn("Blocking queue is killed",
                              cpt.get_exception_message(ex))
                exception = ex
            self.assertIsNotNone(exception)


if __name__ == '__main__':
    unittest.main()
