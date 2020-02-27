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

import sys
import unittest
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import core


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


class TestDygraphDataLoaderMmapFdsClear(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.batch_num = 100
        self.epoch_num = 2
        self.capacity = 50

    def prepare_data_loader(self):
        loader = fluid.io.DataLoader.from_generator(
            capacity=self.capacity, use_multiprocess=True)
        loader.set_batch_generator(
            batch_generator_creator(self.batch_size, self.batch_num),
            places=fluid.CPUPlace())
        return loader

    def run_one_epoch_with_break(self, loader):
        for step_id, data in enumerate(loader()):
            image, label = data
            relu = fluid.layers.relu(image)
            self.assertEqual(image.shape, [self.batch_size, 784])
            self.assertEqual(label.shape, [self.batch_size, 1])
            self.assertEqual(relu.shape, [self.batch_size, 784])
            if step_id == 30:
                break

    def test_data_loader_break(self):
        with fluid.dygraph.guard():
            loader = self.prepare_data_loader()
            for _ in range(self.epoch_num):
                self.run_one_epoch_with_break(loader)
                break

    def test_data_loader_continue_break(self):
        with fluid.dygraph.guard():
            loader = self.prepare_data_loader()
            for _ in range(self.epoch_num):
                self.run_one_epoch_with_break(loader)


if __name__ == '__main__':
    unittest.main()
