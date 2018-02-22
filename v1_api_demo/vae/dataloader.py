# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import numpy as np


class MNISTloader():
    def __init__(self,
                 data_path="./data/mnist_data/",
                 batch_size=60,
                 process='train'):
        self.batch_size = batch_size
        self.data_path = data_path
        self._pointer = 0
        self.image_batches = np.array([])
        self.process = process

    def _extract_images(self, filename, n):
        f = open(filename, 'rb')
        f.read(16)
        data = np.fromfile(f, 'ubyte', count=n * 28 * 28).reshape((n, 28 * 28))
        #Mapping data into [-1, 1]
        data = data / 255. * 2. - 1
        data_batches = np.split(data, 60000 / self.batch_size, 0)

        f.close()

        return data_batches

    @property
    def pointer(self):
        return self._pointer

    def load_data(self):
        TRAIN_IMAGES = '%s/train-images-idx3-ubyte' % self.data_path
        TEST_IMAGES = '%s/t10k-images-idx3-ubyte' % self.data_path

        if self.process == 'train':
            self.image_batches = self._extract_images(TRAIN_IMAGES, 60000)
        else:
            self.image_batches = self._extract_images(TEST_IMAGES, 10000)

    def next_batch(self):
        batch = self.image_batches[self._pointer]
        self._pointer = (self._pointer + 1) % (60000 / self.batch_size)
        return np.array(batch)

    def reset_pointer(self):
        self._pointer = 0
