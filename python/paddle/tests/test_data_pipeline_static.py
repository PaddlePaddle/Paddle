# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.utils.download import get_path_from_url
from paddle.vision.datasets import DatasetFolder
from paddle.vision.ops import image_decode_random_crop, image_resize, \
                                random_flip, mirror_normalize
from paddle.vision.reader import file_label_reader

DATASET_HOME = os.path.expanduser("~/.cache/paddle/datasets")
DATASET_URL = "https://paddlemodels.cdn.bcebos.com/ImageNet_stub.tar"
DATASET_MD5 = "c7110519124a433901cf005a4a91b607"
IMAGE_NUM = 100


class TestDataPipelineStaticCase1(unittest.TestCase):
    def setUp(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)

        self.num_epoches = 2
        self.batch_size = 16
        self.num_threads = 2
        self.host_memory_padding = 1000000
        self.device_memory_padding = 1000000

        self.shuffle = False
        self.drop_last = True
        self.calc_iter_info()

        self.target_size = 224
        self.flip_prob = 0.5
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.120, 57.375]

        self.mean_np = np.array(self.mean).reshape([1, 3, 1, 1])
        self.std_np = np.array(self.std).reshape([1, 3, 1, 1])

        self.build_reader()

    def calc_iter_info(self):
        if self.drop_last:
            self.num_iters = IMAGE_NUM // self.batch_size
        else:
            self.num_iters = (IMAGE_NUM + self.batch_size - 1) \
                                        // self.batch_size

        if self.drop_last:
            self.last_batch_size = self.batch_size
        else:
            self.last_batch_size = IMAGE_NUM % self.batch_size
            if self.last_batch_size == 0:
                self.last_batch_size = self.batch_size

    def build_reader(self):
        def imagenet_reader():
            image, label = file_label_reader(
                self.data_root,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last)

            def decode(image):
                image = image_decode_random_crop(
                    image, num_threads=self.num_threads)
                return image

            def resize(image):
                image = image_resize(image, size=self.target_size)
                return image

            def flip_normalize(image):
                mirror = random_flip(image, prob=self.flip_prob)
                image = mirror_normalize(
                    image, mirror, mean=self.mean, std=self.std)
                return image

            image = paddle.io.map(decode, image)
            image = paddle.io.map(resize, image)
            image = paddle.io.map(flip_normalize, image)

            return {'image': image, 'label': label}

        self.reader = imagenet_reader

    def test_output(self):
        # NOTE: only supoort CUDA kernel currently
        if not core.is_compiled_with_cuda():
            return

        loader = paddle.io.DataLoader(self.reader)

        for eid in range(self.num_epoches):
            num_iters = 0
            for data in loader:
                image = data['image'].numpy()
                assert image.shape[0] == self.batch_size
                assert image.shape[1] == 3
                assert image.shape[2] == self.target_size
                assert image.shape[3] == self.target_size
                assert image.dtype == np.float32

                restore_image = image * self.std_np + self.mean_np
                assert np.all(restore_image > -1.)
                assert np.all(restore_image < 256.)

                label = data['label'].numpy()
                assert label.shape[0] == self.batch_size
                assert label.dtype == np.int64
                assert np.all(label >= 0)
                assert np.all(label <= 1)

                num_iters += 1

            assert num_iters == self.num_iters
            if eid < self.num_epoches - 1:
                loader.reset()


class TestDataPipelineStaticCase2(TestDataPipelineStaticCase1):
    def setUp(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)

        self.num_epoches = 1
        self.batch_size = 32
        self.num_threads = 4
        self.host_memory_padding = 0
        self.device_memory_padding = 0

        self.shuffle = True
        self.drop_last = True
        self.calc_iter_info()

        self.target_size = 128
        self.flip_prob = 0.5
        self.mean = [123.675, 116.28, 103.53]
        self.std = [58.395, 57.120, 57.375]

        self.mean_np = np.array(self.mean).reshape([1, 3, 1, 1])
        self.std_np = np.array(self.std).reshape([1, 3, 1, 1])

        self.build_reader()


if __name__ == '__main__':
    unittest.main()
