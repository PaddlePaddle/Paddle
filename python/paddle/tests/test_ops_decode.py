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
from paddle.vision.ops import image_decode, image_decode_random_crop 
from paddle.vision.reader import file_label_loader


DATASET_HOME = os.path.expanduser("~/.cache/paddle/datasets")
DATASET_URL = "https://paddlemodels.cdn.bcebos.com/ImageNet_stub.tar"
DATASET_MD5 = "c7110519124a433901cf005a4a91b607"


class TestImageReaderDecodeCase1(unittest.TestCase):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)

        self.batch_size = 16
        self.num_threads = 2
        self.host_memory_padding = 1000000
        self.device_memory_padding = 1000000

    def test_static_output(self):
        paddle.enable_static()
        self.setup()

        indices = paddle.arange(self.batch_size)
        image, label = file_label_loader(self.data_root, indices)
        image = image_decode(image,
                             num_threads=self.num_threads)
        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        out_image, out_label = exe.run(paddle.static.default_main_program(),
                                       fetch_list=[image, label])

        assert len(out_image) == self.batch_size
        for i in range(self.batch_size):
            img = np.array(out_image[i])
            assert img.dtype == np.uint8
            assert img.shape[2] == 3
            assert np.all(img >= 0)
            assert np.all(img <= 255)

        assert len(out_label) == self.batch_size
        assert label.dtype == paddle.int64
        label = np.array(out_label)
        assert np.all(label >= 0)
        assert np.all(label <= 1)


class TestImageReaderDecodeCase2(TestImageReaderDecodeCase1):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)

        self.batch_size = 32
        self.num_threads = 4
        self.host_memory_padding = 0
        self.device_memory_padding = 0


class TestImageReaderDecodeRandomCropNCHW(unittest.TestCase):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)

        self.batch_size = 16
        self.num_threads = 2
        self.host_memory_padding = 1000000
        self.device_memory_padding = 1000000

        self.aspect_ratio_min = 3. / 4.
        self.aspect_ratio_max = 4. / 3.
        self.area_min = 0.08
        self.area_max = 1.0
        self.num_attempts = 10

        self.data_format = "NCHW"
        self.channel_dim = 0

    def test_static_output(self):
        paddle.enable_static()
        self.setup()

        indices = paddle.arange(self.batch_size)
        image, label = file_label_loader(self.data_root, indices)
        image = image_decode_random_crop(image,
                            num_threads=self.num_threads,
                            aspect_ratio_min=self.aspect_ratio_min,
                            aspect_ratio_max=self.aspect_ratio_max,
                            area_min=self.area_min,
                            area_max=self.area_max,
                            num_attempts=self.num_attempts,
                            data_format=self.data_format)
        exe = paddle.static.Executor(paddle.CUDAPlace(0))
        out_image, out_label = exe.run(paddle.static.default_main_program(),
                                       fetch_list=[image, label])

        assert len(out_image) == self.batch_size
        for i in range(self.batch_size):
            img = np.array(out_image[i])
            assert img.dtype == np.uint8
            assert img.shape[self.channel_dim] == 3
            assert np.all(img >= 0)
            assert np.all(img <= 255)

        assert len(out_label) == self.batch_size
        assert label.dtype == paddle.int64
        label = np.array(out_label)
        assert np.all(label >= 0)
        assert np.all(label <= 1)


class TestImageReaderDecodeRandomCropNHWC(TestImageReaderDecodeRandomCropNCHW):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)

        self.batch_size = 16
        self.num_threads = 4
        self.host_memory_padding = 0
        self.device_memory_padding = 0

        self.aspect_ratio_min = 4. / 5.
        self.aspect_ratio_max = 5. / 4.
        self.area_min = 0.1
        self.area_max = 0.9
        self.num_attempts = 20

        self.data_format = "NHWC"
        self.channel_dim = 2


if __name__ == '__main__':
    unittest.main()
