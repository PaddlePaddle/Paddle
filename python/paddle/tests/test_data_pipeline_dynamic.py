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

import test_data_pipeline_static
from test_data_pipeline_static import DATASET_HOME, DATASET_URL, \
                                        DATASET_MD5, IMAGE_NUM

DATASET_HOME = os.path.expanduser("~/.cache/paddle/datasets")
DATASET_URL = "https://paddlemodels.cdn.bcebos.com/ImageNet_stub.tar"
DATASET_MD5 = "c7110519124a433901cf005a4a91b607"
IMAGE_NUM = 100


class TestDataPipelineDynamicCase1(
        test_data_pipeline_static.TestDataPipelineStaticCase1):
    def test_output(self):
        # NOTE: only supoort CUDA kernel currently
        if not core.is_compiled_with_cuda():
            return

        data = self.reader()

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


class TestDataPipelineDynamicCase2(TestDataPipelineDynamicCase1):
    def setUp(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)

        self.num_epoches = 1
        self.batch_size = 16
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
