#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import os
import unittest
import numpy as np

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle.utils.download import get_path_from_url
from paddle.vision.datasets import DatasetFolder
from paddle.vision.reader import _sampler_manager, file_label_loader


DATASET_HOME = os.path.expanduser("~/.cache/paddle/datasets")
DATASET_URL = "https://paddlemodels.cdn.bcebos.com/ImageNet_stub.tar"
DATASET_MD5 = "c7110519124a433901cf005a4a91b607"

class TestFileLabelLoaderStatic(unittest.TestCase):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)
        self.batch_size = 16
        self.shuffle = False
        self.drop_last = False
        self.dynamic = False

        if not self.dynamic:
            self.build_program()

    def build_program(self):
        paddle.enable_static()
        self.indices_data = paddle.static.data(
                    shape=[self.batch_size], dtype='int64', name='indices')
        self.sample_data, self.label_data = file_label_loader(self.data_root, self.indices_data, self.batch_size)
        self.exe = paddle.static.Executor(paddle.CPUPlace())
        paddle.disable_static()

    def loader_function(self, indices):
        if paddle.in_dynamic_mode():
            indices = paddle.to_tensor(indices)
            return file_label_loader(self.data_root, indices, self.batch_size)
        else:
            paddle.enable_static()
            return self.exe.run(paddle.static.default_main_program(),
                                feed={'indices': indices},
                                fetch_list=[self.sample_data,
                                            self.label_data])

    def test_check_output(self):
        self.setup()

        data_folder = DatasetFolder(self.data_root)
        samples = [s[0] for s in data_folder.samples]
        targets = [s[1] for s in data_folder.samples]

        sampler_id = fluid.layers.utils._hash_with_id(
                            self.data_root, self.batch_size,
                            self.shuffle, self.drop_last,
                            self.dynamic)
        sampler = _sampler_manager.get(sampler_id,
                                       batch_size=self.batch_size,
                                       num_samples=len(samples),
                                       shuffle=self.shuffle,
                                       drop_last=self.drop_last)

        num_iters = (len(samples) + self.batch_size - 1) // self.batch_size
        for _ in range(num_iters):
            indices = next(sampler)
            sample, target = self.loader_function(indices)
            assert np.array_equal(target, np.array(targets)[indices])


class TestFileLabelLoaderDynamic(TestFileLabelLoaderStatic):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)
        self.batch_size = 16
        self.shuffle = False
        self.drop_last = False
        self.dynamic = True 

        if not self.dynamic:
            self.build_program()


class TestFileLabelLoaderStaticShuffle(TestFileLabelLoaderStatic):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)
        self.batch_size = 16
        self.shuffle = True 
        self.drop_last = False
        self.dynamic = False 

        if not self.dynamic:
            self.build_program()


class TestFileLabelLoaderDynamicShuffle(TestFileLabelLoaderStatic):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)
        self.batch_size = 16
        self.shuffle = True 
        self.drop_last = False
        self.dynamic = True 

        if not self.dynamic:
            self.build_program()


class TestFileLabelLoaderStaticDropLast(TestFileLabelLoaderStatic):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)
        self.batch_size = 16
        self.shuffle = True 
        self.drop_last = True 
        self.dynamic = False

        if not self.dynamic:
            self.build_program()


class TestFileLabelLoaderDynamicDropLast(TestFileLabelLoaderStatic):
    def setup(self):
        self.data_root = get_path_from_url(DATASET_URL, DATASET_HOME,
                                           DATASET_MD5)
        self.batch_size = 16
        self.shuffle = True 
        self.drop_last = True 
        self.dynamic = True

        if not self.dynamic:
            self.build_program()


if __name__ == '__main__':
    unittest.main()
