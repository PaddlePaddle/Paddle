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
import sys
import site
import unittest
import numpy as np


class TestCustomDeviceDataLoader(unittest.TestCase):

    def setUp(self):
        # compile so and set to current path
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        # --inplace to place output so file to current dir
        cmd = 'cd {} && {} custom_cpu_setup.py build_ext --inplace'.format(
            cur_dir, sys.executable)
        os.system(cmd)

        # set environment for loading and registering compiled custom kernels
        # only valid in current process
        os.environ['CUSTOM_DEVICE_ROOT'] = cur_dir

    def test_custom_device_dataloader(self):
        import paddle

        paddle.set_device('custom_cpu')
        dataset = paddle.vision.datasets.MNIST(
            mode='test',
            transform=paddle.vision.transforms.Compose([
                paddle.vision.transforms.CenterCrop(20),
                paddle.vision.transforms.RandomResizedCrop(14),
                paddle.vision.transforms.Normalize(),
                paddle.vision.transforms.ToTensor()
            ]))
        loader = paddle.io.DataLoader(dataset,
                                      batch_size=32,
                                      num_workers=1,
                                      shuffle=True)
        for image, label in loader:
            self.assertTrue(image.place.is_custom_place())
            self.assertTrue(label.place.is_custom_place())
            break

    def tearDown(self):
        del os.environ['CUSTOM_DEVICE_ROOT']


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        exit()
    unittest.main()
