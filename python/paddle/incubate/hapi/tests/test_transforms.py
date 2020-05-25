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

# when test, you should add hapi root path to the PYTHONPATH,
# export PYTHONPATH=PATH_TO_HAPI:$PYTHONPATH
import unittest
import os
import tempfile
import cv2
import shutil
import numpy as np

from paddle.incubate.hapi.datasets import DatasetFolder
from paddle.incubate.hapi.vision.transforms import transforms


class TestTransforms(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()
        for i in range(2):
            sub_dir = os.path.join(self.data_dir, 'class_' + str(i))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for j in range(2):
                if j == 0:
                    fake_img = (np.random.random(
                        (280, 350, 3)) * 255).astype('uint8')
                else:
                    fake_img = (np.random.random(
                        (400, 300, 3)) * 255).astype('uint8')
                cv2.imwrite(os.path.join(sub_dir, str(j) + '.jpg'), fake_img)

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def do_transform(self, trans):
        dataset_folder = DatasetFolder(self.data_dir, transform=trans)

        for _ in dataset_folder:
            pass

    def test_trans_all(self):
        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.120, 57.375])
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224), transforms.GaussianNoise(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4,
                hue=0.4), transforms.RandomHorizontalFlip(),
            transforms.Permute(mode='CHW'), normalize
        ])

        self.do_transform(trans)

    def test_trans_resize(self):
        trans = transforms.Compose([
            transforms.Resize(300, [0, 1]),
            transforms.RandomResizedCrop((280, 280)),
            transforms.Resize(280, [0, 1]),
            transforms.Resize((256, 200)),
            transforms.Resize((180, 160)),
            transforms.CenterCrop(128),
            transforms.CenterCrop((128, 128)),
        ])
        self.do_transform(trans)

    def test_trans_centerCrop(self):
        trans = transforms.Compose([
            transforms.CenterCropResize(224),
            transforms.CenterCropResize(128, 160),
        ])
        self.do_transform(trans)

    def test_flip(self):
        trans = transforms.Compose([
            transforms.RandomHorizontalFlip(1.0),
            transforms.RandomHorizontalFlip(0.0),
            transforms.RandomVerticalFlip(0.0),
            transforms.RandomVerticalFlip(1.0),
        ])
        self.do_transform(trans)

    def test_color_jitter(self):
        trans = transforms.BatchCompose([
            transforms.BrightnessTransform(0.0),
            transforms.HueTransform(0.0),
            transforms.SaturationTransform(0.0),
            transforms.ContrastTransform(0.0),
        ])
        self.do_transform(trans)

    def test_exception(self):
        trans = transforms.Compose([transforms.Resize(-1)])

        trans_batch = transforms.BatchCompose([transforms.Resize(-1)])

        with self.assertRaises(Exception):
            self.do_transform(trans)

        with self.assertRaises(Exception):
            self.do_transform(trans_batch)

        with self.assertRaises(ValueError):
            transforms.ContrastTransform(-1.0)

        with self.assertRaises(ValueError):
            transforms.SaturationTransform(-1.0),

        with self.assertRaises(ValueError):
            transforms.HueTransform(-1.0)

        with self.assertRaises(ValueError):
            transforms.BrightnessTransform(-1.0)

    def test_info(self):
        str(transforms.Compose([transforms.Resize((224, 224))]))
        str(transforms.BatchCompose([transforms.Resize((224, 224))]))


if __name__ == '__main__':
    unittest.main()
