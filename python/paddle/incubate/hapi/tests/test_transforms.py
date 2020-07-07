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
import paddle.incubate.hapi.vision.transforms.functional as F


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

    def test_rotate(self):
        trans = transforms.Compose([
            transforms.RandomRotate(90),
            transforms.RandomRotate([-10, 10]),
            transforms.RandomRotate(
                45, expand=True),
            transforms.RandomRotate(
                10, expand=True, center=(60, 80)),
        ])
        self.do_transform(trans)

    def test_pad(self):
        trans = transforms.Compose([transforms.Pad(2)])
        self.do_transform(trans)

        fake_img = np.random.rand(200, 150, 3).astype('float32')
        trans_pad = transforms.Pad(10)
        fake_img_padded = trans_pad(fake_img)
        np.testing.assert_equal(fake_img_padded.shape, (220, 170, 3))
        trans_pad1 = transforms.Pad([1, 2])
        trans_pad2 = transforms.Pad([1, 2, 3, 4])
        img = trans_pad1(fake_img)
        img = trans_pad2(img)

    def test_erase(self):
        trans = transforms.Compose(
            [transforms.RandomErasing(), transforms.RandomErasing(value=0.0)])
        self.do_transform(trans)

    def test_random_crop(self):
        trans = transforms.Compose([
            transforms.RandomCrop(200),
            transforms.RandomCrop((140, 160)),
        ])
        self.do_transform(trans)

        trans_random_crop1 = transforms.RandomCrop(224)
        trans_random_crop2 = transforms.RandomCrop((140, 160))

        fake_img = np.random.rand(500, 400, 3).astype('float32')
        fake_img_crop1 = trans_random_crop1(fake_img)
        fake_img_crop2 = trans_random_crop2(fake_img_crop1)

        np.testing.assert_equal(fake_img_crop1.shape, (224, 224, 3))

        np.testing.assert_equal(fake_img_crop2.shape, (140, 160, 3))

        trans_random_crop_same = transforms.RandomCrop((140, 160))
        img = trans_random_crop_same(fake_img_crop2)

        trans_random_crop_bigger = transforms.RandomCrop((180, 200))
        img = trans_random_crop_bigger(img)

        trans_random_crop_pad = transforms.RandomCrop((224, 256), 2, True)
        img = trans_random_crop_pad(img)

    def test_grayscale(self):
        trans = transforms.Compose([transforms.Grayscale()])
        self.do_transform(trans)

        trans_gray = transforms.Grayscale()
        fake_img = np.random.rand(500, 400, 3).astype('float32')
        fake_img_gray = trans_gray(fake_img)

        np.testing.assert_equal(len(fake_img_gray.shape), 2)
        np.testing.assert_equal(fake_img_gray.shape[0], 500)
        np.testing.assert_equal(fake_img_gray.shape[1], 400)

        trans_gray3 = transforms.Grayscale(3)
        fake_img = np.random.rand(500, 400, 3).astype('float32')
        fake_img_gray = trans_gray3(fake_img)

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

        with self.assertRaises(ValueError):
            transforms.Pad([1.0, 2.0, 3.0])

        with self.assertRaises(TypeError):
            fake_img = np.random.rand(100, 120, 3).astype('float32')
            F.pad(fake_img, '1')

        with self.assertRaises(TypeError):
            fake_img = np.random.rand(100, 120, 3).astype('float32')
            F.pad(fake_img, 1, {})

        with self.assertRaises(TypeError):
            fake_img = np.random.rand(100, 120, 3).astype('float32')
            F.pad(fake_img, 1, padding_mode=-1)

        with self.assertRaises(ValueError):
            fake_img = np.random.rand(100, 120, 3).astype('float32')
            F.pad(fake_img, [1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            transforms.RandomRotate(-2)

        with self.assertRaises(ValueError):
            transforms.RandomRotate([1, 2, 3])

        with self.assertRaises(ValueError):
            trans_gray = transforms.Grayscale(5)
            fake_img = np.random.rand(100, 120, 3).astype('float32')
            trans_gray(fake_img)

    def test_info(self):
        str(transforms.Compose([transforms.Resize((224, 224))]))
        str(transforms.BatchCompose([transforms.Resize((224, 224))]))


if __name__ == '__main__':
    unittest.main()
