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

import unittest
import os
import numpy as np
import tempfile
import shutil
import cv2

from paddle.incubate.hapi.datasets import *
from paddle.incubate.hapi.datasets.utils import _check_exists_and_download


class TestFolderDatasets(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()
        self.empty_dir = tempfile.mkdtemp()
        for i in range(2):
            sub_dir = os.path.join(self.data_dir, 'class_' + str(i))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for j in range(2):
                fake_img = (np.random.random((32, 32, 3)) * 255).astype('uint8')
                cv2.imwrite(os.path.join(sub_dir, str(j) + '.jpg'), fake_img)

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def test_dataset(self):
        dataset_folder = DatasetFolder(self.data_dir)

        for _ in dataset_folder:
            pass

        assert len(dataset_folder) == 4
        assert len(dataset_folder.classes) == 2

        dataset_folder = DatasetFolder(self.data_dir)
        for _ in dataset_folder:
            pass

    def test_folder(self):
        loader = ImageFolder(self.data_dir)

        for _ in loader:
            pass

        loader = ImageFolder(self.data_dir)
        for _ in loader:
            pass

        assert len(loader) == 4

    def test_transform(self):
        def fake_transform(img):
            return img

        transfrom = fake_transform
        dataset_folder = DatasetFolder(self.data_dir, transform=transfrom)

        for _ in dataset_folder:
            pass

        loader = ImageFolder(self.data_dir, transform=transfrom)
        for _ in loader:
            pass

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            ImageFolder(self.empty_dir)
        with self.assertRaises(RuntimeError):
            DatasetFolder(self.empty_dir)

        with self.assertRaises(ValueError):
            _check_exists_and_download('temp_paddle', None, None, None, False)


class TestMNISTTest(unittest.TestCase):
    def test_main(self):
        mnist = MNIST(mode='test')
        self.assertTrue(len(mnist) == 10000)

        for i in range(len(mnist)):
            image, label = mnist[i]
            self.assertTrue(image.shape[0] == 1)
            self.assertTrue(image.shape[1] == 28)
            self.assertTrue(image.shape[2] == 28)
            self.assertTrue(label.shape[0] == 1)
            self.assertTrue(0 <= int(label) <= 9)


class TestMNISTTrain(unittest.TestCase):
    def test_main(self):
        mnist = MNIST(mode='train', chw_format=False)
        self.assertTrue(len(mnist) == 60000)

        for i in range(len(mnist)):
            image, label = mnist[i]
            self.assertTrue(image.shape[0] == 784)
            self.assertTrue(label.shape[0] == 1)
            self.assertTrue(0 <= int(label) <= 9)


class TestFlowersTrain(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='train')
        self.assertTrue(len(flowers) == 6149)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 6149)
        image, label = flowers[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


class TestFlowersValid(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='valid')
        self.assertTrue(len(flowers) == 1020)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1020)
        image, label = flowers[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


class TestFlowersTest(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='test')
        self.assertTrue(len(flowers) == 1020)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1020)
        image, label = flowers[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


if __name__ == '__main__':
    unittest.main()
