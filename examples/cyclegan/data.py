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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np
from PIL import Image, ImageOps

import paddle

DATASET = "cityscapes"
A_LIST_FILE = "./data/" + DATASET + "/trainA.txt"
B_LIST_FILE = "./data/" + DATASET + "/trainB.txt"
A_TEST_LIST_FILE = "./data/" + DATASET + "/testA.txt"
B_TEST_LIST_FILE = "./data/" + DATASET + "/testB.txt"
IMAGES_ROOT = "./data/" + DATASET + "/"


class Cityscapes(paddle.io.Dataset):
    def __init__(self, root_path, file_path, mode='train', return_name=False):
        self.root_path = root_path
        self.file_path = file_path
        self.mode = mode
        self.return_name = return_name
        self.images = [root_path + l for l in open(file_path, 'r').readlines()]

    def _train(self, image):
        ## Resize
        image = image.resize((286, 286), Image.BICUBIC)
        ## RandomCrop
        i = np.random.randint(0, 30)
        j = np.random.randint(0, 30)
        image = image.crop((i, j, i + 256, j + 256))
        # RandomHorizontalFlip
        if np.random.rand() > 0.5:
            image = ImageOps.mirror(image)
        return image

    def __getitem__(self, idx):
        f = self.images[idx].strip("\n\r\t ")
        image = Image.open(f)
        if self.mode == 'train':
            image = self._train(image)
        else:
            image = image.resize((256, 256), Image.BICUBIC)
        # ToTensor
        image = np.array(image).transpose([2, 0, 1]).astype('float32')
        image = image / 255.0
        # Normalize, mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
        image = (image - 0.5) / 0.5
        if self.return_name:
            return [image], os.path.basename(f)
        else:
            return [image]

    def __len__(self):
        return len(self.images)


def DataA(root=IMAGES_ROOT, fpath=A_LIST_FILE):
    """
    Reader of images with A style for training.
    """
    return Cityscapes(root, fpath)


def DataB(root=IMAGES_ROOT, fpath=B_LIST_FILE):
    """
    Reader of images with B style for training.
    """
    return Cityscapes(root, fpath)


def TestDataA(root=IMAGES_ROOT, fpath=A_TEST_LIST_FILE):
    """
    Reader of images with A style for training.
    """
    return Cityscapes(root, fpath, mode='test', return_name=True)


def TestDataB(root=IMAGES_ROOT, fpath=B_TEST_LIST_FILE):
    """
    Reader of images with B style for training.
    """
    return Cityscapes(root, fpath, mode='test', return_name=True)


class ImagePool(object):
    def __init__(self, pool_size=50):
        self.pool = []
        self.count = 0
        self.pool_size = pool_size

    def get(self, image):
        if self.count < self.pool_size:
            self.pool.append(image)
            self.count += 1
            return image
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self.pool_size - 1)
                temp = self.pool[random_id]
                self.pool[random_id] = image
                return temp
            else:
                return image
