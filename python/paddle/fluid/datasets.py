#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import gzip
import struct
import numpy as np

import paddle.dataset.common

__all__ = ["Dataset", "MnistDataset"]


class Dataset(object):
    """
    An abstract class to encapsulates methods and behaviors of datasets.

    All datasets in map-style(dataset samples can be get by a given key)
    should be a subclass of `fluid.io.Dataset`. All subclasses should
    implement following methods:

    :math:`__getitem__`: get sample from dataset with a given index. This
    method is required by reading dataset sample in `fluid.io.DataLoader`
    subprocesses.
    :math:`__len__`: return dataset sample number. This method is required
    by some implements of `fluid.io.BatchSampler`
    """

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))


def check_exists_and_download(path, url, module_name, md5, download=True):
    if path and os.path.exists(path):
        return path

    if download:
        return paddle.dataset.common.download(url, module_name, md5)
    else:
        raise FileNotFoundError(
            '{} not exists and auto download disabled'.format(path))


class MnistDataset(Dataset):
    """
    Implement of mnist dataset

    Args:
        root(str): mnist dataset root path.
        mode(str): 'train' or 'test' mode, default 'train'.
        download(bool): whether auto download mnist dataset if dataset
                        not exists in `root`, default True
    """

    URL_PREFIX = 'https://dataset.bj.bcebos.com/mnist/'
    TEST_IMAGE_URL = URL_PREFIX + 't10k-images-idx3-ubyte.gz'
    TEST_IMAGE_MD5 = '9fb629c4189551a2d022fa330f9573f3'
    TEST_LABEL_URL = URL_PREFIX + 't10k-labels-idx1-ubyte.gz'
    TEST_LABEL_MD5 = 'ec29112dd5afa0611ce80d1b7f02629c'
    TRAIN_IMAGE_URL = URL_PREFIX + 'train-images-idx3-ubyte.gz'
    TRAIN_IMAGE_MD5 = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
    TRAIN_LABEL_URL = URL_PREFIX + 'train-labels-idx1-ubyte.gz'
    TRAIN_LABEL_MD5 = 'd53e105ee54ea40749a09fcbcd1e9432'

    def __init__(self,
                 image_filename=None,
                 label_filename=None,
                 mode='train',
                 download=True):
        assert mode.lower() in ['train', 'test'], \
                "mode should be 'train' or 'test, but got {}'".format(mode)
        self.mode = mode.lower()

        image_url = self.TRAIN_IMAGE_URL if mode == 'train' else self.TEST_IMAGE_URL
        image_md5 = self.TRAIN_IMAGE_MD5 if mode == 'train' else self.TEST_IMAGE_MD5
        self.image_filename = check_exists_and_download(
            image_filename, image_url, 'mnist', image_md5, download)

        label_url = self.TRAIN_IMAGE_URL if mode == 'train' else self.TEST_IMAGE_URL
        label_md5 = self.TRAIN_IMAGE_MD5 if mode == 'train' else self.TEST_IMAGE_MD5
        self.label_filename = check_exists_and_download(
            label_filename, label_url, 'mnist', label_md5, download)

        # read dataset into memory
        self._parse_dataset()

    def _parse_dataset(self, buffer_size=100):
        self.images = []
        self.labels = []
        with gzip.GzipFile(self.image_filename, 'rb') as image_file:
            img_buf = image_file.read()
            with gzip.GzipFile(self.label_filename, 'rb') as label_file:
                lab_buf = label_file.read()

                step_label = 0

                offset_img = 0
                # read from Big-endian
                # get file info from magic byte
                # image file : 16B
                magic_byte_img = '>IIII'
                magic_img, image_num, rows, cols = struct.unpack_from(
                    magic_byte_img, img_buf, offset_img)
                offset_img += struct.calcsize(magic_byte_img)

                offset_lab = 0
                # label file : 8B
                magic_byte_lab = '>II'
                magic_lab, label_num = struct.unpack_from(magic_byte_lab,
                                                          lab_buf, offset_lab)
                offset_lab += struct.calcsize(magic_byte_lab)

                while True:
                    if step_label >= label_num:
                        break
                    fmt_label = '>' + str(buffer_size) + 'B'
                    labels = struct.unpack_from(fmt_label, lab_buf, offset_lab)
                    offset_lab += struct.calcsize(fmt_label)
                    step_label += buffer_size

                    fmt_images = '>' + str(buffer_size * rows * cols) + 'B'
                    images_temp = struct.unpack_from(fmt_images, img_buf,
                                                     offset_img)
                    images = np.reshape(images_temp, (buffer_size, rows *
                                                      cols)).astype('float32')
                    offset_img += struct.calcsize(fmt_images)

                    images = images / 255.0
                    images = images * 2.0
                    images = images - 1.0

                    for i in range(buffer_size):
                        self.images.append(images[i, :])
                        self.labels.append(int(labels[i]))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
