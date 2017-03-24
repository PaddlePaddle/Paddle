# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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
"""
CIFAR dataset: https://www.cs.toronto.edu/~kriz/cifar.html

TODO(yuyang18): Complete the comments.
"""

import cPickle
import itertools
import numpy
from common import download
import tarfile

__all__ = ['train100', 'test100', 'train10', 'test10']

URL_PREFIX = 'https://www.cs.toronto.edu/~kriz/'
CIFAR10_URL = URL_PREFIX + 'cifar-10-python.tar.gz'
CIFAR10_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_URL = URL_PREFIX + 'cifar-100-python.tar.gz'
CIFAR100_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'


def reader_creator(filename, sub_name):
    def read_batch(batch):
        data = batch['data']
        labels = batch.get('labels', batch.get('fine_labels', None))
        assert labels is not None
        for sample, label in itertools.izip(data, labels):
            yield (sample / 255.0).astype(numpy.float32), int(label)

    def reader():
        with tarfile.open(filename, mode='r') as f:
            names = (each_item.name for each_item in f
                     if sub_name in each_item.name)

            for name in names:
                batch = cPickle.load(f.extractfile(name))
                for item in read_batch(batch):
                    yield item

    return reader


def train100():
    return reader_creator(
        download(CIFAR100_URL, 'cifar', CIFAR100_MD5), 'train')


def test100():
    return reader_creator(download(CIFAR100_URL, 'cifar', CIFAR100_MD5), 'test')


def train10():
    return reader_creator(
        download(CIFAR10_URL, 'cifar', CIFAR10_MD5), 'data_batch')


def test10():
    return reader_creator(
        download(CIFAR10_URL, 'cifar', CIFAR10_MD5), 'test_batch')


def fetch():
    download(CIFAR10_URL, 'cifar', CIFAR10_MD5)
    download(CIFAR100_URL, 'cifar', CIFAR100_MD5)
