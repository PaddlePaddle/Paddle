#/usr/bin/env python
# -*- coding:utf-8 -*-

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
A utility for fetching, reading CIFAR-10 dataset.

https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
from http_download import download
from logger import logger
import hashlib

BASE_URL = 'https://www.cs.toronto.edu/~kriz/cifar-%s-python.tar.gz'


class Categories(object):
    Ten = 10
    Hundred = 100

    __md5__ = dict()

    __md5__[Ten] = 'c58f30108f718f92721af3b95e74349a'
    __md5__[Hundred] = 'eb9058c3a382ffc7106e4002c42a8d85'

__all__ = ['fetch', 'Categories']


def calculate_md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(category=None, directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch for training api.

    Args:

    Returns:
        path to untar file.
    """
    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data_directory', 'cifar'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    cn = 'cifar' + category
    fn = os.path.join(directory, '%s.tar.gz' % cn)

    if os.path.exists(fn) and calculate_md5(fn) == Categories.__md5__[category]:
        return fn

    logger.info("Downloading cifar dataset for %s category" % cn)
    return download(BASE_URL % category,
                    os.path.join(directory, '%s.tar.gz' % cn))
