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
A utility for fetching, reading mnist handwritten digit dataset.

http://yann.lecun.com/exdb/mnist/
"""

import os
from http_download import download
from logger import logger
from base import BaseDataSet
import gzip
import json
import hashlib
import nltk
import collections
import h5py
import numpy

BASE_URL = 'http://yann.lecun.com/exdb/mnist/%s-ubyte.gz'


class Categories(object):
    TrainImage = 'train-images-idx3'
    TrainLabels = 'train-labels-idx1'
    TestImage = 't10k-images-idx3'
    TestLabels = 't10k-labels-idx1'

    All = [TrainImage, TrainLabels, TestImage, TestLabels]

    __md5__ = dict()

    __md5__[TrainImage] = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
    __md5__[TrainLabels] = 'd53e105ee54ea40749a09fcbcd1e9432'
    __md5__[TestImage] = '9fb629c4189551a2d022fa330f9573f3'
    __md5__[TestLabels] = 'ec29112dd5afa0611ce80d1b7f02629c'


__all__ = ['fetch', 'Categories']


def calculate_md5(fn):
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_data(category=None, directory=None):
    """
    Calculate each md5 value.
    :param category:
    :param directory:
    :return:
    """
    cn = category + '-ubyte'
    fn = os.path.join(directory, '%s.gz' % cn)
    if os.path.exists(fn) and \
                    calculate_md5(fn) == Categories.__md5__[category]:
        return fn
    logger.info("Downloading mnist handwritten digit dataset for %s category" % cn)
    return download(BASE_URL % category, fn)


def fetch(category=None, directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch
    for training api.
    :param category:
    :param directory:
    :return:
    """
    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data', 'mnist'))

    if not os.path.exists(directory):
        os.makedirs(directory)

    if category is None:
        category = [category for category in Categories.All]
        fl = []  # download file list
        for index, line in range(len(category)):
            fl.append(fetch_data(line, directory))
        return fl
