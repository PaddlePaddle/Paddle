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

########################################################################
#
# Function for fetch the data untar directory for amazon training api.
# As the python can read the data in "reviews_Electronics_5.json.gz",
#here is no need to untar the data.
#
#
# First,we let the data download path is "~/paddle_data_directory"
# when u no special the download path.
#
#
# Then,download the data,according to the speical source url.
# Here,no need to untar the "reviews_Electronics_5.json.gz".
#
# After download the data,return the path of data.
#
#
#########################################################################


import shutil
import os
import sys
import zipfile
import collections
import stat
from six.moves import urllib
from http_download import data_download


source_url='http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz'
moses_url='https://github.com/moses-smt/mosesdecoder/archive/master.zip'

mose_source = "mosesdecoder-master"


def fetch(directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch for training api.

    Args:

    Returns:
        path for the data untar.
    """
    source_name = "amazon"
    if directory is None:
        directory = os.path.expanduser(os.path.join('~', 'paddle_data_directory'))

    download_path = os.path.join(directory, source_name)
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    moses_src = data_download(download_path, moses_url)
    moses_path = os.path.join(moses_src, mose_source)

    filepath = data_download(download_path, source_url)
    return filepath


