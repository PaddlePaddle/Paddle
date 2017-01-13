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

############################################################################
#
# Function for fetch the data untar directory for mnist training api.
# you can use this data for Digital identification.
#
# First,we special the data download directory is "~/paddle_data_directory".
# For the mnist dataset,it untar the dataset,and returns the untar
# directory for training api.
#
############################################################################

import shutil
import os
import sys
import collections
import numpy as np
from six.moves import urllib
import urlparse
import gzip

source_url = 'http://yann.lecun.com/exdb/mnist/'
filename = [
    'train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz'
]


def fetch(directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch for training api.

    Args:

    Returns:
        path for untar file.
    """
    source_name = "mnist"

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data_directory'))

    download_path = os.path.join(directory, source_name)
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for file in filename:
        url = urlparse.urljoin(source_url, file)
        filepath = data_download(download_path, url)
        data_dir = os.path.join(filepath, file.split('.')[0])
        return data_dir
