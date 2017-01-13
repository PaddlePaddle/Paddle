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
# Function for fetch the data untar directory for cifar10 training api.
# you can use this data for image classifation and gun traing.
# As the python can read the data in "cifar-10-python.tar.gz",herer is
# no need to untar the data.
#
#
# First,we let the data download path is "~/paddle_data_directory",
# when u no special the download path.
#
#
# Then,download the cifar10 dataset,and returns the data directory for
# training api.
#
########################################################################

import shutil
import os
import sys
import collections
import numpy as np
from six.moves import urllib
from http_download import data_download

source_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
source_file = "cifar-10-batches-py"


def fetch(directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch for training api.

    Args:

    Returns:
        path to untar file.
    """
    source_name = "cifar"

    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data_directory'))

    download_path = os.path.join(directory, source_name)
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    filepath = data_download(download_path, source_url)

    return filepath
