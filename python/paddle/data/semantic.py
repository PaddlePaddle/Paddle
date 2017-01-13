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
# Function for fetch the data untar directory for semantic_role_labeling
# training api.you can use this data for semantic.
#
# First,we special the data download directory is "~/paddle_data_directory".
# For the semantic role labeling,it untar the dataset,and returns the untar
# directory for training api.
#
############################################################################

import shutil
import os
import sys
import zipfile
import collections
import numpy as np
from six.moves import urllib
import stat

source_url = [
    'http://www.cs.upc.edu/~srlconll/conll05st-tests.tar.gz',
    'http://paddlepaddle.bj.bcebos.com/demo/srl_dict_and_embedding/verbDict.txt',
    'http://paddlepaddle.bj.bcebos.com/demo/srl_dict_and_embedding/targetDict.txt',
    'http://paddlepaddle.bj.bcebos.com/demo/srl_dict_and_embedding/wordDict.txt',
    'http://paddlepaddle.bj.bcebos.com/demo/srl_dict_and_embedding/emb'
]

file_source = "conll05st-release"


def fetch(directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch for training api.

    Args:

    Returns:
        path to downloaded file.
    """
    source_name = "semantic"
    if directory is None:
        directory = os.path.expanduser(
            os.path.join('~', 'paddle_data_directory'))

    download_path = os.path.join(directory, source_name)
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    for url in source_url:
        file_name = url.split('/')[-1]
        if 'gz' in file_name:
            filepath = data_download(download_path, url)
            data_path = os.path.join(filepath, file_source)

            sub_file = ['est.wsj.words.gz', 'test.wsj.props.gz']
            words_path = os.path.join(data_path,
                                      "test.wsj/words/test.wsj.words.gz")
            props_path = os.path.join(data_path,
                                      "test.wsj/props/test.wsj.props.gz")

            sub_path = [words_path, props_path]
            for sub_file in sub_path:
                new_sub_path = os.path.join(download_path, sub_file)
                shutil.move(sub_path, new_subpath)
                tarfile.open(
                    name=new_subpath, mode="r:gz").extractall(download_path)
                os.remove(new_subpath)
        else:
            filepath = data_download(download_path, url)

    return filepath
