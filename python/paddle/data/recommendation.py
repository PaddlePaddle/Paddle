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


import shutil
import os
import sys
import zipfile
import collections
import numpy as np
from six.moves import urllib
import stat


source_url='http://files.grouplens.org/datasets/movielens/ml-1m.zip'
file_source = "mosesdecoder-master"


def fetch():
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch for training api.

    Args:

    Returns:
        path to downloaded file.
    """
    source_name = "recommendation"
    #Set the download dir for recommendation.
    data_home = set_data_path(source_name)
    filepath = data_download(data_home, source_url)
    """
    for i in range(1, num_batch + 1):
        fpath = os.path.join(filepath, "data_batch_%d" % i)
    """
    return filepath


def _unpickle(file_path):
    with open(file_path, mode='rb') as file:
        if sys.version_info < (3,):
            data = cPickle.load(file)
        else:
            data = cPickle.load(file, encoding='bytes')
    return data


def set_data_path(source_name):
    """
    Set the path for download according to the source name.

    Args:
        source_name:the source

    Returns:
        the data directory for data download.
    """
     data_base = os.path.expanduser(os.path.join('~',' .paddle'))
     if not os.access(data_base, os.W_OK):
         data_base = os.path.join('/tmp', '.paddle')
     datadir = os.path.join(data_base, source_name)
     print datadir
     if not os.path.exists(datadir):
         os.makedirs(datadir)
     return datadir


def data_download(download_dir, source_url):
    """
    Download data according to the url for mnist.
    when downloading,it can see each download process.

    Args:
        download_dir:the directory for data download.
        source_url:the url for data download.

    Returns:
        the path after data downloaded.
    """
    src_file = source_url.strip().split('/')[-1]
    file_path = os.path.join(download_dir, src_file)

    if not os.path.exists(file_path):
        temp_file_name,_ = download_with_urlretrieve(source_url)
        temp_file_path = os.getcwd()
        os.rename(temp_file_name, src_file)
        move_files(src_file, download_dir)
        print("Download finished, Extracting files.")
        tar = zipfile.ZipFile(file_path, 'r')
        infos = tar.infolist()
        for file in infos:
            tar.extract(file, download_dir)
            fpath = os.path.join(download_dir, file.filename)
        os.remove(file_path)
        print("Unpacking done!")
    else:
        tar = zipfile.ZipFile(file_path, 'r')
        infos = tar.infolist()
        for file in infos:
            tar.extract(file, download_dir)
            fpath = os.path.join(download_dir, file.filename)
        os.remove(file_path)
        print("Data has been already downloaded and unpacked!")
    return download_dir


def move_files(source_dire, target_dire):
    """
    Renaming the source file to other name.

    Args:
        source_dire:the source name of file
        target_dire:the target name of file.

    Returns:
    """
    shutil.move(source_dire, target_dire)


def download_with_urlretrieve(url, filename=None):
    """
    Download each file with urlretrieve,and the download process can be seen.

    Args:
        url:the url for data downoad.
        filename:the target name for download.

    Returns:
           the temp name after urlretrieve downloaded.
    """
    return urllib.request.urlretrieve(url, filename, reporthook=check_download_progress)


def check_download_progress(count, block_size, total_size):
    """
    Print and check the download process.

    Args:
        count:
        block_size:
        total_size:

    Returns:
    """
    percent = float(count * block_size) / total_size
    msg = "\r- Download progress: {:.1%}".format(percent)
    sys.stdout.write(msg)
    sys.stdout.flush()


if __name__ == '__main__':
    path = fetch()
    print path
