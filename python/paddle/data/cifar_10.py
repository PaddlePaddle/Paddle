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
import tarfile
import zipfile
import collections
import numpy as np
from six.moves import urllib

source_url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
source_file = "cifar-10-batches-py"
label_map = {
0: "airplane",
1: "automobile",
2: "bird",
3: "cat",
4: "deer",
5: "dog",
6: "frog",
7: "horse",
8: "ship",
9: "truck"
}

def fetch():
    num_images_train = 50000
    num_batch = 5
    source_name = "cifar"
    file_source = "cifar-10-batches-py"
    #Set the download dir for cifar.
    data_home = set_data_path(source_name)
    filepath = data_download(data_home, source_url)
    """
    for i in range(1, num_batch + 1):
        fpath = os.path.join(filepath, "data_batch_%d" % i)
    """

def _unpickle(file_path):
    with open(file_path, mode='rb') as file:
        if sys.version_info < (3,):
            data = cPickle.load(file)
        else:
            data = cPickle.load(file, encoding='bytes')
    return data

def set_data_path(source_name):
     data_base = os.path.expanduser(os.path.join('~','.paddle'))
     print data_base
     if not os.access(data_base, os.W_OK):
         data_base = os.path.join('/tmp', '.paddle')
     datadir = os.path.join(data_base, source_name)
     print datadir
     if not os.path.exists(datadir):
         os.makedirs(datadir)
     return datadir

def data_download(download_dir,source_url):
    src_file = source_url.strip().split('/')[-1]
    file_path = os.path.join(download_dir,src_file)
    if not os.path.exists(file_path):
        temp_file_name,_ = download_with_urlretrieve(source_url)
        temp_file_path = os.getcwd()
        os.rename(temp_file_name,src_file)
        move_files(src_file,download_dir)
        print("Download finished,Extracting files.")
        tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
        print("Unpacking done!")
    else:
        tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)
        print("Data has been already downloaded and unpacked!")
    return download_dir

def move_files(source_dire,target_dire):
    shutil.move(source_dire,target_dire)

def download_with_urlretrieve(url, filename=None):
    return urllib.request.urlretrieve(url, filename)


if __name__ == '__main__':
    path = fetch()
    print path
