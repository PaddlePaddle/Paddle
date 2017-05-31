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

import requests
import hashlib
import os
import shutil
import sys
import importlib
import paddle.v2.dataset
import pickle
import glob

__all__ = ['DATA_HOME', 'download', 'md5file', 'split', 'cluster_files_reader']

DATA_HOME = os.path.expanduser('~/.cache/paddle/dataset')

if not os.path.exists(DATA_HOME):
    os.makedirs(DATA_HOME)


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, module_name, md5sum):
    dirname = os.path.join(DATA_HOME, module_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(dirname, url.split('/')[-1])
    if not (os.path.exists(filename) and md5file(filename) == md5sum):
        print "Cache file %s not found, downloading %s" % (filename, url)
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(filename, 'w') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(filename, 'w') as f:
                dl = 0
                total_length = int(total_length)
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done,
                                                   ' ' * (50 - done)))
                    sys.stdout.flush()

    return filename


def fetch_all():
    for module_name in filter(lambda x: not x.startswith("__"),
                              dir(paddle.v2.dataset)):
        if "fetch" in dir(
                importlib.import_module("paddle.v2.dataset.%s" % module_name)):
            getattr(
                importlib.import_module("paddle.v2.dataset.%s" % module_name),
                "fetch")()

def split(reader, line_count, suffix="%05d.pickle"):
    """
    you can call the function as:

    split(paddle.v2.dataset.cifar.train10(), line_count=1000,
        suffix="imikolov-train-%05d.pickle")

    the output files as:

    |-imikolov-train-00000.pickle
    |-imikolov-train-00001.pickle
    |- ...
    |-imikolov-train-00480.pickle

    :param reader: the reader creator will be split
    :param line_count: line count for each file
    :param suffix: the suffix for each file,
                    contain "%d" means the id for each file
    """
    lines = []
    indx_f = 0
    for i, d in enumerate(reader()):
        lines.append(d)
        if i >= line_count and i % line_count == 0:
            with open(suffix % indx_f, "w") as f:
                pickle.dump(lines, f)
                lines = []
                indx_f += 1
    if not lines:
        with open(suffix % indx_f, "w") as f:
            pickle.dump(lines, f)

def cluster_files_reader(files_pattern, trainers, trainer_id):
    """
    Create a reader that yield element from the given files, select
    a file set according trainer count and trainer_id

    :param files_pattern: the files which generating by split(...)
    :param trainers: total trainer count
    :param trainer_id: the trainer rank id
    """
    def reader():
        file_list = glob.glob(files_pattern)
        file_list.sort()
        my_file_list = []
        for idx, f in enumerate(file_list):
            if idx % trainers == trainer_id:
                print "append file: %s" % f
                my_file_list.append(f)
        for fn in my_file_list:
            with open(fn, "r") as f:
                lines = pickle.load(f)
                for line in lines:
                    yield line
    return reader
