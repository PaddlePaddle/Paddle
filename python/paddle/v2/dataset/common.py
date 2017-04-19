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

__all__ = ['DATA_HOME', 'download', 'md5file']

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
