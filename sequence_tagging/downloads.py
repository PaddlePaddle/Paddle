# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
Download script, download dataset and pretrain models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import time
import hashlib
import tarfile
import requests

FILE_INFO = {
    'BASE_URL': 'https://baidu-nlp.bj.bcebos.com/',
    'DATA': {
        'name': 'lexical_analysis-dataset-2.0.0.tar.gz',
        'md5': '71e4a9a36d0f0177929a1bccedca7dba'
    },
    'LAC_MODEL': {
        'name': 'lexical_analysis-2.0.0.tar.gz',
        'md5': "fc1daef00de9564083c7dc7b600504ca"
    },
}


def usage():
    desc = ("\nDownload datasets and pretrained models for LAC.\n"
            "Usage:\n"
            "   1. python download.py all\n"
            "   2. python download.py dataset\n"
            "   3. python download.py lac\n")
    print(desc)


def md5file(fname):
    hash_md5 = hashlib.md5()
    with io.open(fname, "rb") as fin:
        for chunk in iter(lambda: fin.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract(fname, dir_path):
    """
    Extract tar.gz file
    """
    try:
        tar = tarfile.open(fname, "r:gz")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, dir_path)
            print(file_name)
        tar.close()
    except Exception as e:
        raise e


def _download(url, filename, md5sum):
    """
    Download file and check md5
    """
    retry = 0
    retry_limit = 3
    chunk_size = 4096
    while not (os.path.exists(filename) and md5file(filename) == md5sum):
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError(
                "Cannot download dataset ({0}) with retry {1} times.".format(
                    url, retry_limit))
        try:
            start = time.time()
            size = 0
            res = requests.get(url, stream=True)
            filesize = int(res.headers['content-length'])
            if res.status_code == 200:
                print("[Filesize]: %0.2f MB" % (filesize / 1024 / 1024))
                # save by chunk
                with io.open(filename, "wb") as fout:
                    for chunk in res.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fout.write(chunk)
                            size += len(chunk)
                            pr = '>' * int(size * 50 / filesize)
                            print(
                                '\r[Process ]: %s%.2f%%' %
                                (pr, float(size / filesize * 100)),
                                end='')
            end = time.time()
            print("\n[CostTime]: %.2f s" % (end - start))
        except Exception as e:
            print(e)


def download(name, dir_path):
    url = FILE_INFO['BASE_URL'] + FILE_INFO[name]['name']
    file_path = os.path.join(dir_path, FILE_INFO[name]['name'])

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # download data
    print("Downloading : %s" % name)
    _download(url, file_path, FILE_INFO[name]['md5'])

    # extract data
    print("Extracting : %s" % file_path)
    extract(file_path, dir_path)
    os.remove(file_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        usage()
        sys.exit(1)
    pwd = os.path.join(os.path.dirname(__file__), './')
    ernie_dir = os.path.join(os.path.dirname(__file__), './pretrained')

    if sys.argv[1] == 'all':
        download('DATA', pwd)
        download('LAC_MODEL', pwd)

    if sys.argv[1] == "dataset":
        download('DATA', pwd)

    elif sys.argv[1] == "lac":
        download('LAC_MODEL', pwd)

    else:
        usage()
