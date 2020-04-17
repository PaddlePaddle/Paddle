#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import shutil
import requests
import tqdm
import hashlib
import time
from collections import OrderedDict

from paddle.fluid.dygraph.parallel import ParallelEnv

import logging
logger = logging.getLogger(__name__)

__all__ = ['get_weights_path', 'is_url']

WEIGHTS_HOME = osp.expanduser("~/.cache/paddle/hapi/weights")

DOWNLOAD_RETRY_LIMIT = 3

nlp_models = OrderedDict(
            (('RoBERTa-zh-base', 'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz'),
            ('RoBERTa-zh-large', 'https://bert-models.bj.bcebos.com/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.tar.gz'),
            ('ERNIE-v2-en-base', 'https://ernie.bj.bcebos.com/ERNIE_Base_en_stable-2.0.0.tar.gz'),
            ('ERNIE-v2-en-large', 'https://ernie.bj.bcebos.com/ERNIE_Large_en_stable-2.0.0.tar.gz'),
            ('XLNet-cased-base', 'https://xlnet.bj.bcebos.com/xlnet_cased_L-12_H-768_A-12.tgz'),
            ('XLNet-cased-large', 'https://xlnet.bj.bcebos.com/xlnet_cased_L-24_H-1024_A-16.tgz'),
            ('ERNIE-v1-zh-base', 'https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz'),
            ('ERNIE-v1-zh-base-max-len-512', 'https://ernie.bj.bcebos.com/ERNIE_1.0_max-len-512.tar.gz'),
            ('BERT-en-uncased-large-whole-word-masking', 'https://bert-models.bj.bcebos.com/wwm_uncased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-en-cased-large-whole-word-masking', 'https://bert-models.bj.bcebos.com/wwm_cased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-en-uncased-base', 'https://bert-models.bj.bcebos.com/uncased_L-12_H-768_A-12.tar.gz'),
            ('BERT-en-uncased-large', 'https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-en-cased-base', 'https://bert-models.bj.bcebos.com/cased_L-12_H-768_A-12.tar.gz'),
            ('BERT-en-cased-large','https://bert-models.bj.bcebos.com/cased_L-24_H-1024_A-16.tar.gz'),
            ('BERT-multilingual-uncased-base', 'https://bert-models.bj.bcebos.com/multilingual_L-12_H-768_A-12.tar.gz'),
            ('BERT-multilingual-cased-base', 'https://bert-models.bj.bcebos.com/multi_cased_L-12_H-768_A-12.tar.gz'),
            ('BERT-zh-base', 'https://bert-models.bj.bcebos.com/chinese_L-12_H-768_A-12.tar.gz'),)
            )


def is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')


def get_weights_path(url, md5sum=None):
    """Get weights path from WEIGHT_HOME, if not exists,
    download it from url.
    """
    path, _ = get_path(url, WEIGHTS_HOME, md5sum)
    return path


def map_path(url, root_dir):
    # parse path after download under root_dir
    fname = osp.split(url)[-1]
    fpath = fname
    return osp.join(root_dir, fpath)


def get_path(url, root_dir, md5sum=None, check_exist=True):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.

    url (str): download url
    root_dir (str): root dir for downloading, it should be
                    WEIGHTS_HOME or DATASET_HOME
    md5sum (str): md5 sum of download package
    """
    assert is_url(url), "downloading from {} not a url".format(url)
    # parse path after download to decompress under root_dir
    fullpath = map_path(url, root_dir)

    exist_flag = False
    if osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        exist_flag = True
        if ParallelEnv().local_rank == 0:
            logger.info("Found {}".format(fullpath))
    else:
        if ParallelEnv().local_rank == 0:
            fullpath = _download(url, root_dir, md5sum)
        else:
            while not os.path.exists(fullpath):
                time.sleep(1)
    return fullpath, exist_flag


def _download(url, path, md5sum=None):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))
        if ParallelEnv().local_rank == 0:
            logger.info("Downloading {} from {}".format(fname, url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                for chunk in tqdm.tqdm(
                        req.iter_content(chunk_size=1024),
                        total=(int(total_size) + 1023) // 1024,
                        unit='KB'):
                    f.write(chunk)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

    return fullname


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True
    if ParallelEnv().local_rank == 0:
        logger.info("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        if ParallelEnv().local_rank == 0:
            logger.info("File {} md5 check failed, {}(calc) != "
                        "{}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True
