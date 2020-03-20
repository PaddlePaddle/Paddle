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

import logging
logger = logging.getLogger(__name__)

__all__ = [
    'get_weights_path'
]

WEIGHTS_HOME = osp.expanduser("~/.cache/paddle/hapi/weights")

DOWNLOAD_RETRY_LIMIT = 3


def get_weights_path(url):
    """Get weights path from WEIGHT_HOME, if not exists,
    download it from url.
    """
    path, _ = get_path(url, WEIGHTS_HOME)
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
    # parse path after download to decompress under root_dir
    fullpath = map_path(url, root_dir)

    exist_flag = False
    if osp.exists(fullpath) and check_exist:
        exist_flag = True
        logger.info("Found {}".format(fullpath))
    else:
        if int(os.getenv("PADDLE_TRAINER_ID", "0")) == 0:
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

    logger.info("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.info("File {} md5 check failed, {}(calc) != "
                    "{}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True
