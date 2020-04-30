# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

import os
import os.path as osp
import sys
import tarfile

from hapi.download import _download

import logging
logger = logging.getLogger(__name__)

DATASETS = {
    'voc': [
        ('https://paddlemodels.bj.bcebos.com/hapi/voc.tar',
         '9faeb7fd997aeea843092fd608d5bcb4', ),
    ],
}

def download_decompress_file(data_dir, url, md5):
    logger.info("Downloading from {}".format(url))
    tar_file = _download(url, data_dir, md5)
    logger.info("Decompressing {}".format(tar_file))
    with tarfile.open(tar_file) as tf:
        tf.extractall(path=data_dir)
    os.remove(tar_file)


if __name__ == "__main__":
    data_dir = osp.split(osp.realpath(sys.argv[0]))[0]
    for name, infos in DATASETS.items():
        for info in infos:
            download_decompress_file(data_dir, *info)

