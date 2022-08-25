# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from typing import Dict
from typing import List

from paddle.framework import load as load_state_dict
from paddle.utils import download

from .log import logger

download.logger = logger

__all__ = [
    'decompress',
    'download_and_decompress',
    'load_state_dict_from_url',
]


def decompress(file: str):
    """
    Extracts all files from a compressed file.
    """
    assert os.path.isfile(file), "File: {} not exists.".format(file)
    download._decompress(file)


def download_and_decompress(archives: List[Dict[str, str]],
                            path: str,
                            decompress: bool=True):
    """
    Download archieves and decompress to specific path.
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    for archive in archives:
        assert 'url' in archive and 'md5' in archive, \
            'Dictionary keys of "url" and "md5" are required in the archive, but got: {list(archieve.keys())}'
        download.get_path_from_url(
            archive['url'], path, archive['md5'], decompress=decompress)


def load_state_dict_from_url(url: str, path: str, md5: str=None):
    """
    Download and load a state dict from url
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    download.get_path_from_url(url, path, md5)
    return load_state_dict(os.path.join(path, os.path.basename(url)))
