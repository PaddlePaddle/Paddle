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

from __future__ import print_function

import os
import paddle.dataset.common


def _check_exists_and_download(path, url, md5, module_name, download=True):
    if path and os.path.exists(path):
        return path

    if download:
        return paddle.dataset.common.download(url, module_name, md5)
    else:
        raise ValueError('{} not exists and auto download disabled'.format(
            path))
