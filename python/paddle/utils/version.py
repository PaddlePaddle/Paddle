# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
compare two versions.
"""

import re
import six


def compare_version(v1, v2):
    """Compare two versions, version format is "\\d+(\\.\\d+){0,3}".
       Args:
           v1(str): The first version to compare.
           v2(str): The second version to compare.
       Returns:
           int: 0: v1 == v2; -1: v1 < v2; 1: v1 > v2
    """
    assert isinstance(
        v1,
        str), "version type must be str, but v1 type got {}".format(type(v1))
    assert isinstance(
        v2,
        str), "version type must be str, but v2 type got {}".format(type(v2))

    match = re.match(r'\d+(\.\d+){0,3}', v1)
    assert match is not None and match.group(
    ) == v1, 'version format should be "\\d+(\\.\\d+){0,3}", like "1.5.2.0", but v1 got "{}"'.format(
        v1)
    match = re.match(r'\d+(\.\d+){0,3}', v2)
    assert match is not None and match.group(
    ) == v2, 'version format should be "\\d+(\\.\\d+){0,3}", like "1.5.2.0", but v2 got "{}"'.format(
        v2)

    zero_version = ['0', '0', '0', '0']
    v1_split = v1.split('.')
    _v1 = v1_split + zero_version[len(v1_split):]
    v2_split = v2.split('.')
    _v2 = v2_split + zero_version[len(v2_split):]

    for i in six.moves.range(len(_v1)):
        if int(_v1[i]) > int(_v2[i]):
            return 1
        elif int(_v1[i]) < int(_v2[i]):
            return -1
    return 0
