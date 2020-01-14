# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import collections
import paddle.compat as cpt

ERROR_LOG_HOME = os.path.expanduser('~/.cache/paddle')


def count_error_frequency(dirname):
    if not os.path.exists(dirname):
        return

    error_freq_dict = collections.OrderedDict()
    for filename in os.listdir(dirname):
        filepath = os.path.join(dirname, filename)
        if not os.path.isfile(filepath):
            continue
        print(filepath)
        with open(filepath, 'rb') as f:
            for line in f.readlines():
                if line.startswith(b'FileLine:'):
                    key_str = line[10:]
                    if key_str not in error_freq_dict:
                        error_freq_dict[key_str] = 1
                    else:
                        error_freq_dict[key_str] += 1
    return error_freq_dict


def cut_useless_prefix(filepath):
    start_pos = filepath.rfind('paddle')
    return filepath[start_pos:]


def print_count_result(count_dict):
    print("File:Line - Error Count")
    for key, value in count_dict.items():
        print("%s - %d" % (cut_useless_prefix(cpt.to_text(key)[:-1]), value))


if __name__ == "__main__":
    count_dict = count_error_frequency(ERROR_LOG_HOME)
    print_count_result(count_dict)
