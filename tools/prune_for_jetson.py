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
"""
This script simply removes all grad ops and kernels. You should use this script 
when cmake ON_INFER=ON, which can greatly reduce the volume of the prediction library.
"""

import os
import sys
import re
import glob
import io


def find_type_files(cur_dir, file_type, file_list=[]):
    next_level_dirs = os.listdir(cur_dir)
    for next_level_name in next_level_dirs:
        next_level_dir = os.path.join(cur_dir, next_level_name)
        if os.path.isfile(next_level_dir):
            if os.path.splitext(next_level_dir)[1] == file_type:
                file_list.append(next_level_dir)
        elif os.path.isdir(next_level_dir):
            find_type_files(next_level_dir, file_type, file_list)
    return file_list


def find_kernel(content, pattern):
    res = re.findall(pattern, content, flags=re.DOTALL)
    ret = []
    for p in res:
        left, right = 0, 0
        for c in p:
            if c == '{':
                left += 1
            elif c == '}':
                right += 1

        if left == right:
            ret.append(p)

    return ret, len(ret)


if __name__ == '__main__':

    tool_dir = os.path.dirname(os.path.abspath(__file__))

    if sys.version_info[0] == 3:
        all_op = glob.glob(
            os.path.join(tool_dir, '../paddle/phi/kernels/**/*.cc'),
            recursive=True)
        all_op += glob.glob(
            os.path.join(tool_dir, '../paddle/phi/kernels/**/*.cu'),
            recursive=True)
    elif sys.version_info[0] == 2:
        all_op = find_type_files(
            os.path.join(tool_dir, '../paddle/phi/kernels/'), '.cc')
        all_op = find_type_files(
            os.path.join(tool_dir, '../paddle/phi/kernels/'), '.cu', all_op)

    register_op_count = 0

    for op_file in all_op:
        op_name = os.path.split(op_file)[1]

        all_matches = []
        with io.open(op_file, 'r', encoding='utf-8') as f:
            content = ''.join(f.readlines())
            op_pattern = 'PD_REGISTER_KERNEL\(.*?\).*?\{.*?\}'
            op, op_count = find_kernel(content, op_pattern)
            register_op_count += op_count

            all_matches.extend(op)

        for p in all_matches:
            content = content.replace(p, '')

        with io.open(op_file, 'w', encoding='utf-8') as f:
            f.write(u'{}'.format(content))

    print('We erase all grad op and kernel for Paddle-Inference lib.')
    print('%50s%10s' % ('type', 'count'))
    print('%50s%10s' % ('REGISTER_OPERATOR', register_op_count))
