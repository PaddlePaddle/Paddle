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

import sys
import json


def check_suffix(suffix):
    json_buff = ""
    for line in sys.stdin:
        json_buff = "".join([json_buff, line])
    json_obj = json.loads(json_buff)
    if not isinstance(json_obj, list):
        print('Json String Should be a list Object\n')
        return
    files_end_with_pyc = []
    for i in range(len(json_obj)):
        file_name = json_obj[i]["filename"]
        if file_name == None:
            continue
        if file_name.endswith(suffix):
            files_end_with_pyc.append(file_name)
    if len(files_end_with_pyc) != 0:
        print('Find file(s): [\n')
        for i in range(len(files_end_with_pyc)):
            print('\t' + files_end_with_pyc[i] + '\n')
        print(' ] end(s) with suffix name' + ' py.\n')


if __name__ == "__main__":
    if len(sys.argv) == 2:
        check_suffix(sys.argv[1])
    else:
        print("Usage: python check_suffix.py [ suffix_name ] ")
