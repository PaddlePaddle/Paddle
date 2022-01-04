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


def check_suffix():
    suffix_arr = [".pyc"]
    json_buff = ""
    for line in sys.stdin:
        json_buff = "".join([json_buff, line])
    json_obj = json.loads(json_buff)
    if not isinstance(json_obj, list):
        print('Json String Should be a list Object\n')
        return
    files_with_invalid_suffix = []
    for i in range(len(json_obj)):
        file_name = json_obj[i]["filename"]
        if file_name == None:
            continue
        for suffix in suffix_arr:
            if file_name.endswith(suffix):
                files_with_invalid_suffix.append(file_name)
                break
    if len(files_with_invalid_suffix) != 0:
        print('Error: Find file(s): [\n')
        for i in range(len(files_with_invalid_suffix)):
            print('\t' + files_with_invalid_suffix[i] + '\n')
        print(
            ' ] end(s) with invalid suffix, Please check if these files are temporary.'
        )


if __name__ == "__main__":
    check_suffix()
