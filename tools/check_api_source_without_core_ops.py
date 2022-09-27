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

import difflib
import sys
import importlib
import os
import count_api_without_core_ops

with open(sys.argv[1], 'r') as f:
    origin = f.read()
    origin = origin.splitlines()

with open(sys.argv[2], 'r') as f:
    new = f.read()
    new = new.splitlines()

differ = difflib.Differ()
result = differ.compare(origin, new)

api_with_ops, api_without_ops = count_api_without_core_ops.get_apis_with_and_without_core_ops(
    ['paddle'])

error = False
# get all diff apis
# check if the changed api's source code contains append_op but not core.ops
diffs = []
for each_diff in result:
    if each_diff[0] == '+':
        api_name = each_diff.split(' ')[1].strip()
        if api_name in api_without_ops and api_name.find('sequence') == -1:
            error = True
            diffs += [api_name]

if error:
    for each_diff in diffs:
        print(each_diff)
