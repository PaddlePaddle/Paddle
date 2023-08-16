#!/usr/bin/env python

# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

with open(sys.argv[1], 'r') as f:
    origin = f.read()
    origin = origin.splitlines()

with open(sys.argv[2], 'r') as f:
    new = f.read()
    new = new.splitlines()

differ = difflib.Differ()
result = differ.compare(origin, new)

error = False
diffs = []
for each_diff in result:
    if each_diff[0] in ['-', '?']:  # delete or change API is not allowed
        error = True
    elif each_diff[0] == '+':
        error = True

    if each_diff[0] != ' ':
        diffs.append(each_diff)
'''
If you modify/add/delete the API files, including code and comment,
please follow these steps in order to pass the CI:

  1. cd ${paddle_path}, compile paddle;
  2. pip install build/python/dist/(build whl package);
  3. run "python tools/print_signatures.py paddle.fluid> paddle/fluid/API.spec"
'''
if error:
    print('API Difference is: ')
    for each_diff in diffs:
        print(each_diff)
