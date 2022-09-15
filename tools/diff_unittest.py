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

try:
    f1 = open(sys.argv[1], 'r')
    origin = f1.read()
    origin = origin.splitlines()
except:
    sys.exit(0)
else:
    f1.close()

try:
    f2 = open(sys.argv[2], 'r')
    new = f2.read()
    new = new.splitlines()
except:
    sys.exit(0)
else:
    f2.close()

error = False
diffs = []
for i in origin:
    if i not in new:
        error = True
        diffs.append(i)
'''
If you delete the unit test, such as commenting it out,
please ask for approval of one RD below for passing CI:

    - kolinwei(recommended) or zhouwei25
'''
if error:
    for each_diff in diffs:
        print("- %s" % each_diff)
