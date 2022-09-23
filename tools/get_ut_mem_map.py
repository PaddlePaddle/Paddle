# -*- coding: utf-8 -*-

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import json
import sys


def get_ut_mem(rootPath):
    case_dic = {}
    for parent, dirs, files in os.walk(rootPath):
        for f in files:
            if f.endswith('$-gpu.log'):
                continue
            ut = f.replace('^', '').replace('$.log', '')
            case_dic[ut] = {}
            filename = '%s/%s' % (parent, f)
            fi = open(filename, mode='rb')
            lines = fi.readlines()
            mem_reserved1 = -1
            mem_nvidia1 = -1
            caseTime = -1
            for line in lines:
                line = line.decode('utf-8', errors='ignore')
                if '[Memory Usage (Byte)] gpu' in line:
                    mem_reserved = round(
                        float(
                            line.split(' : Reserved = ')[1].split(
                                ', Allocated = ')[0]), 2)
                    if mem_reserved > mem_reserved1:
                        mem_reserved1 = mem_reserved
                if 'MAX_GPU_MEMORY_USE=' in line:
                    mem_nvidia = round(
                        float(
                            line.split('MAX_GPU_MEMORY_USE=')[1].split('\\n')
                            [0].strip()), 2)
                    if mem_nvidia > mem_nvidia1:
                        mem_nvidia1 = mem_nvidia
                if 'Total Test time (real)' in line:
                    caseTime = float(
                        line.split('Total Test time (real) =')[1].split('sec')
                        [0].strip())
            if mem_reserved1 != -1:
                case_dic[ut]['mem_reserved'] = mem_reserved1
            if mem_nvidia1 != -1:
                case_dic[ut]['mem_nvidia'] = mem_nvidia1
            if caseTime != -1:
                case_dic[ut]['time'] = caseTime
            fi.close()

    if not os.path.exists("/pre_test"):
        os.mkdir("/pre_test")
    ut_mem_map_file = "/pre_test/ut_mem_map.json"
    with open(ut_mem_map_file, "w") as f:
        json.dump(case_dic, f)


if __name__ == "__main__":
    rootPath = sys.argv[1]
    get_ut_mem(rootPath)
