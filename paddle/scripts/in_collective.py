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

import os
import sys
'''
输入是一个字符串，包含所有单测（通信单测与非通信单测）名，各个单测名以‘|^’分隔

输出是一个字符串，包含所有通信单测名，各个单测名以‘|^’分隔

本程序会搜索所有以op_npu.cc结尾 并且包含/collective/的文件名，
以op_npu.cc结尾，表示是npu op， 包含/collective/，表示是通信op
对所有被筛选出的文件，加上前缀test_,同时去掉后缀.cc
通信算子的单测名格式要求是test_xxx_npu 或者 test_xxx_op_npu
 
'''
collective_set = os.popen(
    "find ${PADDLE_ROOT} -name '*op_npu.cc'|grep '/collective/' ").readlines()
collective_set = [
    "test_" + c.split("/")[-1].split(".cc")[0] for c in collective_set
]
collective_set2 = [
    c.replace("_op_npu", "_npu") for c in collective_set if "_op_npu" in c
]
collective_set += collective_set
outer_keys = set(sys.argv[1].split("|^"))
inner_keys = set(collective_set)
output = ""
for key in outer_keys.intersection(inner_keys):
    if output == "":
        output = key
    else:
        output = output + "|^" + key
print(output)
