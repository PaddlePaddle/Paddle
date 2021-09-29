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
