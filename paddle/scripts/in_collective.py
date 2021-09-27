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

import os, sys
colls = os.popen(
    "find ${PADDLE_ROOT} -name '*op_npu.cc'|grep '/collective/' ").readlines()
colls = ["test_" + c.split("/")[-1].split(".cc")[0] for c in colls]
colls2 = [c.replace("_op_npu", "_npu") for c in colls if "_op_npu" in c]
colls += colls2
for c in colls:
    for key in sys.argv[-2:]:
        if key == c:
            print("True")
            exit()
        if '"' in key:
            for k in key.split('"'):
                if k == c:
                    print("True")
                    exit()
print("False")
