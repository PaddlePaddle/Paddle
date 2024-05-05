# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import re


def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


template = '''
python3.8  ${compile_file}     \
/zhoukangkang/triton/python/paddle_tutorials/weight-only-int8.py    \
-n wint8_kernel   \
-o ${wint8_dir}/wint8     \
--out-name wint8_kernel     \
-w ${num_warps}   -ns ${num_stages} \
-s   "*fp16:16, *u8:16, *fp16:16, *fp16:16, *fp16:16, i32,i32:16,i32:16,  i32:16,i32:1,  i32:1,i32:16, i32:16,i32:1, ${block_m}, ${block_n}, ${block_k}, 1, ${split_k}"\
 -g   "((M+${block_m}-1)/${block_m}) * ((N+${block_n}-1)/${block_n}), ${split_k}, 1" \
'''

config_num = 0
thread_num = 200
for num_stages in [2, 3, 4, 5, 6]:
    for block_m in [16, 32, 64, 128]:
        for block_n in [64, 128, 256]:
            for block_k in [64, 128, 256]:
                num_warps = 4
                if block_m * block_n >= 128 * 256:
                    num_warps = 8
                for split_k in [1, 2, 4, 8]:
                    values = {
                        "num_stages": str(num_stages),
                        "block_m": str(block_m),
                        "block_n": str(block_n),
                        "block_k": str(block_k),
                        "split_k": str(split_k),
                        "num_warps": str(num_warps),
                    }
                    result = SubstituteTemplate(template, values)
                    config_num += 1
                    result += " &"
                    if config_num % thread_num == 0:
                        result += "\nwait"
                    print(result)
print("wait")
