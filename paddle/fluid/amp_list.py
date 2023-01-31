# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.fluid import core


def get_amp_op_lists(white_list, include_grad=True):
    if not isinstance(white_list, (list, tuple)):
        white_list = [white_list]

    all_ops = core.get_all_op_names()

    white_set = set()
    for op in white_list:
        if include_grad:
            candidates = [tmp for tmp in all_ops if tmp.startswith(op)]
            while op in candidates:
                white_set.add(op)
                op += '_grad'
        else:
            if op in all_ops:
                white_set.add(op)

    black_set = set()
    for op in all_ops:
        if op not in white_set:
            black_set.add(op)

    white_list = list(white_set)
    black_list = list(black_set)
    white_list.sort()
    black_list.sort()
    assert len(white_list) + len(black_list) == len(all_ops)
    return white_list, black_list


w, b = get_amp_op_lists(['matmul', 'matmul_v2', 'hehe'], False)
print(w)
