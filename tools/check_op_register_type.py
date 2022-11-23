# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
Print all registered kernels of a python module in alphabet order.

Usage:
    python check_op_register_type.py > all_kernels.txt
    python check_op_register_type.py OP_TYPE_DEV.spec OP_TYPE_PR.spec > is_valid
"""
import sys
import re
import difflib
import collections
import paddle.fluid as fluid

INTS = set(['int', 'int64_t'])
FLOATS = set(['float', 'double'])


def get_all_kernels():
    all_kernels_info = fluid.core._get_all_register_op_kernels()
    # [u'data_type[double]:data_layout[ANY_LAYOUT]:place[CPUPlace]:library_type[PLAIN]'
    op_kernel_types = collections.defaultdict(list)
    for op_type, op_infos in all_kernels_info.items():
        is_grad_op = op_type.endswith("_grad")
        if is_grad_op: continue

        pattern = re.compile(r'data_type\[([^\]]+)\]')
        for op_info in op_infos:
            infos = pattern.findall(op_info)
            if infos is None or len(infos) == 0: continue

            register_type = infos[0].split(":")[-1]
            op_kernel_types[op_type].append(register_type.lower())

    for (op_type, op_kernels) in sorted(op_kernel_types.items(),
                                        key=lambda x: x[0]):
        print(op_type, " ".join(sorted(op_kernels)))


def read_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        content = content.splitlines()
    return content


def print_diff(op_type, register_types):
    lack_types = set()
    if len(INTS - register_types) == 1:
        lack_types |= INTS - register_types
    if len(FLOATS - register_types) == 1:
        lack_types |= FLOATS - register_types

    print("{} only supports [{}] now, but lacks [{}].".format(
        op_type, " ".join(register_types), " ".join(lack_types)))


def check_add_op_valid():
    origin = read_file(sys.argv[1])
    new = read_file(sys.argv[2])

    differ = difflib.Differ()
    result = differ.compare(origin, new)

    for each_diff in result:
        if each_diff[0] in ['+'] and len(each_diff) > 2:  # if change or add op
            op_info = each_diff[1:].split()
            if len(op_info) < 2: continue
            register_types = set(op_info[1:])
            if len(FLOATS - register_types) == 1 or \
                    len(INTS - register_types) == 1:
                print_diff(op_info[0], register_types)


if len(sys.argv) == 1:
    get_all_kernels()
elif len(sys.argv) == 3:
    check_add_op_valid()
else:
    print("Usage:\n" \
          "\tpython check_op_register_type.py > all_kernels.txt\n" \
          "\tpython check_op_register_type.py OP_TYPE_DEV.spec OP_TYPE_PR.spec > diff")
