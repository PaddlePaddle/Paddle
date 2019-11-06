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

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import paddle.fluid as fluid
import sys


def get_op_diff(filename):
    ops_created_by_py_func = set(
        fluid.core._get_use_default_grad_op_desc_maker_ops())

    with open(filename, 'r') as f:
        ops_read_from_file = set([line.strip() for line in f.readlines()])

    diff_ops = []

    for op in ops_read_from_file:
        if op not in ops_created_by_py_func:
            diff_ops.append(op)
        else:
            ops_created_by_py_func.remove(op)

    err_msg = []
    diff_ops = list(diff_ops)
    if len(diff_ops) > 0:
        err_msg.append('Added grad op with DefaultGradOpDescMaker: ' + str(
            diff_ops))

    ops_created_by_py_func = list(ops_created_by_py_func)
    if len(ops_created_by_py_func) > 0:
        err_msg.append('Remove grad op with DefaultGradOpDescMaker: ' + str(
            ops_created_by_py_func))

    return err_msg


if len(sys.argv) != 2:
    print('Usage: python diff_use_default_grad_op_maker.py [filepath]')
    sys.exit(1)

file_path = str(sys.argv[1])
err_msg = get_op_diff(file_path)

if len(err_msg) > 0:
    _, filename = os.path.split(file_path)
    print('File `{}` is wrong compared to your PR revision!'.format(filename))
    print(
        'Please use `python generate_op_use_grad_op_desc_maker_spec.py [filepath]` to generate new `{}` file'.
        format(filename))
    print('Error message is: ' + '; '.join(err_msg))
    sys.exit(1)
