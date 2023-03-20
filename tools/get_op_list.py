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

import argparse
import os
import re

import numpy as np

import paddle
from paddle.inference import _get_phi_kernel_name

paddle.enable_static()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default="",
        help='Directory of the inference models that named with pdmodel.',
    )
    parser.add_argument(
        '--op_list',
        type=str,
        default="",
        help='List of ops like "conv2d;pool2d;relu".',
    )
    return parser.parse_args()


def get_model_ops(model_file, ops_set):
    model_bytes = paddle.static.load_from_file(model_file)
    pg = paddle.static.deserialize_program(model_bytes)

    for i in range(0, pg.desc.num_blocks()):
        block = pg.desc.block(i)
        size = block.op_size()

        for j in range(0, size):
            ops_set.add(block.op(j).type())


def get_model_phi_kernels(ops_set):
    phi_set = set()
    for op in ops_set:
        print(op)
        print(_get_phi_kernel_name(op))
        phi_set.add(_get_phi_kernel_name(op))

    return phi_set


if __name__ == '__main__':
    args = parse_args()
    ops_set = set()
    if args.op_list != "":
        op_list = args.op_list.split(";")
        for op in op_list:
            ops_set.add(op)

    if args.model_dir != "":
        for root, dirs, files in os.walk(args.model_dir, topdown=True):
            for name in files:
                if re.match(r'.*pdmodel', name):
                    get_model_ops(os.path.join(root, name), ops_set)
    phi_set = get_model_phi_kernels(ops_set)
    ops = ";".join(ops_set)
    kernels = ";".join(phi_set)
    print("op_list: ", ops)
    print("kernel_list: ", kernels)
    ops = np.array([ops])
    kernels = np.array([kernels])
    np.savetxt("op_list.txt", ops, fmt='%s')
    np.savetxt("kernel_list.txt", kernels, fmt='%s')
