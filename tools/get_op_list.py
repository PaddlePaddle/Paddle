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

import paddle
import argparse
import numpy as np
import os
import re
from paddle.inference import _get_phi_kernel_name

paddle.enable_static()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default="",
        help='Directory of the inference models.',
    )
    return parser.parse_args()


def get_model_ops(model_file):
    model_bytes = paddle.static.load_from_file(model_file)
    pg = paddle.static.deserialize_program(model_bytes)
    ops_set = set()

    for i in range(0, pg.desc.num_blocks()):
        block = pg.desc.block(i)
        size = block.op_size()

        for j in range(0, size):
            ops_set.add(block.op(j).type())

    return ops_set


def get_model_phi_kernels(ops_set):
    phi_set = set()
    for op in ops_set:
        phi_set.add(_get_phi_kernel_name(op))

    return phi_set


if __name__ == '__main__':
    args = parse_args()
    for root, dirs, files in os.walk(args.model_dir, topdown=True):
        for name in files:
            if re.match(r'.*pdmodel', name):
                ops_set = get_model_ops(os.path.join(root, name))
    phi_set = get_model_phi_kernels(ops_set)
    ops = ";".join(ops_set)
    kernels = ";".join(phi_set)
    print("op_list: ", ops)
    print("kernel_list: ", kernels)
    ops = np.array([ops])
    kernels = np.array([kernels])
    np.savetxt("op_list.txt", ops, fmt='%s')
    np.savetxt("kernel_list.txt", kernels, fmt='%s')
