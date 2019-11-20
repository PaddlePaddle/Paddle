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

white_list = [
    'fill_zeros_like',
    'batch_norm',
    'crop',
    'cvm',
    'op_with_kernel',
    'dgc_momentum',
    'fake_quantize_range_abs_max',
    'flatten_grad',
    'batch_norm_grad',
    'squeeze_grad',
]

if len(sys.argv) != 2:
    print(
        'Usage: python generate_op_use_unused_inputs_white_list.py [filepath]')
    sys.exit(1)

with open(sys.argv[1], 'w') as f:
    for op in white_list:
        f.write(op + '\n')
