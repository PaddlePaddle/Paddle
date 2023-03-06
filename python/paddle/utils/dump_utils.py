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

import numpy as np

import paddle

__all__ = [
    'numpy_from_file',
    'init_api_dump',
    'enable_api_dump',
    'disable_api_dump',
]


def numpy_from_file(path):
    f = open(path, 'rb')

    text = f.read()
    p = text.find(b'\n')
    dims = text[0:p].decode('ascii').strip('\n').split(':')[-1].strip(' ')
    if len(dims) == 0:
        dims = [1]
    else:
        dims = [int(x) for x in dims.split(',')]

    text = text[p + 1 :]
    p = text.find(b'\n')
    dtype = text[0:p].decode('ascii').strip('\n').split(':')[-1].strip(' ')

    text = text[p + 1 :]
    p = text.find(b'\n')
    layout = text[0:p].decode('ascii').strip('\n').split(':')[-1].strip(' ')

    text = text[p + 1 :]
    p = text.find(b'\n')
    buffer = text[p + 1 :]
    return np.frombuffer(buffer, dtype=dtype, count=np.prod(dims)).reshape(dims)


def init_api_dump(
    ordered: bool = True, binary: bool = True, api_dump_list: str = ''
):
    paddle.set_flags(
        {
            'FLAGS_ordered_api_dump': ordered,
            'FLAGS_binary_api_dump': binary,
            'FLAGS_api_dump_list': api_dump_list,
        }
    )


def enable_api_dump():
    paddle.set_flags({'FLAGS_api_dump': True})


def disable_api_dump():
    paddle.set_flags({'FLAGS_api_dump': False})
