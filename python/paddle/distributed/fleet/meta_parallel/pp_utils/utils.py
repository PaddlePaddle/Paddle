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

import abc
import paddle
from ...utils import log_util as hp_util

__all__ = []

FLOAT_TYPE_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
}


def is_float_tensor(tensor):
    """Is a float tensor"""
    return tensor.dtype in FLOAT_TYPE_DICT.keys()


def get_tensor_dtype(dtype):
    assert dtype in FLOAT_TYPE_DICT.keys()
    return FLOAT_TYPE_DICT[dtype]


def get_tensor_bytes(tensor):
    """Get the bytes a tensor occupied."""
    elem_size = None
    if tensor.dtype == paddle.float32:
        elem_size = 4
    elif tensor.dtype == paddle.float64:
        elem_size = 8
    elif tensor.dtype == paddle.int64:
        elem_size = 8
    elif tensor.dtype == paddle.int32:
        elem_size = 4
    elif tensor.dtype == paddle.float16:
        elem_size = 2
    elif tensor.dtype == paddle.int8:
        elem_size = 1
    else:
        raise ValueError("unknown data type: {}".format(tensor.dtype))
    return tensor.numel() * elem_size
