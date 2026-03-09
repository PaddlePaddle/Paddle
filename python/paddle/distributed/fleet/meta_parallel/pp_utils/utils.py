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


import paddle
from paddle import _C_ops

__all__ = []


FLOAT_TYPE_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
    paddle.bfloat16: "bfloat16",
    paddle.bool: "bool",
}

PADDLE_TO_NUMBER = {
    paddle.float16: 0,
    paddle.float32: 1,
    paddle.float64: 2,
    paddle.int32: 3,
    paddle.int64: 4,
    paddle.bfloat16: 5,
    paddle.bool: 6,
}

NUMBER_TO_DTYPE = {
    0: "float16",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64",
    5: "bfloat16",
    6: "bool",
}


def is_float_tensor(tensor):
    """Is a float tensor"""
    return tensor.dtype in FLOAT_TYPE_DICT.keys()


def get_tensor_dtype(dtype):
    assert dtype in FLOAT_TYPE_DICT.keys()
    return FLOAT_TYPE_DICT[dtype]


def paddle_2_number(dtype):
    assert dtype in PADDLE_TO_NUMBER.keys()
    return PADDLE_TO_NUMBER[dtype]


def number_2_dtype(number):
    assert number in NUMBER_TO_DTYPE.keys()
    return NUMBER_TO_DTYPE[number]


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
        raise ValueError(f"unknown data type: {tensor.dtype}")
    return tensor.numel() * elem_size


def _all_gather(tensor, group=None, use_calc_stream=True):
    """
    The main difference with paddle.distributed.all_gather:
    no need to pass in tensor_list, the returned tensor is spliced
    """
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id
    nranks = (
        paddle.distributed.collective._get_global_group().nranks
        if group is None
        else group.nranks
    )
    return _C_ops.all_gather(
        tensor,
        ring_id,
        nranks,
    )


def tuple_to_dict_helper(input_tensor):
    # recv tuple -> fwd input dict
    use_dict = False
    if isinstance(input_tensor, tuple):
        use_dict = hasattr(input_tensor[0], "key")
    else:  # single tensor
        use_dict = hasattr(input_tensor, "key")
    if use_dict:
        input_tensor = convert_tensor_tuple_to_dict(input_tensor)
    return input_tensor, use_dict


def dict_to_tuple_helper(output_tensor):
    if isinstance(output_tensor, dict):
        output_tensor_tuple = convert_tensor_dict_to_tuple(
            output_tensor_dict=output_tensor
        )
    else:  # single tensor or tensor tuple
        output_tensor_tuple = output_tensor
    return output_tensor_tuple


def convert_tensor_dict_to_tuple(output_tensor_dict):
    output_tensor = []
    for key, tensor in output_tensor_dict.items():
        if isinstance(tensor, (list, tuple)):
            for idx, t in enumerate(tensor):
                t.key = key + " " + str(idx)
                output_tensor.append(t)
        else:  # single tensor
            tensor.key = key
            output_tensor.append(tensor)

    return tuple(output_tensor)


def convert_tensor_tuple_to_dict(input_tensor_tuple):
    input_tensor_dict = {}
    for tensor in input_tensor_tuple:
        key = tensor.key
        if " " in key:
            real_key, _ = key.split(" ")
            if real_key in input_tensor_dict.keys():
                input_tensor_dict[real_key].append(tensor)
            else:
                input_tensor_dict[real_key] = [tensor]
        else:
            input_tensor_dict[key] = tensor
        delattr(tensor, "key")
    return input_tensor_dict
