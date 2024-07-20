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

import logging
import os
import sys
from enum import Enum
import tensorrt as trt
import torch

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from paddle.base.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

class Frameworks(Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    TRT = "trt"


def has_dynamic_shape(shape):
    return any(s == -1 for s in shape)


def append_ones(network, input, name, num_prepend_ones):
    layer = network.add_shuffle(input)

    if has_dynamic_shape(input.shape):
        input_shape_layer = network.add_shape(input)
        input_shape_layer.name = f"{name}_broadcast_orig_shape"
        prepend_shape_layer = network.add_constant(
            (num_prepend_ones,), np.ones((num_prepend_ones,), dtype=np.int32)
        )
        prepend_shape_layer.name = f"{name}_broadcast_prepend_ones"
        reshape_dim_layer = network.add_concatenation(
            [prepend_shape_layer.get_output(0), input_shape_layer.get_output(0)]
        )
        reshape_dim_layer.axis = 0
        reshape_dim_layer.name = f"{name}_broadcast_final_shape"
        layer.set_input(1, reshape_dim_layer.get_output(0))
    else:
        layer.reshape_dims = (1,) * num_prepend_ones + tuple(input.shape)

    layer.name = name
    return layer.get_output(0)


def broadcast(network, a, b, a_name, b_name, preset_diff=0):
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    diff = len(a_shape) - len(b_shape) - preset_diff
    if diff > 0:
        b = append_ones(network, b, f"{b_name}_broadcast", diff)
    elif diff < 0:
        a = append_ones(network, a, f"{a_name}_broadcast", -diff)

    return a, b


def get_axes_for_reduce_op(
    dim,
    has_implicit_batch_dimension=False,
):
    if isinstance(dim, int):
        dim = (dim,)

    if has_implicit_batch_dimension:
        assert (
            0 not in dim
        ), "Can't reduce over batch dimension when it's implicit."

    axes = 0
    for d in dim:
        axes |= 1 << (d - (1 if has_implicit_batch_dimension else 0))

    return axes


def get_trt_tensor(network, input_val, name, dtype):
    if isinstance(input_val, bool):
        input_val = int(input_val)

    if isinstance(input_val, torch.Tensor) and (
        input_val.dtype == torch.bool or input_val.dtype == torch.int64
    ):
        input_val = input_val.to(torch.int32)
    elif isinstance(input_val, np.ndarray) and (
        input_val.dtype == np.bool_ or input_val.dtype == np.int64
    ):
        input_val = input_val.to(np.int32)

    if isinstance(input_val, (torch.Tensor, np.ndarray, int, float)):
        return create_constant(network, input_val, name, dtype)
    elif isinstance(input_val, trt.tensorrt.ITensor):
        return input_val

    raise RuntimeError(
        f"Received input {input_val} of name {name} that "
        "is not part of the TensorRT region!"
    )


def get_dynamic_dims(shape):
    """
    This function finds the dynamic dimensions in the given
    shape. A dimension is dynamic if it's -1.

    Args:
        shape (Shape): A sequence of integer that represents
            the shape of a tensor.

    Returns:
        A list of integers contains all the dynamic dimensions
        in the given shape
    """
    dynamic_dims = []
    for i, s in enumerate(shape):
        if s == -1:
            dynamic_dims.append(i)
    return dynamic_dims


# def unified_dtype_converter(dtype,to_framework):
#     assert to_framework in Frameworks,f"Expected valid Framework for translation, got {to_framework}" 
#     trt_major_version=int(trt.__version__.split("."[0]))
#     if dtype in (np.int8,torch.int8,trt,int8):
#         return DataTypeEquivalence[trt.int8][to_framework]
#     elif dtype in (np.bool_, torch.bool, trt.bool):
#         return DataTypeEquivalence[trt.bool][to_framework]
    
        
        
    

# def add_binary_elementwise_layer( network,lhs_val,rhs_val,op_type):
#     lhs_dtype=None
#     rhs_dtype=None
#     is_lhs_trt_tensor=False
#     is_rhs_trt_tensor=False
    
#     if isinstance(lhs_val,trt.tensorrt.ITensor):
#         lhs_dtype=unified_dtype_converter(lhs_val.dtype,Framework.TORCH)
    
    