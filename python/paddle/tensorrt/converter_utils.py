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

import numpy as np
import tensorrt as trt

from .util import get_trt_version_list

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from paddle.base.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)

version_list = get_trt_version_list()


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


def get_trt_plugin(plugin_name, field_collection, version, plugin_namespace=""):
    plugin_registry = trt.get_plugin_registry()
    plugin_creator = plugin_registry.get_plugin_creator(
        plugin_name, version, plugin_namespace
    )
    assert (
        plugin_creator
    ), f"Unabled to find plugin creator with name{plugin_name}"
    plugin = plugin_creator.create_plugin(
        name=plugin_name, field_collection=field_collection
    )
    assert plugin is not None, f"Plugin:{plugin_name} could not be fetched"
    return plugin


def get_positive_dim(dim, dim_size):
    if dim < 0:
        return dim % dim_size
    return dim


def add_elementwise_layer(network, paddle_op, inputs, op_type):
    weight_shape = paddle_op.operands()[1].source().shape
    input_shape = paddle_op.operands()[0].source().shape

    weight_tensor = inputs[1]
    input_tensor = inputs[0]
    if type(inputs[1]) == trt.Weights:
        weight_tensor = network.add_constant(
            weight_shape, inputs[1]
        ).get_output(0)
    if type(inputs[0]) == trt.Weights:
        input_tensor = network.add_constant(input_shape, inputs[0]).get_output(
            0
        )
    lhs_val, rhs_val = broadcast(
        network,
        input_tensor,
        weight_tensor,
        input_tensor.name,
        weight_tensor.name,
    )
    layer = network.add_elementwise(lhs_val, rhs_val, op_type)
    return layer.get_output(0)


# Create and add 1D constant layer
def add_1D_constant_layer(network, data, dtype=np.int32):
    if not isinstance(data, list):
        data = [data]
    constant_data = np.array(data, dtype=dtype)
    constant_layer = network.add_constant(constant_data.shape, constant_data)
    return constant_layer.get_output(0)


# Concat not make rank changed
def trt_concat(network, inputs, axis=0):
    concat_layer = network.add_concatenation(inputs=inputs)
    if axis != 0:
        concat_layer.axis = axis
    return concat_layer.get_output(0)


def trt_cast(network, input, dtype):
    identity_layer = network.add_identity(input)
    identity_layer.set_output_type(0, dtype)
    identity_layer.get_output(0).dtype = dtype
    return identity_layer.get_output(0)


def trt_shape(network, input):
    shape_layer = network.add_shape(input)
    if version_list[0] >= 10:  # trt_version >=10
        return trt_cast(network, shape_layer.get_output(0), trt.int32)
    return shape_layer.get_output(0)


def trt_reshape(network, input, new_shape, name="", is_shape_tensor=False):
    reshape_layer = network.add_shuffle(input)
    if is_shape_tensor:
        reshape_layer.set_input(1, new_shape)
    else:
        reshape_layer.reshape_dims = new_shape
    if name != "":
        reshape_layer.name = name
    return reshape_layer.get_output(0)


# Get element tensor of 1D shape tensor
def get_shape_tensor_element(network, x, index):
    assert index >= 0, (
        "The index should be greater or equal than 0, but got %d" % index
    )
    gather_layer = network.add_gather(
        input=x, indices=add_1D_constant_layer(network, index), axis=0
    )
    return gather_layer.get_output(0)


def trt_less(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.LESS)
    return layer.get_output(0)


def trt_sum(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
    return layer.get_output(0)


def trt_max(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.MAX)
    return layer.get_output(0)


def trt_sub(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.SUB)
    return layer.get_output(0)


def trt_min(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.MIN)
    return layer.get_output(0)


def trt_mul(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.PROD)
    return layer.get_output(0)


def trt_div(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.DIV)
    return layer.get_output(0)


def trt_floor_div(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.FLOOR_DIV)
    return layer.get_output(0)


def trt_equal(network, a, b):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.EQUAL)
    return layer.get_output(0)


def cast_tensor(network, input_tensor, dtype):
    layer = network.add_identity(input_tensor)
    layer.set_output_type(0, dtype)
    return layer.get_output(0)


def build_start_tensor(network, rank, axis_tensor, offset):
    # Create indices_tensor [0, 1, ..., rank-1]
    indices = np.arange(rank, dtype=np.int32)
    indices_tensor = network.add_constant([rank], indices).get_output(0)

    # Create mask: mask = (indices == axis_tensor)
    mask = network.add_elementwise(
        indices_tensor, axis_tensor, trt.ElementWiseOperation.EQUAL
    ).get_output(0)
    mask_int = cast_tensor(network, mask, trt.int32)

    # Calculate start_tensor = mask_int * offset
    start_tensor = network.add_elementwise(
        mask_int, offset, trt.ElementWiseOperation.PROD
    ).get_output(0)

    return start_tensor


def build_size_tensor(
    network, rank, axis_tensor, size_value, input_shape_tensor
):
    # Create indices_tensor [0, 1, ..., rank-1]
    indices = np.arange(rank, dtype=np.int32)
    indices_tensor = network.add_constant([rank], indices).get_output(0)

    # Create mask: mask = (indices == axis_tensor)
    mask = network.add_elementwise(
        indices_tensor, axis_tensor, trt.ElementWiseOperation.EQUAL
    ).get_output(0)
    mask_int = cast_tensor(network, mask, trt.int32)

    # Create ones_tensor
    ones_tensor = network.add_constant(
        [rank], np.ones([rank], dtype=np.int32)
    ).get_output(0)

    # Calculate inverse_mask = ones_tensor - mask_int
    inverse_mask = network.add_elementwise(
        ones_tensor, mask_int, trt.ElementWiseOperation.SUB
    ).get_output(0)

    # Calculate size_tensor = mask_int * size_value + inverse_mask * input_shape_tensor
    size_value_broadcast = network.add_elementwise(
        mask_int, size_value, trt.ElementWiseOperation.PROD
    ).get_output(0)

    input_shape_broadcast = network.add_elementwise(
        inverse_mask, input_shape_tensor, trt.ElementWiseOperation.PROD
    ).get_output(0)

    size_tensor = network.add_elementwise(
        size_value_broadcast,
        input_shape_broadcast,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)

    return size_tensor
