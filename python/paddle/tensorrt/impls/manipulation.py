# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

import numpy as np
import tensorrt as trt

from paddle.base.log_helper import get_logger
from paddle.tensorrt.register import converter_registry

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)
from paddle.tensorrt.converter_utils import (
    get_axes_for_reduce_op,
    get_positive_dim,
    get_shape_with_dynamic_shape,
    has_dynamic_shape,
)


@converter_registry.register("pd_op.reshape", trt_version="8.x")
def reshape_converter(network, paddle_op, inputs):
    input_tensor, shape_tensor = inputs
    input_shape = paddle_op.operands()[0].source().shape

    output_shape = paddle_op.results()[0].shape
    if network.has_implicit_batch_dimension:
        output_shape = output_shape[1:]

    if type(input_tensor) == trt.Weights:
        input_tensor = network.add_constant(
            input_shape, input_tensor
        ).get_output(0)

    shuffle_layer = network.add_shuffle(input_tensor)

    try:
        reshape_dims = (
            paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
        )
        shuffle_layer.reshape_dims = tuple(reshape_dims)
    except Exception:
        shuffle_layer.set_input(1, shape_tensor)

    return shuffle_layer.get_output(0)


@converter_registry.register("pd_op.gather_nd", trt_version="8.x")
def gather_nd_converter(network, paddle_op, inputs):
    input_tensor, indices_tensor = inputs
    shuffle_layer = network.add_shuffle(indices_tensor)
    shuffle_layer.first_transpose = trt.Permutation([1, 0])
    # import pdb;pdb.set_trace()
    non_zero_layer = network.add_gather_v2(
        input_tensor, shuffle_layer.get_output(0), trt.GatherMode.ND
    )
    return non_zero_layer.get_output(0)


@converter_registry.register("pd_op.flatten", trt_version="8.x")
def flatten_converter(network, paddle_op, inputs):
    input_val = inputs[0]
    input_val_shape = input_val.shape
    dims = len(input_val_shape)

    start_axis = paddle_op.attrs().get("start_axis")
    stop_axis = paddle_op.attrs().get("stop_axis")

    flatten_layer = network.add_shuffle(input_val)

    if not has_dynamic_shape(input_val_shape):
        if start_axis < 0:
            start_axis += dims + 1
        if stop_axis < 0:
            stop_axis += dims + 1

        flatten_dim = 1
        final_shape = []

        for i, s in enumerate(input_val_shape):
            if i >= start_axis and i <= stop_axis:
                flatten_dim *= s
            elif i == stop_axis + 1:
                final_shape.append(flatten_dim)
                final_shape.append(s)
            else:
                final_shape.append(s)

        if stop_axis == len(input_val.shape) - 1:
            final_shape.append(flatten_dim)

        flatten_layer.reshape_dims = tuple(final_shape)
    else:
        input_shape_layer = network.add_shape(input_val)
        input_shape_layer.name = f"{input_val.name}_origin_shape"

        final_shapes = []
        # Shapes before start_axis
        if start_axis > 0:
            prefix_shape_layer = network.add_slice(
                input_shape_layer.get_output(0),
                start=(0,),
                shape=(start_axis,),
                stride=(1,),
            )
            prefix_shape_layer.name = f"{input_val.name}_prefix_shape"
            final_shapes.append(prefix_shape_layer.get_output(0))

        flatten_shape_layer = network.add_slice(
            input_shape_layer.get_output(0),
            start=(start_axis,),
            shape=(stop_axis - start_axis + 1,),
            stride=(1,),
        )
        flatten_shape_layer.name = f"{input_val.name}_need_flatten"
        flatten_shape_layer = network.add_reduce(
            flatten_shape_layer.get_output(0),
            trt.ReduceOperation.PROD,
            axes=get_axes_for_reduce_op(0, False),
            keep_dims=True,
        )
        flatten_shape_layer.name = f"{input_val.name}_flatten_dim"
        final_shapes.append(flatten_shape_layer.get_output(0))

        # Shapes after stop_axis
        if stop_axis < len(input_val_shape) - 1:
            suffix_shape_layer = network.add_slice(
                input_shape_layer.get_output(0),
                start=(stop_axis + 1,),
                shape=(len(input_val_shape) - stop_axis - 1,),
                stride=(1,),
            )
            suffix_shape_layer.name = f"{input_val.name}_suffix_shape"
            final_shapes.append(suffix_shape_layer.get_output(0))

        final_shape_layer = network.add_concatenation(final_shapes)
        final_shape_layer.axis = 0
        final_shape_layer.name = f"{input_val.name}_final_shape"
        flatten_layer.set_input(1, final_shape_layer.get_output(0))

    return flatten_layer.get_output(0)


# In the converter, pd_op.concat has three inputs, because builtin.combine has two inputs.
@converter_registry.register("pd_op.concat", trt_version="8.x")
def concat_converter(network, paddle_op, inputs):
    input_tensors = inputs[:-1]
    axis_tensor = inputs[-1]
    concat_layer = network.add_concatenation(inputs=input_tensors)

    axis = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    axis = int(axis)
    if axis < 0:
        axis = len(input_tensors[0].shape) + axis
    concat_layer.axis = axis

    return concat_layer.get_output(0)


@converter_registry.register("pd_op.unsqueeze", trt_version="8.x")
@converter_registry.register("pd_op.unsqueeze_", trt_version="8.x")
def unsqueeze_converter(network, paddle_op, inputs):
    input_val = inputs[0]
    input_shape = paddle_op.operands()[0].source().shape
    input_shape_size = len(input_shape)

    if type(input_val) == trt.Weights:
        input_val = network.add_constant(input_shape, input_val).get_output(0)
    axis = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    axis = axis[0]

    axis = get_positive_dim(axis, input_shape_size + 1)
    layer = network.add_shuffle(input_val)
    layer.reshape_dims = (
        tuple(input_val.shape)[:axis] + (1,) + tuple(input_val.shape)[axis:]
    )
    return layer.get_output(0)


@converter_registry.register("pd_op.squeeze", trt_version="8.x")
@converter_registry.register("pd_op.squeeze_", trt_version="8.x")
def squeeze_converter(network, paddle_op, inputs):
    input_val = inputs[0]
    input_shape = paddle_op.operands()[0].source().shape
    input_shape_size = len(input_shape)

    if type(input_val) == trt.Weights:
        input_val = network.add_constant(input_shape, input_val).get_output(0)

    axis = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    axis = axis[0]

    axis = get_positive_dim(axis, input_shape_size + 1)
    output_shape = []
    for i, s in enumerate(input_shape):
        if i == axis and s == 1:
            continue
        output_shape.append(s)

    layer = network.add_shuffle(input_val)
    layer.reshape_dims = tuple(output_shape)
    return layer.get_output(0)


@converter_registry.register("pd_op.slice", trt_version="8.x")
def slice_converter(network, paddle_op, inputs):
    input_val = inputs[0]
    input_shape = paddle_op.operands()[0].source().shape
    start = paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    ends = paddle_op.operands()[2].source().get_defining_op().attrs()["value"]
    axes = paddle_op.attrs().get("axes")
    decrease_axis = paddle_op.attrs().get("decrease_axis")

    input_dim = len(input_shape)

    shape_tensor = network.add_shape(input_val)

    trt_start_dims = [0] * input_dim  # start
    trt_step_dims = [1] * input_dim  # stride

    start_tensor = None
    end_tensor = None

    starts_tensor = []
    ends_tensor = []

    for i in range(input_dim):
        starts_tensor.append(
            network.add_constant([1], np.array([0], dtype=np.int32)).get_output(
                0
            )
        )
        index_val = network.add_constant([1], np.array([i], dtype=np.int32))
        end_val = network.add_gather(
            shape_tensor.get_output(0), index_val.get_output(0), 0
        )
        ends_tensor.append(end_val.get_output(0))
    print("starts_tensor", starts_tensor)
    print("ends_tensor", ends_tensor)


@converter_registry.register("pd_op.split_with_num", trt_version="8.x")
def split_with_num_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape = paddle_op.operands()[0].source().shape
    input_shape_size = len(input_shape)
    dynamic_shape = has_dynamic_shape(input_shape)
    axis = int(
        paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    )

    num_splits = paddle_op.attrs().get("num")
    input_axis_size = input_shape[axis]
    split_size = input_axis_size // num_splits
    split_sizes = [split_size] * num_splits

    start = [0] * input_shape_size
    stride = [1] * input_shape_size
    offset = 0
    outputs = []

    for size in split_sizes:
        shape = list(input_shape)
        shape[axis] = size
        start[axis] = offset
        if dynamic_shape:
            shape_tensor = get_shape_with_dynamic_shape(
                network, shape, input_tensor
            )
        layer = network.add_slice(
            input_tensor,
            start=start,
            shape=[] if dynamic_shape else shape,
            stride=stride,
        )
        if dynamic_shape:
            layer.set_input(2, shape_tensor)
        offset += size
        outputs.append(layer.get_output(0))

    return outputs


@converter_registry.register("pd_op.split", trt_version="8.x")
def split_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape = paddle_op.operands()[0].source().shape
    input_shape_size = len(input_shape)
    dynamic_shape = has_dynamic_shape(input_shape)
    axis = int(
        paddle_op.operands()[2].source().get_defining_op().attrs()["value"]
    )
    num_or_section = (
        paddle_op.operands()[1].source().get_defining_op().attrs()["value"]
    )

    start = [0] * input_shape_size
    stride = [1] * input_shape_size
    offset = 0
    outputs = []
    split_sizes = num_or_section

    for size in split_sizes:
        shape = list(input_shape)
        shape[axis] = size
        start[axis] = offset
        if dynamic_shape:
            shape_tensor = get_shape_with_dynamic_shape(
                network, shape, input_tensor
            )
        layer = network.add_slice(
            input_tensor,
            start=start,
            shape=[] if dynamic_shape else shape,
            stride=stride,
        )
        if dynamic_shape:
            layer.set_input(2, shape_tensor)
        offset += size
        outputs.append(layer.get_output(0))

    return outputs
