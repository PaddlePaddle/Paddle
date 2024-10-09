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


import numpy as np
import tensorrt as trt

from paddle.tensorrt.converter_utils import (
    add_1D_constant_layer,
    cast_tensor,
    get_axes_for_reduce_op,
    get_positive_dim,
    get_shape_tensor_element,
    has_dynamic_shape,
    trt_equal,
    trt_floor_div,
    trt_less,
    trt_max,
    trt_min,
    trt_mul,
    trt_sub,
    trt_sum,
)
from paddle.tensorrt.register import converter_registry


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
    input_tensor = inputs[0]
    input_shape = paddle_op.operands()[0].source().shape
    axes = paddle_op.attrs()["axes"]
    decrease_axis = paddle_op.attrs().get("decrease_axis")

    starts_op = paddle_op.operands()[1].source().get_defining_op()
    ends_op = paddle_op.operands()[2].source().get_defining_op()
    input_shape_tensor = network.add_shape(input_tensor).get_output(0)
    input_rank = len(input_tensor.shape)

    starts_tensor = []
    ends_tensor = []
    for i in range(input_rank):
        starts_tensor.append(add_1D_constant_layer(network, 0))
        ends_tensor.append(
            get_shape_tensor_element(network, input_shape_tensor, i)
        )

    if starts_op.name() == "pd_op.full_int_array":
        starts = starts_op.attrs()["value"]
        assert len(starts) == len(
            axes
        ), "The size of this starts: %d must be equal to the axes: %d." % (
            len(starts),
            len(axes),
        )
        for idx in axes:
            if starts[idx] < 0:
                starts_tensor[axes[idx]] = trt_max(
                    network,
                    trt_sum(
                        network,
                        add_1D_constant_layer(network, starts[idx]),
                        get_shape_tensor_element(
                            network, input_shape_tensor, axes[idx]
                        ),
                    ),
                    add_1D_constant_layer(network, 0),
                )
            else:
                starts_tensor[axes[idx]] = trt_min(
                    network,
                    add_1D_constant_layer(network, starts[idx]),
                    get_shape_tensor_element(
                        network, input_shape_tensor, axes[idx]
                    ),
                )
    else:
        starts = inputs[1]
        for idx in axes:
            starts_tensor[axes[idx]] = get_shape_tensor_element(
                network, starts, idx
            )

    if ends_op.name() == "pd_op.full_int_array":
        ends = ends_op.attrs()["value"]
        assert len(ends) == len(
            axes
        ), "The size of this ends: %d must be equal to the axes: %d." % (
            len(ends),
            len(axes),
        )
        for idx in axes:
            if ends[idx] < 0:
                ends_tensor[axes[idx]] = trt_max(
                    network,
                    trt_sum(
                        network,
                        add_1D_constant_layer(network, ends[idx]),
                        get_shape_tensor_element(
                            network, input_shape_tensor, axes[idx]
                        ),
                    ),
                    add_1D_constant_layer(network, 0),
                )
            else:
                ends_tensor[axes[idx]] = trt_min(
                    network,
                    add_1D_constant_layer(network, ends[idx]),
                    get_shape_tensor_element(
                        network, input_shape_tensor, axes[idx]
                    ),
                )
    else:
        ends = inputs[2]
        for idx in axes:
            ends_tensor[axes[idx]] = get_shape_tensor_element(
                network, ends, idx
            )

    start_tensor_layer = network.add_concatenation(starts_tensor)
    start_tensor_layer.axis = 0
    start_tensor = start_tensor_layer.get_output(0)
    end_tensor_layer = network.add_concatenation(ends_tensor)
    end_tensor_layer.axis = 0
    end_tensor = end_tensor_layer.get_output(0)
    size_tensor = trt_sub(network, end_tensor, start_tensor)

    # Create Slice layer
    slice_layer = network.add_slice(
        input_tensor, [0] * input_rank, [0] * input_rank, [1] * input_rank
    )
    slice_layer.set_input(1, start_tensor)
    slice_layer.set_input(2, size_tensor)

    output_tensor = slice_layer.get_output(0)

    # Handle decrease_axis
    if decrease_axis:
        output_shape = network.add_shape(output_tensor).get_output(0)
        new_shape_dims = []
        for i in range(output_shape.shape[0]):
            if i not in decrease_axis:
                dim = network.add_slice(output_shape, [i], [1], [1]).get_output(
                    0
                )
                new_shape_dims.append(dim)
        if len(new_shape_dims) == 0:
            new_shape_tensor = network.add_constant(
                [1], np.array([1], dtype=np.int32)
            )
        else:
            new_shape_tensor = network.add_concatenation(new_shape_dims)
            new_shape_tensor.axis = 0

        reshape_layer = network.add_shuffle(output_tensor)
        reshape_layer.set_input(1, new_shape_tensor.get_output(0))
        output_tensor = reshape_layer.get_output(0)

    return output_tensor


@converter_registry.register("pd_op.split_with_num", trt_version="8.x")
def split_with_num_converter(network, paddle_op, inputs):
    input_tensor = inputs[0]
    input_shape_size = len(input_tensor.shape)

    # Handle the case where axis is of type pir::Value
    axis_op = paddle_op.operands()[1].source().get_defining_op()
    if axis_op.name() == "pd_op.full":
        axis_value = axis_op.attrs()["value"]
        axis_tensor = add_1D_constant_layer(network, axis_value)
    else:
        axis_tensor = inputs[1]
        axis_tensor = cast_tensor(network, axis_tensor, trt.int32)

    num_splits = paddle_op.attrs().get("num")
    num_splits_tensor = add_1D_constant_layer(network, num_splits)

    # Get the dynamic shape of the input tensor
    input_shape_tensor = network.add_shape(input_tensor).get_output(0)

    # Handle negative axis index
    input_shape_size_tensor = add_1D_constant_layer(network, input_shape_size)
    zero_tensor = add_1D_constant_layer(network, 0)

    is_negative_axis = trt_less(network, axis_tensor, zero_tensor)
    is_negative_axis_int = cast_tensor(network, is_negative_axis, trt.int32)

    axis_adjustment = trt_mul(
        network, is_negative_axis_int, input_shape_size_tensor
    )

    axis_tensor = trt_sum(network, axis_tensor, axis_adjustment)

    # Get the size of the dimension specified by axis
    input_axis_size = network.add_gather(
        input_shape_tensor, axis_tensor, axis=0
    ).get_output(0)

    # Compute the size of each split
    split_size = trt_floor_div(network, input_axis_size, num_splits_tensor)

    outputs = []
    for idx in range(num_splits):
        idx_tensor = add_1D_constant_layer(network, idx)
        offset = trt_floor_div(network, idx_tensor, split_size)

        # Create mask
        indices = np.arange(input_shape_size, dtype=np.int32)
        indices_tensor = network.add_constant(
            [input_shape_size], indices
        ).get_output(0)

        mask = trt_equal(network, indices_tensor, axis_tensor)
        mask_int = cast_tensor(network, mask, trt.int32)

        # Calculate start_tensor
        offset_broadcast = trt_mul(network, offset, mask_int)
        start_tensor = offset_broadcast

        # Calculate size_tensor
        size_tensor = trt_mul(network, split_size, mask_int)

        # Get the sizes of other dimensions
        ones_tensor = network.add_constant(
            [input_shape_size], np.ones([input_shape_size], dtype=np.int32)
        ).get_output(0)
        other_dims_mask = trt_sub(network, ones_tensor, mask_int)
        other_dims = trt_mul(network, input_shape_tensor, other_dims_mask)
        size_tensor = trt_sum(network, size_tensor, other_dims)

        # Create Slice layer
        slice_layer = network.add_slice(
            input_tensor,
            [0] * input_shape_size,
            [0] * input_shape_size,
            [1] * input_shape_size,
        )
        slice_layer.set_input(1, start_tensor)
        slice_layer.set_input(2, size_tensor)

        outputs.append(slice_layer.get_output(0))

    return outputs
