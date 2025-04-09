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

import inspect
import logging
import os
import sys

import numpy as np
import tensorrt as trt

from paddle.tensorrt.util import TensorRTConfigManager, TensorRTConstantManager

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from tensorrt import INetworkDefinition, ITensor

from paddle.base.log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)
from paddle.base.libpaddle.pir import (
    get_attrs_map_json,
    get_inputs_type_json,
    get_outputs_type_json,
)

version = trt.__version__
version_list = list(map(int, version.split('.')))


def has_dynamic_shape(shape):
    return any(s == -1 for s in shape)


def append_ones(network, input, name, num_prepend_ones):
    layer = network.add_shuffle(input)

    if has_dynamic_shape(input.shape):
        input_shape_layer = network.add_shape(input)
        prepend_shape_layer = network.add_constant(
            (num_prepend_ones,), np.ones((num_prepend_ones,), dtype=np.int32)
        )
        reshape_dim_layer = network.add_concatenation(
            [prepend_shape_layer.get_output(0), input_shape_layer.get_output(0)]
        )
        reshape_dim_layer.axis = 0
        layer.set_input(1, reshape_dim_layer.get_output(0))
        if name is not None:
            set_layer_name(input_shape_layer, [name[0], "input_shape_layer"])
            set_layer_name(
                prepend_shape_layer, [name[0], "prepend_shape_layer"]
            )
            set_layer_name(reshape_dim_layer, [name[0], "reshape_dim_layer"])
    else:
        layer.reshape_dims = (1,) * num_prepend_ones + tuple(input.shape)

    if name is not None:
        set_layer_name(layer, name)
    return layer.get_output(0)


def broadcast(network, a, b, a_name, b_name, paddle_op, preset_diff=0):
    a_shape = tuple(a.shape)
    b_shape = tuple(b.shape)

    diff = len(a_shape) - len(b_shape) - preset_diff
    if diff > 0:
        b = append_ones(network, b, [paddle_op.name(), b_name], diff)
    elif diff < 0:
        a = append_ones(network, a, [paddle_op.name(), a_name], -diff)

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
    ), f"Unable to found plugin creator with name {plugin_name}"
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
    from paddle.tensorrt.util import support_fp32_mix_precision

    weight_shape = paddle_op.operands()[1].source().shape
    input_shape = paddle_op.operands()[0].source().shape

    weight_tensor = inputs[1]
    input_tensor = inputs[0]
    if type(inputs[1]) == trt.Weights:
        weight_tensor = network.add_constant(weight_shape, inputs[1])
        set_layer_name(weight_tensor, paddle_op)
        weight_tensor = weight_tensor.get_output(0)
    if type(inputs[0]) == trt.Weights:
        input_tensor = network.add_constant(input_shape, inputs[0])
        set_layer_name(input_tensor, paddle_op)
        input_tensor = input_tensor.get_output(0)
    lhs_val, rhs_val = broadcast(
        network,
        input_tensor,
        weight_tensor,
        "input_tensor_broadcast",
        "weight_tensor_broadcast",
        paddle_op,
    )
    layer = network.add_elementwise(lhs_val, rhs_val, op_type)
    set_layer_name(layer, paddle_op)
    support_fp32_mix_precision(paddle_op.name(), layer)
    return layer.get_output(0)


# Create and add 1D constant layer
def add_1D_constant_layer(
    network, data, dtype=np.int32, is_scalar=False, name=None
):
    if not isinstance(data, list):
        data = [data]
    shape = () if is_scalar else (len(data),)
    constant_data = np.array(data, dtype=dtype)
    constant_layer = network.add_constant(shape, constant_data)
    set_layer_name(constant_layer, name)
    return constant_layer.get_output(0)


# Create and add ND constant layer
def add_constant_layer(network, data, shape, dtype=np.int32, name=None):
    constant_data = np.array(data, dtype=dtype)
    constant_data = np.resize(constant_data, shape)
    constant_layer = network.add_constant(shape, constant_data)
    set_layer_name(constant_layer, name)
    return constant_layer.get_output(0)


# Create an constant layer with shape_tensor and value
def fill_constant_layer(
    network, shape_tensor, tensor_rank, data, trt_dtype, name=None
):
    fill_layer = network.add_fill(
        trt.Dims([tensor_rank]), trt.FillOperation.LINSPACE
    )
    np_dtype = map_trt_dtype(trt_dtype)
    fill_layer.set_input(0, shape_tensor)
    fill_layer.set_input(
        1, add_1D_constant_layer(network, data, np_dtype, is_scalar=True)
    )
    beta = [0] * tensor_rank
    fill_layer.set_input(
        2, add_1D_constant_layer(network, beta, np_dtype, is_scalar=False)
    )
    set_layer_name(fill_layer, name)
    return fill_layer.get_output(0)


def trt_expand(network, input, rank, shape_tensor, shape_rank, name=None):
    def process_names(name, layer_name):
        if name is not None:
            return [name[0], layer_name]
        else:
            return None

    if rank < shape_rank:
        one_rank_tensor = add_1D_constant_layer(
            network,
            [1] * (shape_rank - rank),
            name=process_names(name, "one_rank_tensor"),
        )
        in_shape_tensor = trt_shape(
            network, input, name=process_names(name, "in_shape_tensor")
        )
        itensors = [one_rank_tensor, in_shape_tensor]
        input_shape_tensor = trt_concat(
            network, itensors, name=process_names(name, "input_shape_tensor")
        )
    else:
        input_shape_tensor = trt_shape(
            network, input, name=process_names(name, "input_shape_tensor")
        )

    new_input_tensor = trt_reshape(
        network,
        input,
        input_shape_tensor,
        process_names(name, "new_input_tensor"),
        True,
    )

    start = [0] * shape_rank
    starts_tensor = add_1D_constant_layer(
        network, start, name=process_names(name, "starts_tensor")
    )
    one_tensor = add_1D_constant_layer(
        network, 1, name=process_names(name, "one_tensor")
    )
    sizes_tensor = trt_max(
        network,
        input_shape_tensor,
        shape_tensor,
        name=process_names(name, "sizes_tensor"),
    )
    input_sub_tensor = trt_sub(
        network,
        input_shape_tensor,
        one_tensor,
        name=process_names(name, "input_sub_tensor"),
    )
    strides_tensor = trt_min(
        network,
        one_tensor,
        input_sub_tensor,
        name=process_names(name, "strides_tensor"),
    )

    slice_layer = network.add_slice(
        new_input_tensor, start, [0] * len(start), [0] * len(start)
    )
    slice_layer.set_input(1, starts_tensor)
    slice_layer.set_input(2, sizes_tensor)
    slice_layer.set_input(3, strides_tensor)
    set_layer_name(slice_layer, name)

    return slice_layer.get_output(0)


# Concat not make rank changed
def trt_concat(network, inputs, axis=0, name=None):
    concat_layer = network.add_concatenation(inputs=inputs)
    if axis != 0:
        concat_layer.axis = axis
    set_layer_name(concat_layer, name)
    return concat_layer.get_output(0)


def trt_cast(network, input, dtype, name=None):
    identity_layer = network.add_identity(input)
    identity_layer.set_output_type(0, dtype)
    identity_layer.get_output(0).dtype = dtype
    set_layer_name(identity_layer, name)
    return identity_layer.get_output(0)


def trt_shape(
    network: INetworkDefinition, input: ITensor, name=None
) -> ITensor:
    """
    Add a IShapeLayer to get the shape of `input` ITensor.
    This includes a workaround that casting the shape result(int64) from TRT10 back to int32.
    Many existing paddle op kernels only support input shape tensor as int32
    , to make TRT op more compatible with other paddle op, we cast back to int32.
    NOTE: please remove this workaround when all paddle op supports shape tensor in int64
    """
    shape_layer = network.add_shape(input)
    set_layer_name(shape_layer, name)
    if version_list[0] >= 10:  # trt_version >=10
        # workaround
        if name is not None:
            name = [name[0], "trt_cast"]
        return trt_cast(
            network, shape_layer.get_output(0), trt.int32, name=name
        )
    return shape_layer.get_output(0)


def trt_reshape(network, input, new_shape, name=None, is_shape_tensor=False):
    reshape_layer = network.add_shuffle(input)
    if is_shape_tensor:
        reshape_layer.set_input(1, new_shape)
    else:
        reshape_layer.reshape_dims = new_shape
    if name is not None:
        if isinstance(name, list):
            set_layer_name(reshape_layer, name)
        else:
            reshape_layer.name = name
    return reshape_layer.get_output(0)


# resize shape tensor's shape to 1dim
def resize_to_1d(network, shape_tensor, name=None):
    if shape_tensor is None:
        return shape_tensor
    if len(shape_tensor.shape) > 1:
        # shape_tensor need 1-dim in trt
        shape_tensor_layer = network.add_shuffle(shape_tensor)
        numel = 1
        for ele in shape_tensor.shape:
            numel *= ele
        shape_tensor_layer.reshape_dims = [numel]
        set_layer_name(shape_tensor_layer, name)
        shape_tensor = shape_tensor_layer.get_output(0)
    return shape_tensor


# Get element tensor of 1D shape tensor
def get_shape_tensor_element(network, x, index, is_scalar=False, name=None):
    assert (
        index >= 0
    ), f"The index should be greater or equal than 0, but got {index}"
    index_tensor_name = [name[0], "index_tensor"] if name is not None else None
    index_tensor = add_1D_constant_layer(
        network, index, is_scalar=is_scalar, name=index_tensor_name
    )
    gather_layer = network.add_gather(input=x, indices=index_tensor, axis=0)
    if name is not None:
        set_layer_name(gather_layer, [name[0], "gather_layer"])
    shape_tensor = resize_to_1d(network, gather_layer.get_output(0), name=name)
    return shape_tensor


def trt_less(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.LESS)
    set_layer_name(layer, name)
    return layer.get_output(0)


def trt_sum(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.SUM)
    set_layer_name(layer, name)
    return layer.get_output(0)


def trt_max(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.MAX)
    set_layer_name(layer, name)
    return layer.get_output(0)


def trt_sub(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.SUB)
    set_layer_name(layer, name)
    return layer.get_output(0)


def trt_min(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.MIN)
    set_layer_name(layer, name)
    return layer.get_output(0)


def trt_div(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.DIV)
    set_layer_name(layer, name)
    return layer.get_output(0)


def trt_floor_div(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.FLOOR_DIV)
    set_layer_name(layer, name)
    return layer.get_output(0)


def trt_equal(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.EQUAL)
    set_layer_name(layer, name)
    return layer.get_output(0)


def trt_gather(network, input, indices, axis=0, name=None):
    if name is not None:
        name = [name[0], "indices_tensor"]
    indices_tensor = add_1D_constant_layer(network, indices, name=name)
    gather_layer = network.add_gather(input, indices_tensor, axis)
    set_layer_name(gather_layer, name)
    result = gather_layer.get_output(0)
    return result


def trt_prod(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.PROD)
    set_layer_name(layer, name)
    return layer.get_output(0)


def trt_pow(network, a, b, name=None):
    layer = network.add_elementwise(a, b, trt.ElementWiseOperation.POW)
    set_layer_name(layer, name)
    return layer.get_output(0)


def cast_tensor(network, input_tensor, dtype, name=None):
    layer = network.add_identity(input_tensor)
    layer.set_output_type(0, dtype)
    set_layer_name(layer, name)
    return layer.get_output(0)


def build_start_tensor(network, rank, axis_tensor, offset, name=None):
    # Create indices_tensor [0, 1, ..., rank-1]
    indices = np.arange(rank, dtype=np.int32)
    indices_name = [name[0], "indices_tensor"] if name is not None else None
    indices_tensor = network.add_constant([rank], indices)
    set_layer_name(indices_tensor, indices_name)
    indices_tensor = indices_tensor.get_output(0)

    # Create mask: mask = (indices == axis_tensor)
    mask_name = [name[0], "mask"] if name is not None else None
    mask = network.add_elementwise(
        indices_tensor, axis_tensor, trt.ElementWiseOperation.EQUAL
    )
    set_layer_name(mask, mask_name)
    mask = mask.get_output(0)
    mask_int = cast_tensor(
        network,
        mask,
        trt.int32,
        name=[name[0], "mask_int"] if name is not None else None,
    )

    # Calculate start_tensor = mask_int * offset
    start_tensor = network.add_elementwise(
        mask_int, offset, trt.ElementWiseOperation.PROD
    )
    set_layer_name(start_tensor, name)
    start_tensor = start_tensor.get_output(0)

    return start_tensor


def build_size_tensor(
    network, rank, axis_tensor, size_value, input_shape_tensor, name=None
):
    # Create indices_tensor [0, 1, ..., rank-1]
    indices = np.arange(rank, dtype=np.int32)
    indices_name = [name[0], 'indices_tensor'] if name is not None else None
    indices_tensor = network.add_constant([rank], indices)
    set_layer_name(indices_tensor, indices_name)
    indices_tensor = indices_tensor.get_output(0)

    # Create mask: mask = (indices == axis_tensor)
    mask_name = [name[0], 'mask'] if name is not None else None
    mask = network.add_elementwise(
        indices_tensor, axis_tensor, trt.ElementWiseOperation.EQUAL
    )
    set_layer_name(mask, mask_name)
    mask = mask.get_output(0)
    mask_int = cast_tensor(
        network,
        mask,
        trt.int32,
        name=[name[0], "mask_int"] if name is not None else None,
    )

    # Create ones_tensor
    ones_name = [name[0], 'ones_tensor'] if name is not None else None
    ones_tensor = network.add_constant([rank], np.ones([rank], dtype=np.int32))
    set_layer_name(ones_tensor, ones_name)
    ones_tensor = ones_tensor.get_output(0)

    # Calculate inverse_mask = ones_tensor - mask_int
    inverse_mask_name = [name[0], 'inverse_mask'] if name is not None else None
    inverse_mask = network.add_elementwise(
        ones_tensor, mask_int, trt.ElementWiseOperation.SUB
    )
    set_layer_name(inverse_mask, inverse_mask_name)
    inverse_mask = inverse_mask.get_output(0)

    # Calculate size_tensor = mask_int * size_value + inverse_mask * input_shape_tensor
    size_value_broadcast_name = (
        [name[0], 'size_value_broadcast'] if name is not None else None
    )
    size_value_broadcast = network.add_elementwise(
        mask_int, size_value, trt.ElementWiseOperation.PROD
    )
    set_layer_name(size_value_broadcast, size_value_broadcast_name)
    size_value_broadcast = size_value_broadcast.get_output(0)

    input_shape_broadcast_name = (
        [name[0], 'input_shape_broadcast'] if name is not None else None
    )
    input_shape_broadcast = network.add_elementwise(
        inverse_mask, input_shape_tensor, trt.ElementWiseOperation.PROD
    )
    set_layer_name(input_shape_broadcast, input_shape_broadcast_name)
    input_shape_broadcast = input_shape_broadcast.get_output(0)

    size_tensor = network.add_elementwise(
        size_value_broadcast,
        input_shape_broadcast,
        trt.ElementWiseOperation.SUM,
    )
    set_layer_name(size_tensor, name)
    size_tensor = size_tensor.get_output(0)

    return size_tensor


# convert trt_dtype to numpy dtype
def map_trt_dtype(trt_dtype):
    dtype_map = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT8: np.int8,
        trt.DataType.BOOL: bool,
    }
    if trt_dtype in dtype_map:
        return dtype_map[trt_dtype]
    else:
        raise TypeError(f"Unsupported trt_dtype: {trt_dtype}")


# Reduce the given tensor in the TensorRT network to a scalar
def trt_reduce_to_scalar(network, tensor, dtype=trt.int32, name=None):
    if len(tensor.shape) == 0:
        return tensor
    axes = 0
    for i in range(len(tensor.shape)):
        axes |= 1 << i
    reduce_layer = network.add_reduce(
        tensor, trt.ReduceOperation.SUM, axes, keep_dims=False
    )
    if name is not None:
        set_layer_name(reduce_layer, [name[0], 'reduce_layer'])
        scalar_name = name
    scalar = trt_cast(
        network, reduce_layer.get_output(0), dtype, name=scalar_name
    )
    return scalar


def convert_conv2d(network, paddle_op, inputs):
    from paddle.tensorrt.util import (
        RefitManager,
        RefitRole,
        support_fp32_mix_precision,
    )

    bias = None
    if (
        paddle_op.name() == "pd_op.conv2d"
        or paddle_op.name() == "pd_op.depthwise_conv2d"
    ):
        input_tensor, filter = inputs
    elif (
        paddle_op.name() == "pd_op.conv2d_transpose"
        or paddle_op.name() == "pd_op.depthwise_conv2d_transpose"
    ):
        if len(inputs) == 3:
            input_tensor, filter, output_size = inputs
        elif len(inputs) == 2:
            input_tensor, filter = inputs
            output_size = None
        else:
            raise ValueError("Invalid number of inputs for conv2d_transpose")
    if paddle_op.name() == "pd_op.fused_conv2d_add_act":
        input_tensor, filter, bias, _ = inputs
    input_shape = paddle_op.operands()[0].source().shape
    filter_shape = paddle_op.operands()[1].source().shape

    if len(filter_shape) != 4:
        raise ValueError(
            f"filter's dims size should be 4, but got {len(filter_shape)}"
        )

    n_output = filter_shape[0]
    n_input = filter_shape[1]
    filter_h = filter_shape[2]
    filter_w = filter_shape[3]

    paddings = paddle_op.attrs().get("paddings", [0, 0])
    stride = paddle_op.attrs().get("strides", [1, 1])
    dilation = paddle_op.attrs().get("dilations", [1, 1])
    groups = paddle_op.attrs().get("groups", 1)

    if has_dynamic_shape(input_shape):
        assert (
            input_shape[1] != -1
        ), "Channel dim can't be dynamic for transpose convolution."

    output_padding = paddle_op.attrs().get("output_padding", [0, 0])
    padding_algorithm = paddle_op.attrs().get("padding_algorithm", "EXPLICIT")
    if padding_algorithm == "VALID":
        paddings = [0] * len(paddings)

    nv_ksize = trt.DimsHW(filter_h, filter_w)
    nv_dilations = trt.DimsHW(dilation[0], dilation[1])
    nv_strides = trt.DimsHW(stride[0], stride[1])

    pre_paddings = [0, 0]
    post_paddings = [0, 0]

    if isinstance(filter, trt.Weights):
        weight_filter = filter
    else:
        weight_filter = trt.Weights()

    if len(paddings) == 2:
        pre_paddings[0] = paddings[0]
        pre_paddings[1] = paddings[1]
        post_paddings[0] = paddings[0]
        post_paddings[1] = paddings[1]
    elif len(paddings) == 4:
        pre_paddings[0] = paddings[0]
        pre_paddings[1] = paddings[2]
        post_paddings[0] = paddings[1]
        post_paddings[1] = paddings[3]
    else:
        raise ValueError(f"Unsupported paddings size: {len(paddings)}")

    if paddle_op.name() == "pd_op.fused_conv2d_add_act":
        constant_manager = TensorRTConstantManager()
        bias_source_op = paddle_op.operands()[2].source().get_defining_op()
        if bias_source_op.name() == "builtin.parameter":
            bias_name = bias_source_op.attrs()['parameter_name']
        elif bias_source_op.name() == "builtin.constant":
            bias_np = bias_source_op.attrs()['value']
        else:
            raise ValueError(
                f"Unsupported bias source op: {bias_source_op.name()}"
            )
        bias_np = constant_manager.get_constant_value(bias_name)
        bias_weights = trt.Weights(bias_np)
        layer = network.add_convolution_nd(
            input=input_tensor,
            num_output_maps=n_output,
            kernel_shape=nv_ksize,
            kernel=weight_filter,
            bias=bias_weights,
        )
    elif (
        paddle_op.name() == "pd_op.conv2d"
        or paddle_op.name() == "pd_op.depthwise_conv2d"
    ):
        layer = network.add_convolution_nd(
            input=input_tensor,
            num_output_maps=n_output,
            kernel_shape=nv_ksize,
            kernel=weight_filter,
            bias=None,
        )
    elif (
        paddle_op.name() == "pd_op.conv2d_transpose"
        or paddle_op.name() == "pd_op.depthwise_conv2d_transpose"
    ):
        layer = network.add_deconvolution_nd(
            input=input_tensor,
            num_output_maps=n_input * groups,
            kernel_shape=nv_ksize,
            kernel=weight_filter,
            bias=None,
        )

    if isinstance(filter, trt.ITensor):
        layer.set_input(1, filter)
    layer.stride_nd = nv_strides
    layer.pre_padding = pre_paddings

    if output_padding:
        post_paddings[0] -= output_padding[0]
        post_paddings[1] -= output_padding[1]

    if post_paddings[0] < 0 or post_paddings[1] < 0:
        raise ValueError("The value PostPadding should be >= 0.")

    layer.post_padding = post_paddings
    layer.num_groups = groups

    if padding_algorithm == "SAME":
        layer.padding_mode = trt.PaddingMode.SAME_UPPER
        nv_dilations = trt.DimsHW(1, 1)

    layer.dilation_nd = nv_dilations
    set_layer_name(layer, paddle_op)
    support_fp32_mix_precision(paddle_op.name(), layer)

    filter_param = paddle_op.operands()[1].source()
    filter_name = filter_param.get_defining_op().attrs()['parameter_name']
    refit_manager = RefitManager()
    refit_manager.set_mapping(filter_name, filter_name, RefitRole.CONSTANT)

    return layer.get_output(0)


def convert_conv3d(network, paddle_op, inputs):
    from paddle.tensorrt.util import (
        RefitManager,
        RefitRole,
    )

    input_tensor, filter = inputs
    filter_shape = paddle_op.operands()[1].source().shape

    n_output = filter_shape[0]
    n_input = filter_shape[1]
    filter_d = filter_shape[2]
    filter_h = filter_shape[3]
    filter_w = filter_shape[4]

    if isinstance(filter, trt.Weights):
        weight_filter = filter
    else:
        weight_filter = trt.Weights()

    groups = paddle_op.attrs().get("groups", 1)
    dilations = paddle_op.attrs().get("dilations", [1, 1, 1])
    strides = paddle_op.attrs().get("strides", [1, 1, 1])
    paddings = paddle_op.attrs().get("paddings", [0, 0, 0])
    padding_algorithm = paddle_op.attrs().get("padding_algorithm", "EXPLICIT")
    # for conv3d_transpose
    output_padding = paddle_op.attrs().get("output_padding", [])

    nv_ksize = trt.Dims3(filter_d, filter_h, filter_w)
    nv_dilations = trt.Dims3(dilations[0], dilations[1], dilations[2])
    nv_strides = trt.Dims3(strides[0], strides[1], strides[2])
    nv_pre_paddings = trt.Dims3(paddings[0], paddings[1], paddings[2])

    if paddle_op.name() == "pd_op.conv3d":
        layer = network.add_convolution_nd(
            input=input_tensor,
            num_output_maps=n_output,
            kernel_shape=nv_ksize,
            kernel=weight_filter,
            bias=None,
        )
    elif paddle_op.name() == "pd_op.conv3d_transpose":
        layer = network.add_deconvolution_nd(
            input=input_tensor,
            num_output_maps=n_input * groups,
            kernel_shape=nv_ksize,
            kernel=weight_filter,
            bias=None,
        )
    layer.stride_nd = nv_strides
    layer.pre_padding = nv_pre_paddings

    nv_post_paddings = trt.Dims3(paddings[0], paddings[1], paddings[2])
    if output_padding:
        nv_post_paddings[0] -= output_padding[0]
        nv_post_paddings[1] -= output_padding[1]
        nv_post_paddings[2] -= output_padding[2]

        if (
            nv_post_paddings[0] < 0
            or nv_post_paddings[1] < 0
            or nv_post_paddings[2] < 0
        ):
            raise ValueError(
                "The value in conv3d_transpose's PostPadding should be >= 0."
            )
    if isinstance(filter, trt.ITensor):
        layer.set_input(1, filter)

    layer.post_padding = nv_post_paddings
    layer.num_groups = groups

    if padding_algorithm == "SAME":
        layer.padding_mode = trt.PaddingMode.SAME_UPPER

    layer.dilation_nd = nv_dilations
    set_layer_name(layer, paddle_op)
    filter_param = paddle_op.operands()[1].source()
    filter_name = filter_param.get_defining_op().attrs()['parameter_name']
    refit_manager = RefitManager()
    refit_manager.set_mapping(filter_name, filter_name, RefitRole.CONSTANT)

    return layer.get_output(0)


def get_input_constant_value(paddle_op, inputs, input_index):
    input_op = paddle_op.operands()[input_index].source().get_defining_op()
    constant_manager = TensorRTConstantManager()
    if input_op.name() == "builtin.constant":
        return constant_manager.get_constant_value(
            input_op.attrs()["value"]
        ).tolist()
    elif input_op.name() == "pd_op.full_int_array":
        return input_op.attrs()["value"]
    elif input_op.name() == "pd_op.full":
        return [input_op.attrs()["value"]]
    else:
        return None


def add_reduce_layer(network, paddle_op, inputs, op_type):
    input_tensor = inputs[0]
    axis = get_input_constant_value(paddle_op, inputs, 1)
    input_shape = paddle_op.operands()[0].source().shape
    keepdim = paddle_op.attrs()["keepdim"]
    if network.has_implicit_batch_dimension:
        assert (
            axis != 0
        ), "can't reduce on axis == 0 when network has implicit batch dimension"
    output_shape = []
    if len(axis) == 0:
        axis = list(range(len(input_shape)))
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] = len(input_shape) + axis[i]
    layer = network.add_reduce(
        input_tensor,
        op_type,
        axes=get_axes_for_reduce_op(axis),
        keep_dims=keepdim,
    )
    set_layer_name(layer, paddle_op)
    layer.get_output(0).dtype = layer.get_input(0).dtype
    return layer.get_output(0)


def add_cast_reduce_layer(network, paddle_op, inputs, op_type):
    input_tensor = inputs[0]
    cast_layer = network.add_identity(input_tensor)
    set_layer_name(cast_layer, paddle_op)
    cast_layer.set_output_type(0, trt.int32)
    cast_layer.get_output(0).dtype = trt.int32

    axis = paddle_op.attrs().get("axis")
    input_shape = paddle_op.operands()[0].source().shape
    keepdim = paddle_op.attrs()["keepdim"]
    if network.has_implicit_batch_dimension:
        assert (
            axis != 0
        ), "can't reduce on axis == 0 when network has implicit batch dimension"
    output_shape = []
    if len(axis) == 0:
        axis = list(range(len(input_shape)))
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] = len(input_shape) + axis[i]
    layer = network.add_reduce(
        cast_layer.get_output(0),
        op_type,
        axes=get_axes_for_reduce_op(axis),
        keep_dims=keepdim,
    )
    set_layer_name(layer, paddle_op)
    layer.set_output_type(0, trt.bool)
    layer.get_output(0).dtype = cast_layer.get_output(0).dtype
    return layer.get_output(0)


def fix_negative_indices(network, input_shape, indices, name=None):
    rank = len(input_shape.shape)
    zero_tensor_name = [name[0], 'zero_tensor'] if name else None
    zero_tensor = add_1D_constant_layer(
        network, [0] * rank, name=zero_tensor_name
    )
    minus_one_tensor_name = [name[0], 'minus_one_tensor'] if name else None
    minus_one_tensor = add_1D_constant_layer(
        network, [-1] * rank, name=minus_one_tensor_name
    )

    min_indices_zero_name = [name[0], 'min_indices_zero'] if name else None
    min_indices_zero = trt_min(
        network, indices, zero_tensor, name=min_indices_zero_name
    )
    sign_name = [name[0], 'sign'] if name else None
    sign = trt_max(network, min_indices_zero, minus_one_tensor, name=sign_name)
    sub_name = [name[0], 'sub'] if name else None
    sub = trt_prod(network, sign, input_shape, name=sub_name)
    fixed_indices = trt_sub(network, indices, sub, name=name)
    return fixed_indices


def trt_unsqueeze(network, input_tensor, axes, name=None):
    input_shape_name = [name[0], 'input_shape'] if name else None
    input_shape = network.add_shape(input_tensor)
    set_layer_name(input_shape, input_shape_name)
    input_shape = input_shape.get_output(0)

    axis_set = set(axes)

    subscripts = list(range(len(input_tensor.shape)))

    for axis in sorted(axis_set):
        subscripts.insert(axis, len(input_tensor.shape))

    one_tensor_name = [name[0], 'one_tensor'] if name else None
    one_tensor = network.add_constant((1,), np.array([1], dtype=np.int32))
    set_layer_name(one_tensor, one_tensor_name)
    one_tensor = one_tensor.get_output(0)
    extended_shape_name = [name[0], 'extended_shape'] if name else None
    extended_shape = network.add_concatenation(
        [input_shape, one_tensor],
    )
    set_layer_name(extended_shape, extended_shape_name)
    extended_shape = extended_shape.get_output(0)

    gather_layer_name = [name[0], 'gather_layer'] if name else None
    gather_layer = network.add_gather(
        extended_shape,
        network.add_constant(
            (len(subscripts),), np.array(subscripts, dtype=np.int32)
        ).get_output(0),
        axis=0,
    )
    set_layer_name(gather_layer, gather_layer_name)
    new_shape_tensor = gather_layer.get_output(0)

    reshaped_tensor = network.add_shuffle(input_tensor)
    reshaped_tensor.set_input(1, new_shape_tensor)
    set_layer_name(reshaped_tensor, name)

    return reshaped_tensor.get_output(0)


def squeeze_trt(network, input_tensor, axes, name=None):
    input_shape_name = [name[0], 'input_shape'] if name else None
    input_shape = network.add_shape(input_tensor)
    set_layer_name(input_shape, input_shape_name)
    input_shape = input_shape.get_output(0)
    input_shape = input_tensor.shape
    all_dims = list(range(len(input_shape)))
    remaining_dims = [dim for dim in all_dims if dim not in axes]

    input_shape_tensor_name = [name[0], 'input_shape_tensor'] if name else None
    input_shape_tensor = network.add_shape(input_tensor)
    set_layer_name(input_shape_tensor, input_shape_tensor_name)
    input_shape_tensor = input_shape_tensor.get_output(0)

    remaining_dims_tensor_name = (
        [name[0], 'remaining_dims_tensor'] if name else None
    )
    remaining_dims_tensor = network.add_constant(
        (len(remaining_dims),), np.array(remaining_dims, dtype=np.int32)
    )
    set_layer_name(remaining_dims_tensor, remaining_dims_tensor_name)
    remaining_dims_tensor = remaining_dims_tensor.get_output(0)

    new_shape_tensor_name = [name[0], 'new_shape_tensor'] if name else None
    new_shape_tensor = network.add_gather(
        input_shape_tensor, remaining_dims_tensor, axis=0
    )
    set_layer_name(new_shape_tensor, new_shape_tensor_name)
    new_shape_tensor = new_shape_tensor.get_output(0)
    reshape_layer = network.add_shuffle(input_tensor)
    reshape_layer.set_input(1, new_shape_tensor)
    set_layer_name(reshape_layer, name)
    return reshape_layer.get_output(0)


def unary_op_converter(network, paddle_op, inputs):
    from paddle.tensorrt import PrecisionMode

    ops_type_map = {
        "pd_op.sqrt": [trt.UnaryOperation.SQRT],
        "pd_op.sqrt_": [trt.UnaryOperation.SQRT],
        "pd_op.floor": [trt.UnaryOperation.FLOOR],
        "pd_op.exp": [trt.UnaryOperation.EXP],
        "pd_op.abs": [trt.UnaryOperation.ABS],
        "pd_op.abs_": [trt.UnaryOperation.ABS],
        "pd_op.sin": [trt.UnaryOperation.SIN],
        "pd_op.cos": [trt.UnaryOperation.COS],
        "pd_op.sinh": [trt.UnaryOperation.SINH],
        "pd_op.cosh": [trt.UnaryOperation.COSH],
        "pd_op.asinh": [trt.UnaryOperation.ASINH],
        "pd_op.acosh": [trt.UnaryOperation.ACOSH],
        "pd_op.atanh": [trt.UnaryOperation.ATANH],
        "pd_op.ceil": [trt.UnaryOperation.CEIL],
        "pd_op.reciprocal": [trt.UnaryOperation.RECIP],
        "pd_op.erf": [trt.UnaryOperation.ERF],
        "pd_op.sign": [trt.UnaryOperation.SIGN],
        "pd_op.round": [trt.UnaryOperation.ROUND],
        "pd_op.logical_not": [trt.UnaryOperation.NOT],
        "pd_op.rsqrt": [trt.UnaryOperation.SQRT, trt.UnaryOperation.RECIP],
        "pd_op.tan": [trt.UnaryOperation.TAN],
        "pd_op.asin": [trt.UnaryOperation.ASIN],
        "pd_op.acos": [trt.UnaryOperation.ACOS],
        "pd_op.atan": [trt.UnaryOperation.ATAN],
    }

    input_tensor = inputs[0]
    layer = None
    org_type = input_tensor.dtype

    trt_type_mapping = {
        trt.DataType.INT8: trt.int8,
        trt.DataType.INT32: trt.int32,
    }

    trt_manager = TensorRTConfigManager()
    precision_mode = trt_manager.get_precision_mode()

    need_cast = org_type in [trt.DataType.INT8, trt.DataType.INT32]
    if need_cast:
        identity_layer = network.add_identity(input_tensor)
        if precision_mode == PrecisionMode.FP32:
            identity_layer.set_output_type(0, trt.float32)
        else:
            identity_layer.set_output_type(0, trt.float16)
        set_layer_name(identity_layer, paddle_op)
        input_tensor = identity_layer.get_output(0)

    if paddle_op.name() in ops_type_map:
        for trt_op in ops_type_map[paddle_op.name()]:
            layer = network.add_unary(input_tensor, trt_op)
            set_layer_name(layer, paddle_op)
            input_tensor = layer.get_output(0)
    else:
        raise NotImplementedError(
            f"Unsupported unary operation: {paddle_op.name()}"
        )
    if need_cast:
        restore_layer = network.add_identity(input_tensor)
        restore_layer.set_output_type(0, trt_type_mapping[org_type])
        set_layer_name(restore_layer, paddle_op)
        input_tensor = restore_layer.get_output(0)

    return input_tensor


# get the length of the specified axis for input_tensor
def get_axis_length(network, input_tensor, axis, is_scalar=False, name=None):
    input_shape = input_tensor.shape
    if input_shape[axis] >= 0:
        output_tensor = add_1D_constant_layer(
            network, input_shape[axis], is_scalar=is_scalar, name=name
        )
    else:
        shape_name = [name[0], 'dynamic_shape'] if name else None
        dynamic_shape = trt_shape(network, input_tensor, name=shape_name)
        output_tensor = get_shape_tensor_element(
            network, dynamic_shape, axis, is_scalar, name=name
        )
    return output_tensor


def WithFp16():
    from paddle.tensorrt import PrecisionMode

    trt_manager = TensorRTConfigManager()
    precision_mode = trt_manager.get_precision_mode()
    enable_fp16 = False
    if precision_mode == PrecisionMode.FP16:
        enable_fp16 = True
    # TODO(lizexu123) WithInt8() and use_dla are not yet implemented
    return enable_fp16


def set_layer_name(layer, second_param):
    """
    Sets standardized names for converter output layers following the format: `<id>_<pd_op>-><layerName>(<inputIds>)`

    Naming Rule:
        Format: <sequence_number>_<paddle_op_name>-><layer_variable_name>(<comma_separated_input_ids>)
        Components:
            - sequence_number: Output tensor's unique ID from layer
            - paddle_op_name: Name of source Paddle operator
            - layer_variable_name: Variable name referencing the layer in code
            - input_ids: Input tensor IDs from preceding layers

    Args:
        layer (ILayer): Target layer to name
        second_param: Context-dependent parameter:
            - For non-public functions: paddle_op (op object)
            - For public functions: [paddle_op_name (str), layer_var_name (str)] list
            - When name=None in public functions: Enables nested handling
    """
    if second_param is not None:
        if isinstance(second_param, list):
            # Handling for public function layer
            op_name, layer_var_name = second_param
        else:
            # Handling for layer
            op_name = second_param.name()
            layer_var_name = None
            if op_name is not None:
                # Retrieve the name of the variable that refers to the layer
                for (
                    var_name,
                    var_val,
                ) in inspect.currentframe().f_back.f_locals.items():
                    if var_val is layer:
                        layer_var_name = var_name
                        break

        # Retrieve the input id of the layer
        if op_name is not None and layer_var_name is not None:
            input_ids = []
            i = 0
            while (input_tensor := layer.get_input(i)) is not None:
                input_name = input_tensor.name
                if "Unnamed Layer" in input_name:
                    input_id = input_name.split("*")[1].split(")")[0].strip()
                else:
                    input_id = input_name
                input_ids.append(input_id)
                i += 1

            # Retrieve the output id of the layer
            output_name = layer.get_output(0).name
            if "Unnamed Layer" in output_name:
                sequence_number = (
                    output_name.split("*")[1].split(")")[0].strip()
                )
            else:
                sequence_number = output_name

            formatted_name = (
                f"{sequence_number}_"
                f"{op_name}->"
                f"{layer_var_name}"
                f"({', '.join(input_ids)})"
            )
            layer.name = formatted_name


def generic_plugin_converter(network, paddle_op, inputs, extra_attrs=None):
    op_name = paddle_op.name()

    if extra_attrs is not None:
        attrs_map_info = get_attrs_map_json(extra_attrs)
    else:
        attrs_map_info = get_attrs_map_json(paddle_op)

    input_type_info = get_inputs_type_json(paddle_op)
    output_type_info = get_outputs_type_json(paddle_op)

    plugin_fields = [
        trt.PluginField(
            "op_name",
            np.array(list(op_name), dtype=np.bytes_),
            trt.PluginFieldType.CHAR,
        ),
        trt.PluginField(
            "attrs_map_info",
            np.array(list(attrs_map_info), dtype=np.bytes_),
            trt.PluginFieldType.CHAR,
        ),
        trt.PluginField(
            "inputs_type_info",
            np.array(list(input_type_info), dtype=np.bytes_),
            trt.PluginFieldType.CHAR,
        ),
        trt.PluginField(
            "outputs_type_info",
            np.array(list(output_type_info), dtype=np.bytes_),
            trt.PluginFieldType.CHAR,
        ),
    ]

    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)

    plugin_name = "pir_generic_plugin"
    plugin_version = "1"
    plugin = get_trt_plugin(
        plugin_name, plugin_field_collection, plugin_version
    )

    layer = network.add_plugin_v2(inputs, plugin)
    return layer
