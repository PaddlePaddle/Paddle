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

import copy

_FLOPS_COMPUTE_FUNC_MAP = {}


def prod(s):
    p = 1
    for v in s:
        p *= v
    return p


def flops(op_type: str, input_shapes: dict, attrs: dict) -> int:
    """
    count FLOPs for operation.

    Args:
        op_type (str): the type of operation.
        input_shapes (dict): the shapes of inputs.
        attrs (dict): the attributes of the operation.

    Returns:
        the total FLOPs of the operation.
    """

    if op_type not in _FLOPS_COMPUTE_FUNC_MAP:
        return 0
    else:
        func = _FLOPS_COMPUTE_FUNC_MAP[op_type]
        try:
            flops = func(input_shapes, attrs)
        except Exception as e:
            return 0
        return flops


def register_flops(op_type):
    """
    register flops computation function for operation.
    """

    def register(func):
        global _FLOPS_COMPUTE_FUNC_MAP
        _FLOPS_COMPUTE_FUNC_MAP[op_type] = func
        return func

    return register


@register_flops("c_embedding")
def _c_embedding_flops(input_shapes, attrs):
    """FLOPs computation for c_embedding op.
    For c_embedding(input):
        equation: flops = 0
    """
    return 0


@register_flops("conv2d")
def _conv2d_flops(input_shapes, attrs):
    """FLOPs computation for conv2d op.
    For conv2d(input,filter):
        active_elements = batch_size * numel(output)
        conv_flops = 2 * macs_per_position_conv * active_elements
        bias_flops = out_channels * active_elements
        equation: flops = conv_flops + bias_flops
    """

    bias = (
        input_shapes.get('Bias')[0]
        if len(input_shapes.get('Bias')) > 0
        else None
    )
    input = input_shapes.get('Input')[0]
    weight = input_shapes.get('Filter')[0]

    padding = attrs.get('paddings')
    stride = attrs.get('strides')
    dilation = attrs.get('dilations')
    groups = attrs.get('groups')

    batch_size = input[0]
    in_channels = input[1]
    out_channels = weight[0]
    kernel_dims = list(weight[2:])
    input_dims = list(input[2:])
    length = len(input_dims)

    paddings = (
        padding
        if isinstance(padding, list)
        else [
            padding,
        ]
        * length
    )
    strides = (
        stride
        if isinstance(stride, list)
        else [
            stride,
        ]
        * length
    )
    dilations = (
        dilation
        if isinstance(dilation, list)
        else [
            dilation,
        ]
        * length
    )

    output_dims = []
    for idx, input_dim in enumerate(input_dims):
        output_dim = (
            input_dim
            + 2 * paddings[idx]
            - (dilations[idx] * (kernel_dims[idx] - 1) + 1)
        ) // strides[idx] + 1
        output_dims.append(output_dim)
    filters_per_channel = out_channels // groups
    macs_conv_per_position = (
        prod(kernel_dims) * in_channels * filters_per_channel
    )
    active_elements = batch_size * prod(output_dims)
    overall_conv_macs = macs_conv_per_position * active_elements
    overall_conv_flops = 2 * overall_conv_macs

    overall_bias_flops = 0

    if bias is not None:
        overall_bias_flops = out_channels * active_elements

    return overall_conv_flops + overall_bias_flops


@register_flops("dropout")
def _dropout_flops(input_shapes, attrs):
    """FLOPs computation for dropout op.
    For dropout(input):
        equation: flops = 0
    """
    return 0


def _elementwise_flops_compute(input_shapes, attrs):
    input_x = input_shapes.get("X")[0]
    input_y = input_shapes.get("Y")[0]
    dim_x = len(input_x)
    dim_y = len(input_y)
    dim_output = max(dim_x, dim_y)
    output = []
    for i in range(dim_output):
        in_x = input_x[dim_x - 1 - i] if i < dim_x else 1
        in_y = input_y[dim_y - 1 - i] if i < dim_y else 1
        output.append(max(in_x, in_y))
    return prod(output)


@register_flops("elementwise_add")
def _elementwise_add_flops(input_shapes, attrs):
    """FLOPs computation for elementwise_add op.
    For elementwise_add(input,other):
        input_shapes = [shape_of_input, shape_of_other]
        shape_of_input = [dim1, dim2, dim3 ...]
        shape_of_other = [odim1, odim2, odim3...]
        equation: flops = max(dim1, odim1) * max(dim2, odim2) * max()...
    """
    return _elementwise_flops_compute(input_shapes, attrs)


@register_flops("elementwise_mul")
def _elementwise_mul_flops(input_shapes, attrs):
    """FLOPs computation for elementwise_mul op.
    For elementwise_mul(input,other):
        input_shapes = [shape_of_input, shape_of_other]
        shape_of_input = [dim1, dim2, dim3 ...]
        shape_of_other = [odim1, odim2, odim3...]
        equation: flops = max(dim1, odim1) * max(dim2, odim2)* max()...
    """
    return _elementwise_flops_compute(input_shapes, attrs)


@register_flops("elementwise_div")
def _elementwise_div_flops(input_shapes, attrs):
    """FLOPs computation for elementwise_div op.
    For elementwise_div(input,other):
        input_shapes = [shape_of_input, shape_of_other]
        shape_of_input = [dim1, dim2, dim3 ...]
        shape_of_other = [odim1, odim2, odim3...]
        equation: flops = max(dim1,odim1)*max(dim2,odim2)*max()...
    """
    return _elementwise_flops_compute(input_shapes, attrs)


@register_flops("gelu")
def _gelu_flops(input_shapes, attrs):
    """FLOPs computation for gelu op.
    For gelu(input):
        equation: flops = 5 * (numel)total number of elements in the input tensor.
    """
    input = input_shapes.get('X')[0]
    return prod(input) * 5


@register_flops("layer_norm")
def _layer_norm_flops(input_shapes, attrs):
    """FLOPs computation for layer_norm op.
    For layer_norm(input):
        equation:
        1): WITHOUT epsilon flops = 7 * (numel)total number of elements in the input tensor.
        2): WITH epsilon flops = 8 * (numel)total number of elements in the input tensor.
    """
    input = input_shapes.get('X')[0]
    flops = prod(input) * 7
    if attrs.get('epsilon'):
        flops += prod(input)
    return flops


@register_flops("matmul")
def _matmul_flops(input_shapes, attrs):
    """FLOPs computation for matmul op.
    For matmul(input,other):
        input_shapes = [shape_of_input, shape_of_other]
        shape_of_input =                  [dim1,dim2 ...dim_n_1,dim_n]  length:n
        shape_of_other = [odim1,odim2 ... odim(n-m)... odim_m_1,dim_m]  length:m
        suppose n > m and dim_n = odim_m_1:
        shape_of_output = [dim1, dim2 ... max(dim(n-m), odim(n-m)), max(dim(n-m+1), odim(n-m+1)) ... dim_n_1, dim_m]
        equation: flops = 2 * numel(output) * dim_n
    """

    x_shape = copy.deepcopy(
        input_shapes.get("X", input_shapes.get("x", [[0]]))[0]
    )
    y_shape = copy.deepcopy(
        input_shapes.get("Y", input_shapes.get("y", [[0]]))[0]
    )
    if attrs.get('transpose_X') or attrs.get('transpose_x'):
        x_shape[-1], x_shape[-2] = x_shape[-2], x_shape[-1]

    if attrs.get('transpose_Y') or attrs.get('transpose_y'):
        y_shape[-1], y_shape[-2] = y_shape[-2], y_shape[-1]
    dim_x = len(x_shape)
    dim_y = len(y_shape)
    output_len = max(dim_x, dim_y)
    output_shape = []

    for idx in range(output_len, 2, -1):
        x_idx = x_shape[dim_x - idx] if idx <= dim_x else 1
        y_idx = y_shape[dim_y - idx] if idx <= dim_y else 1
        output_shape.append(max(x_idx, y_idx))

    macs = prod(output_shape) * x_shape[-2] * x_shape[-1] * y_shape[-1]
    return 2 * macs


@register_flops("matmul_v2")
def _matmul_v2_flops(input_shapes, attrs):
    """FLOPs computation for matmul_v2 op.
    For matmul_v2(input,other):
        input_shapes = [shape_of_input, shape_of_other]
        shape_of_input =                   [dim1, dim2 ...dim_n_1, dim_n] length:n
        shape_of_other = [odim1, odim2 ... odim(n-m) ... odim_m_1, dim_m] length:m
        suppose n > m and dim_n = odim_m_1:
        shape_of_output = [dim1, dim2 ... max(dim(n-m), odim(n-m)), max(dim(n-m+1), odim(n-m+1))...dim_n_1, dim_m]
        equation: flops = 2 * numel(outputs) * dim_n
    """
    x_shape = copy.deepcopy(input_shapes.get('X')[0])
    y_shape = copy.deepcopy(input_shapes.get('Y')[0])
    if attrs.get('trans_x'):
        x_shape[-1], x_shape[-2] = x_shape[-2], x_shape[-1]
    if attrs.get('trans_y'):
        y_shape[-1], y_shape[-2] = y_shape[-2], y_shape[-1]
    dim_x = len(x_shape)
    dim_y = len(y_shape)
    output_len = max(dim_x, dim_y)
    output_shape = []
    for idx in range(output_len, 2, -1):
        x_idx = x_shape[dim_x - idx] if idx <= dim_x else 1
        y_idx = y_shape[dim_y - idx] if idx <= dim_y else 1
        output_shape.append(max(x_idx, y_idx))

    macs = prod(output_shape) * x_shape[-2] * x_shape[-1] * y_shape[-1]
    return 2 * macs


def _relu_class_flops(input_shapes, attrs):
    """FLOPs computation for relu_like ops.
    For elu/leaky_relu/prelu/relu/relu6/silu (input):
        equation: flops = (numel)total number of elements in the input tensor.
    """
    input = input_shapes.get('X')[0]
    return prod(input)


@register_flops("elu")
def _elu_flops(input_shapes, attrs):
    return _relu_class_flops(input_shapes, attrs)


@register_flops("leaky_relu")
def _leaky_relu_flops(input_shapes, attrs):
    return _relu_class_flops(input_shapes, attrs)


@register_flops("prelu")
def _prelu_flops(input_shapes, attrs):
    return _relu_class_flops(input_shapes, attrs)


@register_flops("relu")
def _relu_flops(input_shapes, attrs):
    return _relu_class_flops(input_shapes, attrs)


@register_flops("relu6")
def _relu6_flops(input_shapes, attrs):
    return _relu_class_flops(input_shapes, attrs)


@register_flops("silu")
def _silu_flops(input_shapes, attrs):
    return _relu_class_flops(input_shapes, attrs)


@register_flops("reshape2")
def _reshape2_flops(input_shapes, attrs):
    """FLOPs computation for reshape2 op.
    For reshape2(input):
        equation: flops = 0
    """
    return 0


@register_flops("softmax")
def _softmax_flops(input_shapes, attrs):
    """FLOPs computation for softmax op.
    For softmax(input):
        equation: flops = 3 * (numel)total number of elements in the input tensor.
    """
    input = input_shapes.get('X')[0]
    return prod(input) * 3


@register_flops("transpose2")
def _transpose2_flops(input_shapes, attrs):
    """FLOPs computation for transpose2 op.
    For transpose2(input):
        equation: flops = 0
    """
    return 0


@register_flops("pool")
def _pool_flops(input_shapes, attrs):
    """FLOPs computation for pool op.
    For pool(input):
        equation: flops = (numel)total number of elements in the input tensor.
    """
    input = input_shapes.get('X')[0]
    return prod(input)
