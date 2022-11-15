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

from numpy import prod

_FLOPS_COMPUTE_FUNC_MAP = {}


def flops(op_type: str, input_shapes: tuple, **attrs) -> int:
    """
    count flops for operation.

    Args:
        op_type (str): the type of operation.
        input_shapes (tuple|list|dict): the shapes of inputs.
        attrs (dict): the attributes of the operation.

    Returns:
        the total flops of the operation.
    """

    if isinstance(input_shapes, dict):
        input_shapes_tmp = []
        for k in sorted(input_shapes.keys()):
            input_shapes_tmp += input_shapes[k]
        input_shapes = input_shapes_tmp

    if op_type not in _FLOPS_COMPUTE_FUNC_MAP:
        return 0
    else:
        func = _FLOPS_COMPUTE_FUNC_MAP[op_type]
        return func(input_shapes, **attrs)


def register_flops(op_type):
    """
    register flops computation function for operation.
    """

    def register(func):
        global _FLOPS_COMPUTE_FUNC_MAP
        _FLOPS_COMPUTE_FUNC_MAP[op_type] = func
        return func

    return register


@register_flops("dropout")
def _dropout_flops(input_shapes, **attrs):
    return 0


@register_flops("relu")
def _relu_flops(input_shapes, **attrs):
    return prod(input_shapes[0])


def _elementwise_flops_compute(input_shapes, **attrs):
    input_x = input_shapes[0]
    input_y = input_shapes[1]
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
def _elementwise_add_flops(input_shapes, **attrs):
    return _elementwise_flops_compute(input_shapes, **attrs)


@register_flops("elementwise_mul")
def _elementwise_mul_flops(input_shapes, **attrs):
    return _elementwise_flops_compute(input_shapes, **attrs)


@register_flops("elementwise_div")
def _elementwise_mul_flops(input_shapes, **attrs):
    return _elementwise_flops_compute(input_shapes, **attrs)


@register_flops("gelu")
def _gelu_flops(input_shapes, **attrs):
    return prod(input_shapes[0]) * 5


@register_flops("layer_norm")
def _layer_norm_flops(input_shapes, **attrs):
    input = input_shapes[2]
    flops = prod(input) * 7
    if attrs['epsilon']:
        flops += prod(input)
    return flops


@register_flops("matmul")
def _matmul_flops(input_shapes, **attrs):
    x_shape = input_shapes[0]
    y_shape = input_shapes[1]
    if attrs['transpose_X']:
        x_shape[-1], x_shape[-2] = x_shape[-2], x_shape[-1]
    if attrs['transpose_Y']:
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
def _matmul_v2_flops(input_shapes, **attrs):
    x_shape = input_shapes[0]
    y_shape = input_shapes[1]
    if attrs['trans_x']:
        x_shape[-1], x_shape[-2] = x_shape[-2], x_shape[-1]
    if attrs['trans_y']:
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


@register_flops("relu")
def _relu_flops(input_shapes, **attrs):
    return prod(input_shapes[0])


@register_flops("reshape2")
def _reshape2_flops(input_shapes, **attrs):
    return 0


@register_flops("soft_max")
def _soft_max_flops(input_shapes, **attrs):
    return prod(input_shapes[0]) * 3


@register_flops("transpose2")
def _transpose2_flops(input_shapes, **attrs):
    return 0
