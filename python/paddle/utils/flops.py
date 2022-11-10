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
        return func(input_shapes, attrs)


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
def _dropout_flops(input_shapes, attrs):
    return 0


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
def _relu_flops(input_shapes, attrs):
    return prod(input_shapes.get('X')[0])
