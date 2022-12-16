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
        input_shapes = [shape_of_input, shape_of_ohther]
        shape_of_input = [dim1, dim2, dim3 ...]
        shape_of_other = [odim1, odim2, odim3...]
        equation: flops = max(dim1, odim1) * max(dim2, odim2) * max()...
    """
    return _elementwise_flops_compute(input_shapes, attrs)


@register_flops("elementwise_mul")
def _elementwise_mul_flops(input_shapes, attrs):
    """FLOPs computation for elementwise_mul op.
    For elementwise_mul(input,other):
        input_shapes = [shape_of_input, shape_of_ohther]
        shape_of_input = [dim1, dim2, dim3 ...]
        shape_of_other = [odim1, odim2, odim3...]
        equation: flops = max(dim1, odim1) * max(dim2, odim2)* max()...
    """
    return _elementwise_flops_compute(input_shapes, attrs)


@register_flops("elementwise_div")
def _elementwise_div_flops(input_shapes, attrs):
    """FLOPs computation for elementwise_div op.
    For elementwise_div(input,other):
        input_shapes = [shape_of_input, shape_of_ohther]
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
        input_shapes = [shape_of_input, shape_of_ohther]
        shape_of_input =                  [dim1,dim2 ...dim_n_1,dim_n]  length:n
        shape_of_other = [odim1,odim2 ... odim(n-m)... odim_m_1,dim_m]  length:m
        suppose n > m and dim_n = odim_m_1:
        shape_of_output = [dim1, dim2 ... max(dim(n-m), odim(n-m)), max(dim(n-m+1), odim(n-m+1)) ... dim_n_1, dim_m]
        equation: flops = 2 * numel(output) * dim_n
    """

    x_shape = input_shapes.get("X", input_shapes.get("x", [[0]]))[0]
    y_shape = input_shapes.get("Y", input_shapes.get("y", [[0]]))[0]
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
        input_shapes = [shape_of_input, shape_of_ohther]
        shape_of_input =                  [dim1, dim2 ...dim_n_1, dim_n] length:n
        shape_of_other = [odim1, odim2 ... odim(n-m) ... odim_m_1, dim_m] length:m
        suppose n > m and dim_n = odim_m_1:
        shape_of_output = [dim1, dim2 ... max(dim(n-m), odim(n-m)), max(dim(n-m+1), odim(n-m+1))...dim_n_1, dim_m]
        equation: flops = 2 * numel(output) * dim_n
    """
    x_shape = input_shapes.get('X')[0]
    y_shape = input_shapes.get('Y')[0]
    if attrs.get('trans_x') is not None:
        x_shape[-1], x_shape[-2] = x_shape[-2], x_shape[-1]
    if attrs.get('trans_y') is not None:
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
    """FLOPs computation for relu op.
    For relu(input):
        equation: flops = (numel)total number of elements in the input tensor.
    """
    return prod(input_shapes.get('X')[0])


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
