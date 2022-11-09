#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import numpy as np
from ....framework import IrNode
from ....framework import Operator

_weight_supported_quantizable_op_type = [
    'conv2d',
    'depthwise_conv2d',
    'conv2d_transpose',
    'mul',
    'matmul',
    'matmul_v2',
]

_act_supported_quantizable_op_type = [
    "pool2d",
    "elementwise_add",
    "concat",
    "softmax",
    "argmax",
    "transpose",
    "equal",
    "gather",
    "greater_equal",
    "greater_than",
    "less_equal",
    "less_than",
    "mean",
    "not_equal",
    "reshape",
    "reshape2",
    "dropout",
    "bilinear_interp",
    "nearest_interp",
    "trilinear_interp",
    "slice",
    "squeeze",
    "elementwise_sub",
    "mul",
    "matmul",
    "relu",
    "relu6",
    "leaky_relu",
    "tanh",
    "swish",
    "transpose",
    "transpose2",
    "sigmoid",
    "pad2d",
    "flatten",
    "flatten2",
    "batch_norm",
    "layer_norm",
    "matmul_v2",
    "split",
    "flatten_contiguous_range",
    "squeeze2",
    "nearest_interp_v2",
    "bilinear_interp",
    "bilinear_interp_v2",
    "fill_constant_batch_size_like",
    "arg_max",
    "abs",
    "assign",
    "cast",
    "clip",
    "box_coder",
    "crop",
    "cumsum",
    "elementwise_mul",
    "elementwise_pow",
    "expand_v2",
    "fill_any_like",
    "fill_constant",
    "gelu",
    "hard_sigmoid",
    "hard_swish",
    "instance_norm",
    "lookup_table",
    "lookup_table_v2",
    "norm",
    "p_norm",
    "pad3d",
    "pow",
    "prelu",
    "reduce_mean",
    "unsqueeze",
    "unsqueeze2",
    "logical_and",
    "logical_not",
    "meshgrid",
    "roi_align",
    "strided_slice",
    "where",
    "grid_sampler",
    "tile",
    "group_norm",
    "reduce_sum",
    "square",
    "softplus",
    "shuffle_channel",
    "reduce_max",
    "scale",
]

QUANT_SUPPORTED_OP_TYPE_LIST = list(
    set(
        _weight_supported_quantizable_op_type
        + _act_supported_quantizable_op_type
    )
)

_out_scale_op_list = QUANT_SUPPORTED_OP_TYPE_LIST

_channelwise_quant_axis1_ops = [
    'conv2d_transpose',
    'mul',
    'matmul',
    'matmul_v2',
]

# list op real input and output names, to avoid processing input such as AxisTensor.
_op_real_in_out_name = {
    "conv2d": [["Input", "Filter"], ["Output"]],
    "depthwise_conv2d": [["Input", "Filter"], ["Output"]],
    "conv2d_transpose": [["Input", "Filter"], ["Output"]],
    "mul": [["X", "Y"], ["Out"]],
    "matmul": [["X", "Y"], ["Out"]],
    "matmul_v2": [["X", "Y"], ["Out"]],
    "pool2d": [["X"], ["Out"]],
    "elementwise_add": [["X", "Y"], ["Out"]],
    "concat": [["X"], ["Out"]],
    "softmax": [["X"], ["Out"]],
    "argmax": [["X"], ["Out"]],
    "transpose": [["X"], ["Out"]],
    "equal": [["X", "Y"], ["Out"]],
    "gather": [["X"], ["Out"]],
    "greater_equal": [["X", "Y"], ["Out"]],
    "greater_than": [["X", "Y"], ["Out"]],
    "less_equal": [["X", "Y"], ["Out"]],
    "less_than": [["X", "Y"], ["Out"]],
    "mean": [["X"], ["Out"]],
    "not_equal": [["X", "Y"], ["Out"]],
    "reshape": [["X"], ["Out"]],
    "reshape2": [["X"], ["Out"]],
    "transpose2": [["X"], ["Out"]],
    "bilinear_interp": [["X"], ["Out"]],
    "nearest_interp": [["X"], ["Out"]],
    "trilinear_interp": [["X"], ["Out"]],
    "slice": [["Input"], ["Out"]],
    "squeeze": [["X"], ["Out"]],
    "elementwise_sub": [["X", "Y"], ["Out"]],
    "relu": [["X"], ["Out"]],
    "relu6": [["X"], ["Out"]],
    "leaky_relu": [["X"], ["Out"]],
    "prelu": [["X", "Alpha"], ["Out"]],
    "tanh": [["X"], ["Out"]],
    "swish": [["X"], ["Out"]],
    "dropout": [["X"], ["Out"]],
    "batch_norm": [["X"], ["Y"]],
    "layer_norm": [["X"], ["Y"]],
    "sigmoid": [["X"], ["Out"]],
    "elementwise_mul": [["X", "Y"], ["Out"]],
    "elementwise_pow": [["X", "Y"], ["Out"]],
    "hard_swish": [["X"], ["Out"]],
    "hard_sigmoid": [["X"], ["Out"]],
    "gru": [["Input", "Weight"], ["Hidden"]],
    "lstm": [["Input", "Weight"], ["Hidden"]],
    "pad2d": [["X"], ["Out"]],
    "pad3d": [["X"], ["Out"]],
    "flatten": [["X"], ["Out"]],
    "flatten2": [["X"], ["Out"]],
    "unsqueeze2": [["X"], ["Out"]],
    "unsqueeze2": [["X"], ["Out"]],
    "flatten_contiguous_range": [["X"], ["Out"]],
    "split": [["X"], ["Out"]],
    "squeeze2": [["X"], ["Out"]],
    "nearest_interp_v2": [["X"], ["Out"]],
    "bilinear_interp": [["X"], ["Out"]],
    "bilinear_interp_v2": [["X"], ["Out"]],
    "fill_constant_batch_size_like": [["Input"], ["Out"]],
    "arg_max": [["X"], ["Out"]],
    "abs": [["X"], ["Out"]],
    "assign": [["X"], ["Out"]],
    "cast": [["X"], ["Out"]],
    "clip": [["X"], ["Out"]],
    "box_coder": [["PriorBox"], ["OutputBox"]],
    "crop": [["X"], ["Out"]],
    "cumsum": [["X"], ["Out"]],
    "expand_v2": [["X"], ["Out"]],
    "fill_any_like": [["X"], ["Out"]],
    "fill_constant": [[], ["Out"]],
    "gelu": [["X"], ["Out"]],
    "instance_norm": [["X"], ["Y"]],
    "lookup_table": [["W", "Ids"], ["Out"]],
    "lookup_table_v2": [["W", "Ids"], ["Out"]],
    "norm": [["X"], ["Norm"]],
    "p_norm": [["X"], ["Out"]],
    "pow": [["X"], ["Out"]],
    "reduce_mean": [["X"], ["Out"]],
    "stack": [["X"], ["Y"]],
    "top_k_v2": [["X"], ["Out", "Indices"]],
    "logical_and": [["X", "Y"], ["Out"]],
    "logical_not": [["X"], ["Out"]],
    "meshgrid": [["X"], ["Out"]],
    "roi_align": [["X", "ROIs"], ["Out"]],
    "strided_slice": [["Input"], ["Out"]],
    "where": [["Condition", "X", "Y"], ["Out"]],
    "grid_sampler": [["X", "Grid"], ["Output"]],
    "tile": [["X"], ["Out"]],
    "group_norm": [["X"], ["Y", "Mean", "Variance"]],
    "reduce_sum": [["X"], ["Out"]],
    "square": [["X"], ["Out"]],
    "softplus": [["X"], ["Out"]],
    "shuffle_channel": [["X"], ["Out"]],
    "reduce_max": [["X"], ["Out"]],
    "scale": [["X"], ["Out"]],
}


def _get_op_input_var_names(op):
    """
    Get the input var names of the op.
    Args:
        op(IrNode, Operator): the input op.
    Returns:
        input_var_names or None.
    """
    assert isinstance(
        op, (IrNode, Operator)
    ), "The input op should be IrNode or Operator."
    var_names = []
    op_name = op.name() if isinstance(op, IrNode) else op.type
    if op_name not in _op_real_in_out_name:
        return []

    name_list = _op_real_in_out_name[op_name][0]
    for name in name_list:
        var_name = op.input(name)
        if isinstance(var_name, list):
            var_names.extend(var_name)
        else:
            var_names.append(var_name)
    return var_names


def _get_op_output_var_names(op):
    """ """
    assert isinstance(
        op, (IrNode, Operator)
    ), "The input op should be IrNode or Operator."
    var_names = []
    op_name = op.name() if isinstance(op, IrNode) else op.type
    if op_name not in _op_real_in_out_name:
        return []

    name_list = _op_real_in_out_name[op_name][1]
    for name in name_list:
        var_name = op.output(name)
        if isinstance(var_name, list):
            var_names.extend(var_name)
        else:
            var_names.append(var_name)
    return var_names


def _get_input_name_index(op, input_var_name):
    """Get the input name and index of the var_name in the op"""
    assert isinstance(
        op, (IrNode, Operator)
    ), "The input op should be IrNode or Operator."
    op_name = op.name() if isinstance(op, IrNode) else op.type
    if op_name not in _op_real_in_out_name:
        return None

    res = None
    for argname in _op_real_in_out_name[op_name][0]:
        var_names = op.input(argname)
        for index, name in enumerate(var_names):
            if name == input_var_name:
                res = (argname, index)
    return res


def _get_output_name_index(op, output_var_name):
    """Get the output name and index of the var_name in the op"""
    assert isinstance(
        op, (IrNode, Operator)
    ), "The input op should be IrNode or Operator."
    op_name = op.name() if isinstance(op, IrNode) else op.type
    if op_name not in _op_real_in_out_name:
        return None

    name_list = _op_real_in_out_name[op_name][1]
    res = None
    for name in name_list:
        var_name = op.output(name)
        for index, val in enumerate(var_name):
            if val == output_var_name:
                res = (name, index)
    return res


def load_variable_data(scope, var_name):
    '''
    Load variable value from scope
    '''
    var_node = scope.find_var(var_name)
    assert var_node is not None, "Cannot find " + var_name + " in scope."
    return np.array(var_node.get_tensor())


def set_variable_data(scope, place, var_name, np_value):
    '''
    Set the value of var node by name, if the node exits,
    '''
    assert isinstance(
        np_value, np.ndarray
    ), 'The type of value should be numpy array.'
    var_node = scope.find_var(var_name)
    if var_node is not None:
        tensor = var_node.get_tensor()
        tensor.set(np_value, place)


def quant_tensor(x, scale, quant_axis=0, weight_bits=8, onnx_format=False):
    # symmetry quant
    def _clip(x, scale):
        x[x > scale] = scale
        x[x < -scale] = -scale
        return x

    bnt = (1 << (weight_bits - 1)) - 1
    if isinstance(scale, list) and len(scale) == 1:
        scale = scale[0]
    if isinstance(scale, list):
        assert quant_axis in [0, 1], 'quant_axis should be 0 or 1 for now.'
        for i, s in enumerate(scale):
            if s == 0.0:
                s = 1e-8
            if quant_axis == 0:
                if onnx_format:
                    x[i] = np.round(x[i] / s * bnt)
                    x[i] = np.clip(x[i], -bnt - 1, bnt)
                else:
                    x[i] = _clip(x[i], s)
                    x[i] = x[i] / s * bnt
            else:
                if onnx_format:
                    x[:, i] = np.round(x[:, i] / s * bnt)
                    x[:, i] = np.clip(x[:, i], -bnt - 1, bnt)
                else:
                    x[:, i] = _clip(x[:, i], s)
                    x[:, i] = x[:, i] / s * bnt
    else:
        scale = 1e-8 if scale == 0.0 else scale
        if onnx_format:
            x = np.round(x / scale * bnt)
            x = np.clip(x, -bnt - 1, bnt)
        else:
            x = _clip(x, scale)
            x = x / scale * bnt
    return x


def dequant_tensor(x, scale, quant_axis=0, weight_bits=8):
    assert quant_axis in [0, 1], 'quant_axis should be 0 or 1 for now.'
    bnt = (1 << (weight_bits - 1)) - 1
    if isinstance(scale, list):
        for i, s in enumerate(scale):
            if s == 0.0:
                s = 1e-8
            if quant_axis == 0:
                x[i] = x[i] * s / bnt
            else:
                x[:, i] = x[:, i] * s / bnt
    else:
        scale = 1e-8 if scale == 0.0 else scale
        x = x * scale / bnt
    return x


def bias_correction_w(x, x_quant, scale_v, quant_axis, weight_bits=8):
    '''
    Bias correction for weight
    '''
    eps = 1e-8
    bnt = (1 << (weight_bits - 1)) - 1
    x_dequant = x_quant.copy()
    if isinstance(scale_v, list):
        if quant_axis == 0:
            for i, s in enumerate(scale_v):
                x_dequant[i] = x_dequant[i] * s / bnt
            quant_bias = x - x_dequant
            mean_bias = quant_bias.reshape(quant_bias.shape[0], -1).mean(-1)
            std_orig = x.reshape(x.shape[0], -1).std(-1)
            std_quant = x_dequant.reshape(x_dequant.shape[0], -1).std(-1)
            std_bias = std_orig / (std_quant + eps)
        else:
            for i, s in enumerate(scale_v):
                x_dequant[:, i] = x_quant[:, i] * s / bnt
            quant_bias = x - x_dequant
            mean_bias = np.array(
                [quant_bias[:, i].mean() for i in range(quant_bias.shape[1])]
            )
            std_orig = np.array([x[:, i].std() for i in range(x.shape[1])])
            std_quant = np.array(
                [x_dequant[:, i].std() for i in range(x_dequant.shape[1])]
            )
            std_bias = std_orig / (std_quant + eps)
    else:
        x_dequant = x_quant * scale_v / bnt
        mean_bias = (x - x_dequant).mean()
        std_bias = x.std() / (x_dequant.std() + eps)
    if mean_bias.ndim == 1:
        std_bias = np.resize(std_bias, x.shape)
        mean_bias = np.resize(mean_bias, x.shape)

    x_dequant = (mean_bias + x_dequant) * std_bias
    quantized_param_v = quant_tensor(
        x_dequant, scale_v, quant_axis, weight_bits
    )
    return quantized_param_v


def stable_sigmoid(x):
    sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
    return sig


def calculate_quant_cos_error(orig_tensor, qdq_tensor):
    cos_sim = np.inner(orig_tensor.flatten(), qdq_tensor.flatten()) / (
        np.linalg.norm(orig_tensor.flatten())
        * np.linalg.norm(qdq_tensor.flatten())
    )
    return cos_sim


def move_persistable_var_to_global_block(program):
    # Move sub blocks persistable var to global block
    global_block = program.global_block()
    for _op in global_block.ops:
        if _op.type == "while":
            _block_id = _op.attr("sub_block").id
            _block = program.block(_block_id)
            persistables = []
            for _name, _var in _block.vars.items():
                if _var.persistable:
                    global_block._clone_variable(_var)
                    persistables.append(_name)
            for _name in persistables:
                _block._remove_var(_name)
            persistables.extend(_op.input('X'))
            _op.desc.set_input("X", persistables)


def l2_loss(gt, pred):
    return ((gt - pred) ** 2).mean()


class tqdm:
    def __init__(self, total, bar_format='Loading|{bar}', ncols=80):
        self.total = total
        self.bar_format = bar_format
        self.ncols = ncols
        self.n = 0

    def update(self, n=1):
        self.n += n
        a = "=" * round((self.n / self.total) * self.ncols)
        b = " " * (self.ncols - len(a))
        prefix = self.bar_format.split('|')[0]
        sys.stderr.write(
            "\r{}|{}=>{}| {}/{}".format(prefix, a, b, self.n, self.total)
        )
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.write('\n')
