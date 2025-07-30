#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from ...base.framework import IrNode, Operator
from .quant_config import SUPPORT_QUANTIZATION_OP_DICT

_channelwise_quant_axis1_ops = [
    'conv2d_transpose',
    'mul',
    'matmul',
    'matmul_v2',
]


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
    if op_name not in SUPPORT_QUANTIZATION_OP_DICT:
        return []

    name_list = SUPPORT_QUANTIZATION_OP_DICT[op_name][0]
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
    if op_name not in SUPPORT_QUANTIZATION_OP_DICT:
        return []

    name_list = SUPPORT_QUANTIZATION_OP_DICT[op_name][1]
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
    if op_name not in SUPPORT_QUANTIZATION_OP_DICT:
        return None

    res = None
    for argname in SUPPORT_QUANTIZATION_OP_DICT[op_name][0]:
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
    if op_name not in SUPPORT_QUANTIZATION_OP_DICT:
        return None

    name_list = SUPPORT_QUANTIZATION_OP_DICT[op_name][1]
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
    tensor = np.array(var_node.get_tensor())
    if tensor.shape == ():
        return tensor.reshape(1)
    else:
        return tensor


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
        assert quant_axis in [-1, 0, 1], 'quant_axis should be 0 or 1 for now.'
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
        sys.stderr.write(f"\r{prefix}|{a}=>{b}| {self.n}/{self.total}")
        sys.stderr.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.write('\n')
