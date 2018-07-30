#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import numpy as np

from paddle.fluid.framework import default_main_program, default_startup_program
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import unique_name
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
from .. import core
from ..framework import Variable
from ..executor import global_scope

_QUANTIZABLE_OP_TYPES = ['conv2d', 'depthwise_conv2d', 'mul']


def _quantized_var_name(var_name):
    """
    Return quantized variable name for the input `var_name`.
    """
    return "%s.quantized" % (var_name)


def _dequantized_var_name(var_name):
    """
    Return dequantized variable name for the input `var_name`.
    """
    return "%s.dequantized" % (var_name)


def _quantized_scale_name(var_name):
    """
    Return quantized variable name for the input `var_name`.
    """
    return "%s.scale" % (var_name)


def _original_var_name(var_name):
    """
    Return the original variable name.
    """
    if var_name.endswith('.quantized.dequantized'):
        return var_name[:-len('.quantized.dequantized')]
    if var_name.endswith('.quantized'):
        return var_name[:-len('.quantized')]
    if var_name.endswith('.dequantized'):
        return var_name[:-len('.dequantized')]
    if var_name.endswith('.scale'):
        return var_name[:-len('.scale')]
    else:
        return var_name


def _is_float(v):
    return isinstance(v, float) or isinstance(v, np.float32)


def quant(x, scale, num_bits):
    y = np.round(x / scale * ((1 << (num_bits - 1)) - 1))
    return y


class QuantizeTranspiler(object):
    """
    TODO(qingqing): add comments
    """

    def __init__(self,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max',
                 window_size=10000):
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.weight_quantize_type = weight_quantize_type
        self.activation_quantize_type = activation_quantize_type
        self.window_size = window_size
        self.helper = LayerHelper(self.__class__.__name__)
        self.fake_quant_op_types = ['fake_quantize']
        self.fake_dequant_op_types = ['fake_dequantize_max_abs']
        self.is_inference = None

    def transpile(self, program):
        """
        TODO(qingqing): add comments
        """
        self.is_inference = False
        program = default_main_program() if program is None else program
        startup_program = default_startup_program()

        # marked the variable which has been quantized and dequantized.
        dequanted_vars = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]
        grad_op_types = ['%s_grad' % (type) for type in _QUANTIZABLE_OP_TYPES]

        params = [p.name for p in program.global_block()._iter_parameters()]

        def _transpile_forward(block, op):
            idx = block.ops.index(op)
            block_id = block.idx
            # insert quant op and dequant op
            for name in op.input_arg_names:
                if name in dequanted_vars[block_id]:
                    dequant_var = dequanted_vars[block_id][name]
                else:
                    var = block.var(name)
                    quant_bits = self.weight_bits if var.name in params \
                                 else self.activation_bits
                    quant_type = self.weight_quantize_type if var.name \
                        in params else self.activation_quantize_type

                    quant_var, scale_var = self._insert_quant_op(
                        block, idx, var, quant_bits, quant_type)
                    dequant_var = self._insert_dequant_op(
                        block, idx + 1, quant_var, scale_var, quant_bits)
                    dequanted_vars[block_id][name] = dequant_var
                # rename the forward op inputs
                op.rename_input(name, dequant_var.name)

        def _transpile_backward(block, op):
            block_id = block.idx
            no_dequanted_input_vars = True
            for name in op.input_arg_names:
                if name in dequanted_vars[block_id]:
                    dequant_var = dequanted_vars[block_id][name]
                    op.rename_input(name, dequant_var.name)
                    no_dequanted_input_vars = False
            if no_dequanted_input_vars:
                raise ValueError("There is no dequanted inputs for op %s." %
                                 (op.type))

        for block in program.blocks:
            ops = list(block.ops)
            block_id = block.idx
            for op in ops:
                # rewrite the forward ProgramDes
                if op.type in _QUANTIZABLE_OP_TYPES:
                    _transpile_forward(block, op)
                # rename the backward op inputs
                if op.type in grad_op_types:
                    _transpile_backward(block, op)

    def freeze_program(self, program, place, freeze_weight=True, scope=None):
        """
        TODO(qingqing): add comments
        """
        self.is_inference = True
        scope = global_scope() if scope is None else scope
        program = default_main_program() if program is None else program

        persistable_vars = [
            v.name
            for v in filter(lambda var: var.persistable, program.list_vars())
        ]
        op_in_rename_map = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]
        op_out_rename_map = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]
        var_scale_map = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]

        def _remove_fake_quant_and_dequant_op(block, op):
            idx = block.ops.index(op)
            block_id = block.idx
            k = op.output('Out')[0]
            v = op.input('X')[0]
            if v not in op_in_rename_map[block_id]:
                op_in_rename_map[block_id][k] = v
            else:
                op_in_rename_map[block_id][k] = op_in_rename_map[block_id][v]
            block._remove_op(idx)

        def _insert_post_dequant_op(block, op):
            idx = block.ops.index(op)
            block_id = block.idx
            max_range = None
            scale_var = None
            for name in op.input_arg_names:
                if name in op_in_rename_map[block_id]:
                    op.rename_input(name, op_in_rename_map[block_id][name])

                scale_v = var_scale_map[block_id][_original_var_name(name)]
                if _original_var_name(name) in persistable_vars:
                    param_range = (1 << (self.weight_bits - 1)) - 1
                    act_range = (1 << (self.activation_bits - 1)) - 1
                    assert _is_float(scale_v)
                    max_range = param_range * act_range / scale_v
                else:
                    assert isinstance(scale_v, Variable)
                    scale_var = var_scale_map[block_id][_original_var_name(
                        name)]

            if len(op.output_arg_names) != 1:
                raise ValueError("Only support one output, but op %s has"
                                 " more than one output." % (op.type))
            out_var = block.var(op.output_arg_names[0])
            dequant_var = block.create_var(
                name=_dequantized_var_name(out_var.name),
                type=out_var.type,
                shape=out_var.shape,
                dtype=out_var.dtype)
            # insert fake_dequantize_op
            dequant_op = block._insert_op(
                idx + 1,
                type="fake_dequantize_max_abs",
                attrs={'range': float(max_range)},
                inputs={"X": out_var,
                        'Scale': scale_var},
                outputs={"Out": dequant_var})
            op_out_rename_map[block_id][out_var.name] = dequant_var.name
            return dequant_var

        def _load_var(name):
            return np.array(scope.find_var(name).get_tensor())

        def _restore_var(name, arr):
            t = scope.find_var(name).get_tensor()
            t.set(arr, place)

        for block in program.blocks:
            ops = list(block.ops)
            block_id = block.idx
            for op in ops:
                op_type = op.type

                # insert dequant_op after fc/conv, need to rename
                # input of the followed ops
                for name in op.input_arg_names:
                    if name in op_out_rename_map[block_id]:
                        op.rename_input(name, op_out_rename_map[block_id][name])

                if op_type in self.fake_quant_op_types:
                    in_arg_name = op.input('X')[0]
                    if in_arg_name in persistable_vars:
                        if self.weight_quantize_type == 'abs_max':
                            param = _load_var(in_arg_name)
                            scale_v = np.max(np.abs(param))
                        else:
                            scale_v = _load_var(op.output('OutMovingScale')[0])
                        var_scale_map[block_id][in_arg_name] = scale_v
                    else:
                        scale_v = block.var(op.output('OutMovingScale')[0])
                        var_scale_map[block_id][in_arg_name] = scale_v

                    if in_arg_name in persistable_vars and freeze_weight:
                        _remove_fake_quant_and_dequant_op(block, op)
                        # quantize weight and restore
                        param_t = _load_var(in_arg_name)
                        param_q_t = quant(param_t, scale_v, self.weight_bits)
                        _restore_var(in_arg_name, param_q_t)

                if op_type in self.fake_dequant_op_types:
                    _remove_fake_quant_and_dequant_op(block, op)

                if op_type in _QUANTIZABLE_OP_TYPES:
                    dequant_var = _insert_post_dequant_op(block, op)

        # remove the unused var in ProgramDesc
        self._remove_unused_var(program)
        #program = program.clone()

    def convert_to_int8(self, program, place, scope=None):
        scope = global_scope() if scope is None else scope
        program = default_main_program() if program is None else program

        def _load_var(name):
            return np.array(scope.find_var(name).get_tensor())

        global_block = program.global_block()

        def convert_to_int8(var):
            int8_var_name = var.name + ".int8"
            int8_var = global_block.create_parameter(
                name=int8_var_name.encode('ascii'),
                type=var.type,
                dtype=core.VarDesc.VarType.INT8,
                shape=var.shape)

            tensor = _load_var(var.name)

            scope.var(int8_var_name)
            int8_tensor = scope.find_var(int8_var_name).get_tensor()
            int8_tensor.set(tensor.astype(np.int8), place)
            return int8_var

        input_map = {}
        for block in program.blocks:
            for op in list(block.ops):
                if op.type in _QUANTIZABLE_OP_TYPES:
                    for name in op.input_arg_names:
                        var = block.var(name)
                        if var.persistable:
                            if name not in input_map:
                                int8_var = convert_to_int8(var)
                                input_map[name] = int8_var.name
                            op.rename_input(name, input_map[name])
        self._remove_unused_var(program)

    def _remove_unused_var(self, program):
        for block in program.blocks:
            args = []
            for op in block.ops:
                args += op.input_arg_names
                args += op.output_arg_names
            args = list(set(args))
            for var in block.vars.keys():
                if var not in args:
                    block._remove_var(var)

    def _insert_quant_op(self, block, idx, var, quant_bits, quant_type):
        """
        insert fake_quantize_op
        """
        quant_var = block.create_var(
            name=_quantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)

        iter = self.helper.create_global_variable(
            name=unique_name.generate('iteration'),
            persistable=True,
            dtype='int32',
            shape=[1])
        self.helper.set_variable_initializer(
            iter, initializer=Constant(value=0))

        scales = self.helper.create_global_variable(
            name=unique_name.generate('scales'),
            persistable=True,
            dtype=var.dtype,
            shape=[self.window_size])
        self.helper.set_variable_initializer(
            scales, initializer=Constant(value=0))

        scale = self.helper.create_parameter(
            attr=ParamAttr(
                name=_quantized_scale_name(var.name),
                initializer=Constant(0.0),
                trainable=False),
            shape=[1],
            dtype=var.dtype)
        scale.stop_gradient = True

        ins = {
            'X': var,
            'InScales': scales,
            'InMovingScale': scale,
            'InCurrentIter': iter
        }
        outs = {
            'Out': quant_var,
            'OutScales': scales,
            'OutMovingScale': scale,
            'OutCurrentIter': iter
        }
        attrs = {
            'quantize_type': quant_type,
            'window_size': self.window_size,
            'bit_length': quant_bits,
            'is_test': self.is_inference
        }

        quant_op = block._insert_op(
            idx, type='fake_quantize', attrs=attrs, inputs=ins, outputs=outs)
        return quant_var, scale

    def _insert_dequant_op(self, block, idx, var, scale, quant_bits):
        """
        insert fake_quantize_op
        """
        dequant_var = block.create_var(
            name=_dequantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        # insert fake_dequantize_op
        max_range = (1 << (quant_bits - 1)) - 1
        dequant_op = block._insert_op(
            idx,
            type="fake_dequantize_max_abs",
            attrs={'range': float(max_range)},
            inputs={"X": var,
                    'Scale': scale},
            outputs={"Out": dequant_var})
        return dequant_var
