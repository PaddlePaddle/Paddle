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
import copy

from paddle.fluid.framework import default_main_program, default_startup_program
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import unique_name
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr

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


class QuantizeTranspiler(object):
    """
    TODO(qingqing): add comments
    """

    def transpile(self,
                  program,
                  is_inference=False,
                  weight_bits=8,
                  activation_bits=8,
                  weight_quantize_type='abs_max',
                  activation_quantize_type='abs_max',
                  window_size=1000,
                  quant_delay=0):

        self.is_inference = is_inference
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quant_delay = quant_delay

        self.weight_quantize_type = weight_quantize_type
        self.activation_quantize_type = activation_quantize_type

        self.window_size = window_size

        self.helper = LayerHelper(self.__class__.__name__)

        if not self.is_inference:
            self._training_transpile(program)
        else:
            # TODO(qingqing)
            pass

    def _training_transpile(self, program):
        """
        TODO(qingqing): add comments
        """
        program = default_main_program()
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
                # rewrite the forward ProgramDes
                # rename the backward op inputs
                if op.type in grad_op_types:
                    _transpile_backward(block, op)

    def _insert_quant_op(self, block, idx, var, quant_bits, quant_type):
        """
        TODO(qingqing): add comments
        """
        quant_var = block.create_var(
            name=_quantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        # insert fake_quantize_op

        iter = self.helper.create_global_variable(
            name=unique_name.generate('iteration'),
            persistable=True,
            dtype='int32',
            shape=[1])
        self.helper.set_variable_initializer(
            iter, initializer=Constant(value=0))

        accum = self.helper.create_global_variable(
            name=unique_name.generate('accum'),
            persistable=True,
            dtype=var.dtype,
            shape=[1])
        self.helper.set_variable_initializer(
            accum, initializer=Constant(value=1))

        state = self.helper.create_global_variable(
            name=unique_name.generate('state'),
            persistable=True,
            dtype=var.dtype,
            shape=[1])
        self.helper.set_variable_initializer(
            state, initializer=Constant(value=1))


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
                initializer=Constant(0.001),
                trainable=False),
            shape=[1],
            dtype=var.dtype)
        scale.stop_gradient = True

        ins = {
            'X': var,
            'InScales': scales,
            'InMovingScale': scale,
            'InCurrentIter': iter,
            'InAccum': accum,
            'InState': state
        }
        outs = {
            'Out': quant_var,
            'OutScales': scales,
            'OutMovingScale': scale,
            'OutCurrentIter': iter,
            'OutAccum': accum,
            'OutState': state
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
        TODO(qingqing): add comments
        """
        dequant_var = block.create_var(
            name=_dequantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        # insert fake_dequantize_op
        dequant_op = block._insert_op(
            idx,
            type="fake_dequantize_max_abs",
            attrs={'num_bits': quant_bits},
            inputs={"X": var,
                    'Scale': scale},
            outputs={"Out": dequant_var})
        return dequant_var
