#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import numpy as np
from .... import core
from ....framework import Program, Operator, Variable, program_guard
from .... import unique_name
from ....layer_helper import LayerHelper
from ....param_attr import ParamAttr
from ....initializer import Constant
from ....log_helper import get_logger

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


class QuantizeTranspilerV2(object):
    def __init__(self,
                 weight_bits=8,
                 activation_bits=8,
                 weight_quantize_type='abs_max',
                 activation_quantize_type='abs_max',
                 quantizable_op_type=['conv2d', 'depthwise_conv2d', 'mul'],
                 skip_pattern=['skip_quant']):
        """
        Add quant_dequant op before the quantized op to quantize the fluid Program.
        It is a patch for distributed quantization, we will support others module for
        distributed quantization.

        Args:
            weight_bits(int): the bit of quantized weight.
            activation_bits(int): the bit of quantized activation.
            weight_quantize_type(str): the quantization type for weight.
                Only support to be 'abs_max' for now.
            activation_quantize_type(str): the quantization type for activation.
                Only support to be 'abs_max' for now.
            quantizable_op_type(str): set the op type for quantization.
            skip_pattern(str|list): The user-defined quantization skip pattern, which
                will be presented in the name scope of an op. When the skip pattern is
                detected in an op's name scope, the corresponding op will not be quantized.
        """
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits

        assert activation_quantize_type == "abs_max", \
            "activation_quantize_type should be abs_max for now."
        assert weight_quantize_type == "abs_max", \
            "weight_quantize_type should be abs_max for now."
        self._activation_quantize_type = activation_quantize_type
        self._weight_quantize_type = weight_quantize_type

        self._quantizable_ops = quantizable_op_type
        self._quantizable_grad_ops = [
            '%s_grad' % (op) for op in self._quantizable_ops
        ]

        self._skip_pattern = skip_pattern
        self.helper = LayerHelper(self.__class__.__name__)

    def apply(self, program, startup_program):
        """
        Apply quantization to fluid Program.

        Args:
            program(Program): the train or test program to be quantized.
            startup_program(Program): the corresponding startup_program.
        Returns:
            None
        """
        assert isinstance(program, Program), \
            "program must be the instance of Program"
        assert isinstance(startup_program, Program), \
            "startup_program must be the instance of Program"

        quant_dequant_vars = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]
        with program_guard(program, startup_program):
            for block in program.blocks:
                ops = list(block.ops)
                for op in ops:
                    if op.type in self._quantizable_ops and \
                        (not self._is_skip_quant(op)):
                        self._transform_forward(block, op, quant_dequant_vars)
            for block in program.blocks:
                ops = list(block.ops)
                for op in ops:
                    if op.type in self._quantizable_grad_ops and \
                        (not self._is_skip_quant(op)):
                        self._transform_backward(block, op, quant_dequant_vars)

    def _is_skip_quant(self, op):
        """
        Analyse whether the op should skip quantization or not.
        """
        user_skipped = False
        if isinstance(self._skip_pattern, list):
            user_skipped = op.has_attr("op_namescope") and \
                            any(pattern in op.attr("op_namescope") \
                                for pattern in self._skip_pattern)
        elif isinstance(self._skip_pattern, str):
            user_skipped = op.has_attr("op_namescope") and \
                            op.attr("op_namescope").find(
                                self._skip_pattern) != -1
        return user_skipped

    def _transform_forward(self, block, op, quant_dequant_vars):
        op._set_attr("quantization_type", "qat_with_weight")
        idx = block.ops.index(op)
        block_id = block.idx
        for in_name in op.input_arg_names:
            if in_name in quant_dequant_vars[block_id]:
                quant_dequant_var = quant_dequant_vars[block_id][in_name]
            else:
                in_var = block.var(in_name)
                quant_bits = self._weight_bits if in_var.persistable \
                        else self._activation_bits
                quant_type = self._weight_quantize_type if in_var.persistable \
                        else self._activation_quantize_type
                if quant_type == "abs_max":
                    quant_dequant_var = self._insert_quant_dequant_abs_max_op(
                        block, idx, in_var, quant_bits)
                else:
                    _logger.error("Quant_type only supported to be abs_max")
                quant_dequant_vars[block_id][in_name] = quant_dequant_var
                op._rename_input(in_name, quant_dequant_var.name)

    def _transform_backward(self, block, op, quant_dequant_vars):
        block_id = block.idx
        no_dequanted_input_vars = True
        for name in op.input_arg_names:
            if name in quant_dequant_vars[block_id]:
                dequant_var = quant_dequant_vars[block_id][name]
                op._rename_input(name, dequant_var.name)
                no_dequanted_input_vars = False
        if no_dequanted_input_vars:
            raise ValueError("There is no dequanted inputs for op %s." %
                             (op.type))

    def _insert_quant_dequant_abs_max_op(self, block, idx, in_var, quant_bits):
        quant_dequant_var = block.create_var(
            type=in_var.type,
            name="{}.quant_dequant".format(in_var.name),
            shape=in_var.shape,
            dtype=in_var.dtype)
        scale_var = self.helper.create_parameter(
            attr=ParamAttr(
                name="{}.quant_dequant.scale".format(in_var.name),
                initializer=Constant(0.001),
                trainable=False),
            shape=[1],
            dtype=in_var.dtype)
        scale_var.stop_gradient = True

        inputs = {'X': in_var}
        outputs = {'Out': quant_dequant_var, 'OutScale': scale_var}
        attrs = {'bit_length': quant_bits}
        block._insert_op(
            idx,
            type='fake_quantize_dequantize_abs_max',
            attrs=attrs,
            inputs=inputs,
            outputs=outputs)
        return quant_dequant_var
