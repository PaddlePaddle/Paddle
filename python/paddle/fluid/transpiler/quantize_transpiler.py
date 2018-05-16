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

from ..framework import Program

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


class QuantizeTranspiler(object):
    """
    TODO(qingqing): add comments
    """

    def transpile(self,
                  program,
                  is_inference=False,
                  weight_bits=8,
                  activation_bits=8,
                  quant_delay=0):

        if not isinstance(program, Program):
            raise TypeError("program should be as Program type")

        self.is_inference = is_inference
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quant_delay = quant_delay

        if not self.is_inference:
            self._training_transpile(program)
        else:
            self._inference_transpile(program)

    def _training_transpile(self, program):
        """
        TODO(qingqing): add comments
        """
        # marked the variable which has been quantized and dequantized.
        dequanted_vars = [
            collections.OrderedDict() for _ in range(len(program.blocks))
        ]
        grad_op_types = ['%s_grad' % (type) for type in _QUANTIZABLE_OP_TYPES]

        def _transpile_forward(block, op):
            idx = block.ops.index(op)
            block_id = block.idx
            # insert quant op and dequant op
            for name in op.input_arg_names:
                if name in dequanted_vars[block_id]:
                    dequant_var = dequanted_vars[block_id][name]
                else:
                    var = block.var(name)
                    quant_var = self._insert_quant_op(block, idx, var)
                    dequant_var = self._insert_dequant_op(block, idx + 1,
                                                          quant_var)
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

    def _insert_quant_op(self, block, idx, var):
        """
        TODO(qingqing): add comments
        """
        quant_var = block.create_var(
            name=_quantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        # insert fake_quantize_op
        # since the fake_quantize_op is not implemented now,
        # use tanh op for testing
        quant_op = block.insert_op(
            idx, type="cos", inputs={"X": var}, outputs={"Out": quant_var})
        return quant_var

    def _insert_dequant_op(self, block, idx, var):
        """
        TODO(qingqing): add comments
        """
        dequant_var = block.create_var(
            name=_dequantized_var_name(var.name),
            type=var.type,
            shape=var.shape,
            dtype=var.dtype)
        # insert fake_dequantize_op
        # since the fake_dequantize_op is not implemented now,
        # use relu op for testing
        dequant_op = block.insert_op(
            idx, type="sin", inputs={"X": var}, outputs={"Out": dequant_var})
        return dequant_var
