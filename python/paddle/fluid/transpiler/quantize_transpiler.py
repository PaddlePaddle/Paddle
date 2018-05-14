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

from ..framework import Program

_QUANTIZABLE_OP_TYPES = ['conv2d', 'depthwise_conv2d', 'mul']
_INPUT_KEYS = [['Input', 'Filter'], ['Input', 'Filter'], ['Input', 'Filter']]


class QuantizeTranspiler(object):
    def transpile(self,
                  program,
                  is_train,
                  weight_bits=8,
                  activation_bits=8,
                  quant_delay=0):

        if not isinstance(program, Program):
            raise TypeError("program should be as Program type")

        self.program = program
        self.is_train = is_train
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.quant_delay = quant_delay
        # {string:bool} map is used to marked which variable has been quantizied.
        self.quanted_vars = collections.OrderedDict()

        if self.is_train:
            transpile_for_training()
        else:
            transpile_for_testing()

    def transpile_for_training(self):
        for block in program.blocks:
            for i, op in enumerate(block.ops):
                if op.type in _QUANTIZABLE_OP_TYPES:
                    inputs = _INPUT_KEYS[_QUANTIZABLE_OP_TYPES.index(op.type)]
                    self._insert_quant(block, i, op, inputs)
                    self._insert_dequant(block, i, op, inputs)

    def _insert_quant_op(self, block, idx, op, inputs):
        for key in inputs:
            var = block.var(op.input(key)[0])
            if self.quanted_vars[var.name]:
                continue
            # quant_var = block.create_var(name="%s.quanted" % (var.name),
            #                             type=var.type,
            #                             hape=var.shape,
            #                             dtype=var.dtype)
            # insert fake_quantize_op, which is a in-place op
            # since the fake_quantize_op is not implemented now,
            # use tanh op for testing
            quant_op = block.insert_op(
                idx, type="cos", inputs={"X": var}, outputs={"Out": var})
            self.quanted_vars[var.name] = True

    def _insert_dequant_op(self, block, idx, op, inputs):
        for key in inputs:
            var = block.var(op.input(key)[0])
            if self.quanted_vars[var.name]:
                continue
            # insert fake_dequantize_op, which is a in-place op
            # since the fake_dequantize_op is not implemented now,
            # use relu op for testing
            dequant_op = block.insert_op(
                idx + 1, type="sin", inputs={"X": var}, outputs={"Out": var})
            self.quanted_vars[var.name] = True
