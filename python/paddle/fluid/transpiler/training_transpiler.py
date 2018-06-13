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

import os
import numpy as np
from .. import core
from ..framework import Program
from ..executor import global_scope
from base_transpiler import BaseTranspiler


class TrainingTranspiler(BaseTranspiler):
    def transpile(self, program, place, scope=None):
        '''
        Transpile the program. Support only mkldnn relu fuse now.

        :param program: program to transpile
        :type program: Program
        '''
        if not isinstance(program, Program):
            raise TypeError("program should be as Program type")
        if not isinstance(place, core.CPUPlace) and not isinstance(
                place, core.CUDAPlace):
            raise TypeError("place should be as CPUPlace/CUDAPlace type")
        if scope is None:
            scope = global_scope()
        if not isinstance(scope, core.Scope):
            raise TypeError("scope should be as Scope type or None")
        self.fuse_relu_mkldnn(program)

    def fuse_relu_mkldnn(self, program):
        '''
        Transpile the program by fused relu activation for MKLDNN program.

        Relu activation following batch norm OP can be fused by adding
        :math:`fuse_with_relu` attribute to batch norm OP.

        The result of fuse is:

        - before:

          - batch_norm->relu->any_other_op

        - after:

          - batch_norm->any_other_op

        :param program: program to transpile
        :type program: Program
        '''
        use_mkldnn = bool(os.getenv("FLAGS_use_mkldnn", False))
        if not use_mkldnn:
            return

        self.block = program.block(0)

        i = 0
        while i < len(self.block.ops) - 1:
            current_op = self.block.ops[i]
            if current_op.type in ['batch_norm']:
                next_op = self.block.ops[i + 1]
                if next_op.type == 'relu':
                    # modify bnorm OP to include relu
                    current_op.set_attr("fuse_with_relu", True)
                    # remove relu OP
                    self.block._remove_op(i + 1)
            elif current_op.type == 'relu_grad':
                next_op = self.block.ops[i + 1]
                if next_op.type == 'batch_norm_grad':
                    # remove relu OP
                    self.block._remove_op(i)
            i = i + 1

        self._remove_unused_var()
        # TODO(luotao): use clone() method to flush the program.desc in force,
        # since some large program.desc will not be flushed immediately.
        # And a better solution will be considered later.
        program = program.clone()
