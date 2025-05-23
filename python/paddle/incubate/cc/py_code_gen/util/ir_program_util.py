# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.incubate.cc.py_code_gen.ir import ir_op


def IsBackwardProgram(ir_program):
    for name, op in vars(ir_program).items():
        if not isinstance(op, ir_op.Op):
            continue
        if op.name != "builtin.module":
            continue
        keyword_arg_names = op.block_keyword_arg_names[0][0]
        if len(keyword_arg_names) > 0:
            return True
    return False


def GetProgramId(ir_program):
    return int(type(ir_program).__name__[len("PirProgram_") :])
