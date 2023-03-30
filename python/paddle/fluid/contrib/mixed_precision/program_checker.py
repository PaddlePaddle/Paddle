# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from ... import core
from ... import default_main_program
from ... import default_startup_program
from ... import framework
from ... import layers
from ... import program_guard
from ... import unique_name
from . import fp16_utils
from .fp16_utils import rewrite_program
from .fp16_utils import cast_model_to_fp16
from .fp16_utils import cast_parameters_to_fp16
from .fp16_utils import update_role_var_grad
from .fp16_lists import AutoMixedPrecisionLists
from .amp_nn import check_finite_and_unscale
from .amp_nn import update_loss_scaling
import types
import warnings
import paddle

__all__ = ["collect_operator_stats"]

class op_checker(object):
    def __init__(self) -> None:
        self.op_name = None
        self.fp32_calls = 0
        self.fp16_calls = 0
        self.bf16_calls = 0

    def inc_calls(self, out_var):
        if out_var.dtype == core.VarDesc.VarType.FP32:
            self.fp32_calls = self.fp32_calls + 1
        if out_var.dtype == core.VarDesc.VarType.FP16:
            self.fp16_calls = self.fp16_calls + 1
        if out_var.dtype == core.VarDesc.VarType.BF16:
            self.bf16_calls = self.bf16_calls + 1


def collect_operator_stats(program=default_main_program()):
    block = program.global_block()
    
    op_checker_list = []
    param_names = [p.name for p in block.all_parameters()]

    global_block = program.global_block()
    
    for block in program.blocks:
        ops = block.ops
        for op in ops:
            if op.type == 'create_py_reader' or op.type == 'read' or op.type == 'create_double_buffer_reader':
                continue

            op_name = op.type
            if 'Out' in op.output_names:
                out_names = op.output('Out')
            elif 'Y' in op.output_names:
                out_names = op.output('Y')
            elif 'X@GRAD' in op.output_names:
                out_names = op.output('X@GRAD')
            else:
                continue

            out_name = out_names[0]

            if op.type == 'elementwise_mul':
                print(f"outvar={global_block.var(out_name)}, op={op}")

            is_in_list = False
            for each_checker in op_checker_list:
                if op_name == each_checker.op_name:
                    each_checker.inc_calls(global_block.var(out_name))
                    is_in_list = True
                    break
            
            if not is_in_list:
                static_op_checker = op_checker()
                static_op_checker.op_name = op_name
                static_op_checker.inc_calls(global_block.var(out_name))
                op_checker_list.append(static_op_checker)

    for each_checker in op_checker_list:
        print(f"op={each_checker.op_name}, fp32 calls={each_checker.fp32_calls}, fp16 calls={each_checker.fp16_calls}, bf16 calls={each_checker.bf16_calls}")


            
    #print(param_names)