# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import paddle.fluid.core as core
import paddle.fluid.framework as framework

from paddle.fluid.transpiler.details.program_utils import delete_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_heter_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import create_heter_program
from paddle.fluid.incubate.fleet.parameter_server.ir.public import create_trainer_program
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_block_joint
from paddle.fluid.incubate.fleet.parameter_server.ir.public import add_vars_by_op_map
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_block_input_output
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_op_input_output
from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_vars_name_in_block
from paddle.fluid.incubate.fleet.parameter_server.ir.public import block_append_op
from paddle.fluid.incubate.fleet.parameter_server.ir.public import replace_ops_by_communicate_op
from paddle.fluid.incubate.fleet.parameter_server.ir.public import _get_input_map_from_op, _get_output_map_from_op
from paddle.fluid.incubate.fleet.parameter_server.ir.pserver_pass import add_listen_and_serv_pass


def split_heter_worker_ops_pass(program, config):
    default_deveice = "cpu"
    program, heter_ops, _, program_block_ops = find_heter_ops(
        program, default_deveice)
    if len(heter_ops) == 0:
        return program
    current_device = "gpu"
    if current_device not in heter_ops:
        raise ValueError(
            "Op which run on device {} not exist.".format(current_device))

    block_vars_detail = find_block_joint(program, program_block_ops)
    heter_program = framework.Program()
    create_heter_program(program, heter_program, heter_ops,
                         block_vars_detail, current_device)
    return program


def split_trainer_ops_pass(program, config):
    default_deveice = "cpu"
    program, heter_ops, _, program_block_ops = find_heter_ops(
        program, default_deveice)
    block_vars_detail = find_block_joint(program, program_block_ops)
    create_trainer_program(program, config, heter_ops, block_vars_detail)
    return program


def delete_startup_useless_ops_var_pass(startup_program, main_program, config):
    # find all op and its var
    vars_in_main_program = get_vars_name_in_block(main_program.global_block())

    block_nums = startup_program.num_blocks
    for block_index in range(1, block_nums):
        current_block = startup_program.block(block_index)
        # delete useless op
        need_delete_op = []
        for op in current_block.ops:
            inputs, outputs = find_op_input_output(
                startup_program, current_block, op)
            inputs += outputs
            # Todo: delete some concat op
            if list(set(inputs) & set(vars_in_main_program)) == None:
                need_delete_op.append(op)
        delete_ops(current_block, need_delete_op)

        # delete useless var
        for var in current_block.vars:
            if var.name not in vars_in_main_program:
                startup_program._remove_var(var.name)

    return startup_program
