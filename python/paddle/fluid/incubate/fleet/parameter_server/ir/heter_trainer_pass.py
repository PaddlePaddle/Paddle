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

from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_heter_ops
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_block_joint
from paddle.fluid.incubate.fleet.parameter_server.ir.public import add_vars_by_op_map
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_block_input_output
from paddle.fluid.incubate.fleet.parameter_server.ir.public import find_op_input_output
from paddle.fluid.incubate.fleet.parameter_server.ir.public import get_vars_name_in_block
from paddle.fluid.incubate.fleet.parameter_server.ir.public import block_append_op
from paddle.fluid.incubate.fleet.parameter_server.ir.public import replace_ops_by_communicate_op
from paddle.fluid.incubate.fleet.parameter_server.ir.program_utils import _get_input_map_from_op, _get_output_map_from_op
from paddle.fluid.incubate.fleet.parameter_server.ir.pserver_pass import add_listen_and_serv_pass


def split_heter_worker_ops_pass(program, config):
    default_deveice = "cpu"
    current_device = "gpu"
    program, heter_ops, _, program_block_ops = find_heter_ops(
        program, default_deveice)

    if len(heter_ops) == 0:
        return program

    if current_device not in heter_ops:
        raise ValueError(
            "Op which run on device {} not exist.".format(current_device))

    block_vars_detail = find_block_joint(program, program_block_ops)

    heter_program = Program()

    # add heter op
    pre_block_idx = heter_program.num_blocks - 1
    for index, heter_block_ops in heter_ops[current_device].items():
        heter_block = heter_program._create_block(pre_block_idx)
        for _, op in enumerate(heter_block_ops):
            block_append_op(heter_program, program, heter_block, op)

            # add relate variables
            inputs = _get_input_map_from_op(
                program.global_block().vars, op)
            add_vars_by_op_map(inputs, heter_program)

            outputs = _get_output_map_from_op(
                program.global_block().vars, op)
            add_vars_by_op_map(outputs, heter_program)

        # entrance_vars = block_vars_detail[index]["entrance"]
        # exit_vars = block_vars_detail[index]["exit"]

    # attrs = {
    #     "optimize_blocks": optimize_block,
    #     "endpoint": endpoint,
    #     "Fanin": self.trainer_num,
    #     "distributed_mode": DistributedMode.GEO,
    #     "grad_to_block_id": param_to_block_id,
    #     "sparse_grad_to_param": sparse_grad_to_param,
    #     "rpc_get_thread_num": self.server_config._rpc_get_thread_num,
    #     "rpc_send_thread_num": self.server_config._rpc_send_thread_num,
    #     "rpc_prefetch_thread_num":
    #         self.server_config._rpc_prefetch_thread_num
    # }

    # append the listen_and_serv op
    program.global_block().append_op(
        type="listen_and_serv",
        inputs={'X': []},
        outputs={},
        attrs={})

    return program


def split_trainer_ops_pass(program, config):
    default_deveice = "cpu"
    # 复用XPU-Trainer逻辑找到连接点
    origin_program = program.clone()
    origin_program, _, _, program_block_ops = find_heter_ops(
        origin_program, default_deveice)
    block_vars_detail = find_block_joint(origin_program, program_block_ops)

    block_nums = heter_program.num_blocks
    for block_index in range(1, block_nums):
        current_block = heter_program.block(block_index)
        block_input, block_output = find_block_input_output(
            heter_program, current_block)
        # find entrance & exit
        block_private_vars = list(set(block_input) & set(block_output))
        block_entrance = list(set(block_input)-set(block_private_vars))
        block_exit = list(set(block_output)-set(block_private_vars))

        # delete useless op & add communicate op
        replace_ops_by_communicate_op(origin_program.global_block(),
                                      current_block.ops, origin_program, block_entrance, block_exit, config)
        # delete useless var
        for var in block_private_vars:
            if origin_program.global_block().has_var(var):
                origin_program.global_block()._remove_var(var)

    return origin_program


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
