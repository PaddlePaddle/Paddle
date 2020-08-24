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
from functools import reduce

import collections
import math
import os
import warnings

import six
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.core import CommContext
import paddle.fluid.framework as framework
from paddle.fluid.incubate.fleet.parameter_server.mode import DistributedMode
from paddle.fluid.incubate.fleet.parameter_server.ir import vars_metatools
from paddle.fluid.incubate.fleet.parameter_server.ir.ps_dispatcher import RoundRobin, PSDispatcher
from paddle.fluid.transpiler.details.program_utils import delete_ops

OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "@CLIP"
STEP_COUNTER = "@PS_STEP_COUNTER@"
LEARNING_RATE_DECAY_COUNTER = "@LR_DECAY_COUNTER@"

OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize
DEVICE_LIST = ["cpu", "gpu", "xpu"]
COMMUNICATE_OPS_TYPE = ["send", "recv", "fetch_barrier", "send_barrier"]
DEFAULT_DEVICE = 'cpu'


def _get_lr_ops(program):
    lr_ops = []
    for index, op in enumerate(program.global_block().ops):
        role_id = int(op.attr(RPC_OP_ROLE_ATTR_NAME))
        if role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) or \
                role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) | \
                int(OPT_OP_ROLE_ATTR_VALUE):
            lr_ops.append(op)
    return lr_ops


def _has_global_step(lr_ops):
    if len(lr_ops) > 0:
        for idx, op in enumerate(lr_ops):
            if op.type != 'increment':
                continue
            counter = op.input("X")[0]
            if counter == LEARNING_RATE_DECAY_COUNTER:
                return True
    return False


def is_sparse_op(op):
    if op.type == "lookup_table" and op.attr('is_sparse') is True and op.attr(
            'is_distributed') is False:
        return True

    if op.type == "distributed_lookup_table" and op.attr(
            'is_distributed') is False:
        return True

    return False


def is_distributed_sparse_op(op):
    if op.type == "lookup_table" and op.attr('is_distributed') is True:
        return True

    if op.type == "distributed_lookup_table" and op.attr(
            'is_distributed') is True:
        return True

    return False


def get_sparse_tablename(op):
    return op.input("W")[0]


def get_sparse_tablenames(program, is_distributed):
    tablenames = set()
    if is_distributed:
        for op in program.global_block().ops:
            if is_distributed_sparse_op(op):
                tablenames.add(get_sparse_tablename(op))
    else:
        for op in program.global_block().ops:
            if is_sparse_op(op):
                tablenames.add(get_sparse_tablename(op))
    return list(tablenames)


def find_heter_ops(program, default_device="cpu"):
    if default_device not in DEVICE_LIST:
        raise ValueError("Given device {} is not in device list {}".format(
            default_device, DEVICE_LIST))

    def _is_heter_op(op, current_heter_device, default_device="cpu"):
        heter_devices = list(DEVICE_LIST)
        heter_devices.remove(default_device)
        op_device = op.attr("op_device")
        op_type = op.type
        if op_device in heter_devices:
            return True
        elif op_type in COMMUNICATE_OPS_TYPE and current_heter_device != default_device:
            # for distributed communciate ops: send & recv & barrier
            # Todo: need update this method
            op._set_attr('op_device', current_heter_device)
            return True
        elif op_device == None or op_device == default_device:
            op._set_attr('op_device', default_device)
            return False
        return False

    def _is_same_device(op, pre_device, default_device="cpu"):
        op_device = op.attr("op_device")
        if op_device == pre_device:
            return True
        if pre_device == default_device:
            return True
        return False

    def _append_heter_op(op, current_heter_block_ops, heter_ops):
        op_device = op.attr("op_device")
        if op_device not in heter_ops:
            heter_ops[op_device] = {}
        current_heter_block_ops.append(op)

    origin_porgram = program.clone()
    block = program.global_block()

    program_block_ops = []
    default_ops = {default_device: {}}
    heter_ops = {}
    block_index = 0
    # heter_ops: {"gpu": {1:[op1, op2, ...], 2:[op1, op2, ...] }; "xpu": {3:[op1, op2, ...], 4:[op1, op2, ...] }}

    current_heter_block_ops = []
    current_default_block_ops = []
    current_heter_device = default_device
    is_heter = False
    for op in block.ops:
        if _is_heter_op(op, current_heter_device, default_device):
            # for gpu/xpu-op
            is_heter = True

            # for cpu-op block append
            if len(current_default_block_ops) > 1:
                default_ops[default_device][
                    block_index] = current_default_block_ops
                program_block_ops.append(current_default_block_ops)
                current_default_block_ops = []
                block_index += 1

            if _is_same_device(op, current_heter_device, default_device):
                # for gpu-op, gpu-op -> gpu-op,...
                current_heter_device = op.attr("op_device")
                _append_heter_op(op, current_heter_block_ops, heter_ops)
            else:
                # for gpu-op -> xpu-op, ...
                op_device = current_heter_block_ops[0].attr("op_device")
                heter_ops[op_device][block_index] = current_heter_block_ops
                program_block_ops.append(current_heter_block_ops)
                block_index += 1
                current_heter_block_ops = []
                current_heter_device = op.attr("op_device")
                _append_heter_op(op, current_heter_block_ops, heter_ops)

        elif is_heter:
            # for gpu/xpu-op -> cpu-op
            op_device = current_heter_block_ops[0].attr("op_device")
            heter_ops[op_device][block_index] = current_heter_block_ops
            program_block_ops.append(current_heter_block_ops)
            block_index += 1
            current_heter_block_ops = []
            current_heter_device = default_device
            is_heter = False
            current_default_block_ops.append(op)
        else:
            # for cpu-op
            current_default_block_ops.append(op)

    if current_default_block_ops != []:
        default_ops[default_device][block_index] = current_default_block_ops
        program_block_ops.append(current_default_block_ops)

    if current_heter_block_ops != []:
        op_device = current_heter_block_ops[0].attr("op_device")
        heter_ops[op_device][block_index] = current_heter_block_ops
        program_block_ops.append(current_heter_block_ops)

    if len(heter_ops) == 0:
        warnings.warn(
            "No heterogeneous OP was found in your program , "
            " please using fluid.device_guard() to run OPs on different device.")

    total_heter_ops = 0
    heter_blocks = 0
    for device in heter_ops.keys():
        heter_block_dict = heter_ops[device]
        heter_blocks += len(heter_block_dict)
        for _, heter_block in heter_block_dict.items():
            total_heter_ops += len(heter_block)
    print(
        "There are {} OPs in your main_program, and contains {} heter-OPs which is made up of {} heter-blocks.".
        format(len(block.ops), total_heter_ops, heter_blocks))
    return origin_porgram, heter_ops, default_ops, program_block_ops


def create_heter_program(program, config, heter_program, heter_ops,
                         block_var_detail, current_device):
    # add heter op
    optimizer_block = []
    grad_to_block_id = []
    send_grad_var_list = []

    pre_block_idx = heter_program.num_blocks - 1
    for index, heter_block_ops in heter_ops[current_device].items():
        heter_block = heter_program._create_block(pre_block_idx)
        optimizer_block.append(heter_block)
        for _, op in enumerate(heter_block_ops):
            block_append_op(heter_program, program, heter_block, op)

            # add relate variables
            inputs = _get_input_map_from_op(program.global_block().vars, op)
            add_vars_by_op_map(inputs, heter_program)

            outputs = _get_output_map_from_op(program.global_block().vars, op)
            add_vars_by_op_map(outputs, heter_program)

        entrance_vars = block_var_detail[index]["entrance"]
        add_vars_by_var_list(entrance_vars, program, heter_program)
        exit_vars = block_var_detail[index]["exit"]
        add_vars_by_var_list(exit_vars, program, heter_program)

        comm_info = get_communicate_var_info(program, index, entrance_vars,
                                             exit_vars)

        grad_to_block_id.append(comm_info["block_input_var_name"] + ":" + str(
            heter_block.idx))

        # create slice op
        first_op_index = 0

        get_type_var_name = comm_info["input_var_reshape_name"][0].split(
            "@HETER_BLOCK")[0]
        get_type_var = heter_program.global_block().vars[get_type_var_name]

        insert_recv_slice_op(
            heter_program, heter_block, first_op_index,
            comm_info["block_input_var_name"],
            (-1, sum(comm_info["input_var_reshape_dim"])), get_type_var.dtype,
            get_type_var.type, comm_info["input_var_reshape_name"], [
                (-1, comm_info["input_var_reshape_dim"][i])
                for i in range(len(comm_info["input_var_reshape_dim"]))
            ])
        first_op_index += len(comm_info["input_var_reshape_dim"])
        # create reshape op
        for i in range(len(comm_info["input_var_reshape_name"])):
            var_name = entrance_vars[i]
            insert_reshape_op(
                heter_program,
                heter_block,
                first_op_index,
                comm_info["input_var_reshape_name"][i],
                var_name, )
            first_op_index += 1

        first_op_index = len(heter_block.ops)

        # create send reshape op
        for i in range(len(exit_vars)):
            insert_reshape_op(heter_program, heter_block, first_op_index,
                              exit_vars[i],
                              comm_info["output_var_reshape_name"][i],
                              [-1, comm_info["output_var_reshape_dim"][i]])
            first_op_index += 1

        # create send concat op
        insert_send_concat_op(heter_program, heter_block, first_op_index,
                              comm_info["output_var_reshape_name"],
                              comm_info["block_output_var_name"],
                              [-1, sum(comm_info["output_var_reshape_dim"])])
        check_op_device(heter_block, current_device)
        send_grad_var_list = send_grad_var_list + add_heter_send_op(
            program, heter_program, heter_block, block_var_detail[index])

    # add step conter
    send_input_vars = []
    dummy_output = []
    trainer_id = config.get_role_id()
    pserver_endpoints = config.get_ps_endpoints()
    optimizer_block[-1].append_op(
        type="send",
        inputs={"X": send_input_vars},
        outputs={"Out": dummy_output},
        attrs={
            "send_varnames": [STEP_COUNTER],
            "merge_add": True,
            "use_send_handler": False,
            "endpoints": pserver_endpoints
        })

    # add info in listen&serv
    attrs = {
        "grad_to_block_id": grad_to_block_id,
        "sparse_grad_to_param": None,
        "lr_decay_block_id": None,
        "dense_optimize_blocks": None,
        "sparse_optimize_blocks": None,
        "optimize_blocks": optimizer_block,

        # runtime attribute
        "endpoint": config.get_heter_worker_endpoint(),
        "pserver_id": config.get_role_id(),
        "Fanin": config.get_trainers(),
        "distributed_mode": config.get_distributed_mode(),
        "rpc_get_thread_num": 12,
        "rpc_send_thread_num": 12,
        "rpc_prefetch_thread_num": 12
    }

    # append the listen_and_serv op
    heter_program.global_block().append_op(
        type="listen_and_serv", inputs={'X': []}, outputs={}, attrs=attrs)

    check_heter_compile_time_strategy(program, config, send_grad_var_list)


def check_heter_compile_time_strategy(program, config, send_grad_var_list):
    origin_grad_var_list = []
    for _, var_grad in config.merged_variables_pairs:
        origin_grad_var_list.append(var_grad.merged_var.name)

    origin_grad_var_list = list(set(origin_grad_var_list))
    send_grad_var_list = list(set(send_grad_var_list))
    useless_grad_var_list = list(
        set(origin_grad_var_list) - set(send_grad_var_list))

    for useless_grad_var in useless_grad_var_list:
        config.remove_var_pair_by_grad(useless_grad_var)


def create_trainer_program(program, config, heter_ops, block_var_detail):
    for device in heter_ops.keys():
        for heter_block_index in sorted(heter_ops[device]):
            replace_ops_by_communicate_op(program, config, heter_block_index,
                                          heter_ops[device][heter_block_index],
                                          block_var_detail)
            remove_trainer_send_op(program, config, heter_block_index,
                                   block_var_detail)
    deleter_trainer_useless_var(program)
    check_op_device(program.global_block(), DEFAULT_DEVICE)


def replace_ops_by_communicate_op(program, config, heter_block_index, ops_list,
                                  block_var_detail):
    all_op = program.global_block().ops
    start_op = ops_list[0]
    first_op_idx = -1
    for op in all_op:
        if is_same_op(op, start_op):
            first_op_idx = all_op.index(op)
            break
    assert first_op_idx != -1
    delete_same_ops(program.global_block(), ops_list)

    mode = config.get_distributed_mode()
    heter_worker_endpoint = config.get_heter_worker_endpoint()
    entrance_var = block_var_detail[heter_block_index]["entrance"]
    exit_var = block_var_detail[heter_block_index]["exit"]

    default_device_comm_info = get_communicate_var_info(
        program, heter_block_index - 1,
        block_var_detail[heter_block_index - 1]["entrance"],
        block_var_detail[heter_block_index - 1]["exit"])
    comm_info = get_communicate_var_info(program, heter_block_index,
                                         entrance_var, exit_var)

    print("entrance_var: {}".format(entrance_var))
    print("Block {} create: \ndefault_device_comm_info {}, \ncomm_info {}".format(
        heter_block_index, default_device_comm_info, comm_info))
    # create reshape op
    for i in range(len(entrance_var)):
        insert_reshape_op(
            program,
            program.global_block(), first_op_idx, entrance_var[i],
            default_device_comm_info["output_var_reshape_name"][i],
            [-1, default_device_comm_info["output_var_reshape_dim"][i]])
        first_op_idx += 1

    # create concat op
    insert_send_concat_op(
        program,
        program.global_block(), first_op_idx,
        default_device_comm_info["output_var_reshape_name"],
        default_device_comm_info["block_output_var_name"],
        [-1, sum(default_device_comm_info["output_var_reshape_dim"])])
    first_op_idx += 1

    # create send op
    send_input_vars = [
        program.global_block().vars[default_device_comm_info[
            "block_output_var_name"]]
    ]

    get_type_var_name = comm_info["output_var_reshape_name"][0].split(
        "@HETER_BLOCK")[0]
    get_type_var = program.global_block().vars[get_type_var_name]

    program.global_block().create_var(
        name=comm_info["block_output_var_name"],
        shape=(-1, sum(comm_info["output_var_reshape_dim"])),
        dtype=get_type_var.dtype,
        type=get_type_var.type)

    recv_vars = [
        program.global_block().vars[comm_info["block_output_var_name"]]
    ]

    program.global_block()._insert_op(
        index=first_op_idx,
        type="send_and_recv",
        inputs={"X": send_input_vars},
        outputs={"Out": recv_vars},
        attrs={
            "send_var_name": default_device_comm_info["block_output_var_name"],
            "recv_var_name": comm_info["block_output_var_name"],
            "endpoint": heter_worker_endpoint,
            "trainer_id": config.get_role_id(),
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
        })
    first_op_idx += 1

    # recv
    # create slice op
    insert_recv_slice_op(
        program,
        program.global_block(), first_op_idx,
        comm_info["block_output_var_name"],
        (-1, sum(comm_info["output_var_reshape_dim"])), get_type_var.dtype,
        get_type_var.type, comm_info["output_var_reshape_name"], [
            (-1, comm_info["output_var_reshape_dim"][i])
            for i in range(len(comm_info["output_var_reshape_dim"]))
        ])

    first_op_idx += len(comm_info["output_var_reshape_dim"])

    # create reshape op
    for i in range(len(comm_info["output_var_reshape_name"])):
        var_name = comm_info["output_var_reshape_name"][i].split(
            "@HETER_BLOCK")[0]
        insert_reshape_op(
            program,
            program.global_block(),
            first_op_idx,
            comm_info["output_var_reshape_name"][i],
            var_name, )
        first_op_idx += 1


def remove_trainer_send_op(program, config, heter_block_index,
                           block_var_detaile):
    # if trainer do FF->BP->SEND, it has follow vars: var, var@GRAD
    # if trainer only do SEND, it has one var: var@GRAD
    # Delete Send op ,if trainer doesn't has pair var (var<->var@GRAD)
    persistables = block_var_detaile[heter_block_index]["persistables"]
    need_remove_send_op = []
    need_remove_grad_var = []
    for op in find_send_op(program):
        input_list, _ = find_op_input_output(program,
                                             program.global_block(), op)
        for var_name in input_list:
            origin_var_name = var_name.split("@GRAD")[0]
            if origin_var_name in persistables:
                need_remove_send_op.append(op)
                need_remove_grad_var.append(var_name)
    need_remove_send_op = list(set(need_remove_send_op))
    delete_ops(program.global_block(), need_remove_send_op)
    print("need_remove_grad_var: {}".format(need_remove_grad_var))
    for grad_var_name in need_remove_grad_var:
        config.remove_var_pair_by_grad(grad_var_name)


def add_heter_send_op(program, heter_program, block, block_var_detail):
    def _get_send_op_dict():
        send_op_dict = {}
        send_op_list = find_send_op(program)
        for op in send_op_list:
            input_list, _ = find_op_input_output(program,
                                                 program.global_block(), op)
            for var in input_list:
                send_op_dict[var] = op
        return send_op_dict

    send_grad_var_list = []
    send_op_dict = _get_send_op_dict()
    for persistable_var in block_var_detail["persistables"]:
        # check var_name ==  var@GRAD
        if "@GRAD" not in persistable_var:
            continue
        if "GRAD" != persistable_var.split("@")[-1]:
            continue
        if persistable_var not in send_op_dict:
            continue
        block_append_op(program, heter_program, block,
                        send_op_dict[persistable_var])
        send_grad_var_list.append(persistable_var)
    return send_grad_var_list


def find_send_op(program):
    send_op_list = []
    for op in program.global_block().ops:
        if op.type == "send":
            send_op_list.append(op)
    return send_op_list


def get_communicate_var_info(program, block_index, entrance_var_list,
                             exit_var_list):
    input_var_reshape_dim = []
    input_var_reshape_name = []
    block_input_var_name = "HETER_BLOCK@{}_TO_{}@JOINT_VAR".format(
        block_index - 1, block_index)
    output_var_reshape_dim = []
    output_var_reshape_name = []
    block_output_var_name = "HETER_BLOCK@{}_TO_{}@JOINT_VAR".format(
        block_index, block_index + 1)
    entrance_var_list.sort()
    exit_var_list.sort()
    # input
    # HETER_BLOCK_index@JOINT_VAR -> slice -> var@HETER_BLOCK@INPUT_RESHAPE_VAR -> reshape -> var
    for name in entrance_var_list:
        var = program.global_block().vars[name]
        shape = var.shape
        if len(shape) < 2 or shape[0] != -1:
            raise ValueError(
                "Variable {} not support heter training. its shape is {}".
                format(name, shape))
        recv_var_dim = -1 * reduce(lambda x, y: x * y, shape)
        input_var_reshape_dim.append(recv_var_dim)
        input_var_reshape_name.append("{}@HETER_BLOCK@INPUT_RESHAPE_VAR".format(
            name))

    # output
    # var -> reshape -> var@HETER_BLOCK@INPUT_RESHAPE_VAR -> concat -> HETER_BLOCK_index@JOINT_VAR
    for var_name in exit_var_list:
        var = program.global_block().vars[var_name]
        shape = var.shape
        if len(shape) < 2 or shape[0] != -1:
            raise ValueError(
                "Variable {} not support heter training. its shape is {}".
                format(var_name, shape))
        send_reshape_dim = -1 * reduce(lambda x, y: x * y, shape)
        output_var_reshape_dim.append(send_reshape_dim)
        output_var_reshape_name.append(
            "{}@HETER_BLOCK@OUTPUT_RESHAPE_VAR".format(var_name))

    info = {
        "input_var_reshape_dim": input_var_reshape_dim,
        "input_var_reshape_name": input_var_reshape_name,
        "block_input_var_name": block_input_var_name,
        "output_var_reshape_dim": output_var_reshape_dim,
        "output_var_reshape_name": output_var_reshape_name,
        "block_output_var_name": block_output_var_name
    }

    return info


def find_block_joints(program, program_block_ops_list, heter_ops):
    block_var_detail = find_entrance_exit_private(program,
                                                  program_block_ops_list)
    block_var_detail = entrance_exit_check(program, program_block_ops_list,
                                           block_var_detail, heter_ops)
    block_var_detail = delete_block_useless_exit(
        program, program_block_ops_list, block_var_detail)
    return block_var_detail


def find_entrance_exit_private(program, program_block_ops_list):
    block_var_detail = []
    persistables = []
    for index, block_op_list in enumerate(program_block_ops_list):
        block_input, block_output = find_ops_list_input_output(program,
                                                               block_op_list)
        persistables = screen_persistables(
            program, block_input) + screen_persistables(program, block_output)
        # find entrance & exit
        block_private_vars = list(set(block_input) & set(block_output))
        block_entrance = list(set(block_input) - set(block_private_vars))
        block_exit = list(set(block_output) - set(block_private_vars))
        detail = {
            "entrance": block_entrance,
            "exit": block_exit,
            "private": block_private_vars,
            "persistables": persistables
        }
        #print("block index: {}, block_var_detail: {}".format(index, detail))
        block_var_detail.append(detail)
    return block_var_detail


def entrance_exit_check(program, program_block_ops_list, block_var_detail,
                        heter_ops):
    for index in range(len(block_var_detail) - 1, -1, -1):
        if index - 1 < 0:
            break
        previous_block_exit = block_var_detail[index - 1]["exit"]
        previous_block_exit.sort()
        current_block_entrance = block_var_detail[index]["entrance"]
        current_block_entrance.sort()
        if previous_block_exit == current_block_entrance:
            continue
        exist_vars = list(
            set(previous_block_exit) & set(current_block_entrance))
        need_add_vars = list(set(current_block_entrance) - set(exist_vars))
        need_add_vars = find_need_var_from_previous_block(
            need_add_vars, block_var_detail, index, heter_ops)

        previous_block_private = block_var_detail[index - 1]["private"]
        previous_block_entrance = block_var_detail[index - 1]["entrance"]
        for var in need_add_vars:
            if var not in previous_block_private and var not in previous_block_entrance:
                previous_block_entrance.append(var)
            previous_block_exit.append(var)
    return block_var_detail


def find_need_var_from_previous_block(need_add_vars, block_var_detail,
                                      current_index, heter_ops):
    # create index_device_map
    index_device_map = {}
    for index in range(len(block_var_detail)):
        index_device_map[index] = DEFAULT_DEVICE
    for device in heter_ops:
        for index in heter_ops[device].keys():
            index_device_map[index] = device

    pre_index = current_index - 1
    need_ignore_var = []

    # if need_add_var in current device, no need communicate
    for var in need_add_vars:
        while (pre_index >= 0):
            previous_block_private = block_var_detail[pre_index]["private"]
            previous_block_exit = block_var_detail[pre_index]["exit"]
            previous_block_entrance = block_var_detail[pre_index]["entrance"]
            total_var = previous_block_private + previous_block_exit + previous_block_entrance
            if var in total_var:
                if index_device_map[current_index] == index_device_map[
                        pre_index] and index_device_map[
                            current_index] == DEFAULT_DEVICE:
                    need_ignore_var.append(var)
                    break
            pre_index -= 1

    need_add_vars = list(set(need_add_vars).difference(set(need_ignore_var)))
    return need_add_vars


def delete_block_useless_exit(program, program_block_ops_list,
                              block_var_detail):
    for index in range(len(block_var_detail)):
        if index == len(block_var_detail) - 1:
            break
        current_block_exit = block_var_detail[index]["exit"]
        next_block_entrance = block_var_detail[index + 1]["entrance"]
        need_delete_var = []
        for var in current_block_exit:
            if var not in next_block_entrance:
                need_delete_var.append(var)

        for var in need_delete_var:
            current_block_exit.remove(var)

    return block_var_detail


def check_op_device(block, device):
    for op in block.ops:
        op._set_attr('op_device', device)


def screen_persistables(program, var_list):
    need_remove = []
    for var_name in var_list:
        if "@GRAD" in var_name:
            origin_var_name = var_name.split("@GRAD")[0]
            var = program.global_block().vars[origin_var_name]
        else:
            var = program.global_block().vars[var_name]

        if fluid.io.is_persistable(var):
            need_remove.append(var_name)

    for var_name in need_remove:
        var_list.remove(var_name)
    return need_remove


def insert_reshape_op(program,
                      block,
                      index,
                      var_name,
                      new_var_name,
                      new_var_shape=None):
    input_var = program.global_block().vars[var_name]

    if new_var_name not in program.global_block().vars:
        out = program.global_block().create_var(
            name=new_var_name,
            shape=new_var_shape,
            dtype=input_var.dtype,
            type=input_var.type)
    else:
        out = program.global_block().vars[new_var_name]
        new_var_shape = out.shape

    print("Insert Reshape op for: output var {}, input var {}, shape {}".format(
        new_var_name, var_name, new_var_shape))
    x_shape = program.global_block().create_var(
        name="{}@XShape".format(var_name), dtype=input_var.dtype)
    block._insert_op(
        index=index,
        type="reshape2",
        inputs={"X": input_var},
        attrs={'shape': new_var_shape},
        outputs={"Out": out,
                 "XShape": x_shape})


def insert_send_concat_op(program, block, index, var_name_list, new_var_name,
                          new_var_shape):
    input_var_list = [
        program.global_block().vars[var_name] for var_name in var_name_list
    ]

    out = program.global_block().create_var(
        name=new_var_name,
        shape=new_var_shape,
        dtype=input_var_list[0].dtype,
        type=input_var_list[0].type)

    block._insert_op(
        index=index,
        type='concat',
        inputs={"X": input_var_list},
        outputs={'Out': [out]},
        attrs={'axis': -1,
               'use_stack': False})


def insert_recv_slice_op(program, block, index, var_name, var_shape, dtype,
                         type, new_var_name_list, new_var_shape_list):

    if var_name not in program.global_block().vars:
        input_var = program.global_block().create_var(
            name=var_name, shape=var_shape, dtype=dtype, type=type)
    else:
        input_var = program.global_block().vars[var_name]

    out_list = []
    for i in range(len(new_var_name_list)):
        if new_var_name_list[i] not in program.global_block().vars:
            out = program.global_block().create_var(
                name=new_var_name_list[i],
                shape=new_var_shape_list[i],
                dtype=input_var.dtype,
                type=input_var.type)
        else:
            out = program.global_block().vars[new_var_name_list[i]]
        out_list.append(out)

    start_index = 0
    end_index = 0
    for i in range(len(new_var_name_list)):
        starts = []
        ends = []
        attrs = {'axes': [1]}
        end_index += new_var_shape_list[i][1]
        starts.append(start_index)
        ends.append(end_index)
        attrs['starts'] = starts
        attrs['ends'] = ends

        block._insert_op(
            index=index,
            type='slice',
            inputs={'Input': input_var},
            attrs=attrs,
            outputs={'Out': out_list[i]})
        start_index = end_index
        index += 1


def get_op_by_type(block, op_type):
    for op in block.ops:
        if op.type == op_type:
            return op
    raise ValueError("add_listen_and_serv_pass must at first")


def deleter_trainer_useless_var(program):
    porgram_useful_var_list = []
    for op in program.global_block().ops:
        input_var_list, output_var_list = find_op_input_output(
            program, program.global_block(), op)
        op_var_list = list(set(input_var_list).union(set(output_var_list)))
        porgram_useful_var_list = list(
            set(porgram_useful_var_list).union(set(op_var_list)))

    program_useless_var_list = list(
        set(get_vars_name_in_block(program.global_block())).difference(
            set(porgram_useful_var_list)))
    for var in program_useless_var_list:
        program.global_block()._remove_var(var)
    return program_useless_var_list


def block_append_op(program, origin_program, block, op):
    inputs = _get_input_map_from_op(origin_program.global_block().vars, op)
    for key, varlist in six.iteritems(inputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if var.name not in program.global_block().vars:
                program.global_block()._clone_variable(var)

    outputs = _get_output_map_from_op(origin_program.global_block().vars, op)
    for key, varlist in six.iteritems(outputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if var.name not in program.global_block().vars:
                program.global_block()._clone_variable(var)

    if "_grad" not in op.type:
        # for forward op
        return block.append_op(
            type=op.type, inputs=inputs, outputs=outputs, attrs=op.all_attrs())
    else:
        # for grad op
        op_desc = op.desc
        op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()

        # append grad op
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(op_desc)
        new_op_desc._set_attr(op_role_attr_name, backward)

        # set device gard
        if op.desc.has_attr(device_attr_name):
            op_device = op_desc.attr(device_attr_name)
            new_op_desc._set_attr(device_attr_name, op_device)
        block._sync_with_cpp()


def add_vars_by_op_map(var_map, program):
    for key, varlist in six.iteritems(var_map):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for i in range(len(varlist)):
            var = varlist[i]
            if var.name not in program.global_block().vars:
                program.global_block()._clone_variable(var)


def add_vars_by_var_list(var_name_list, origin_program, program):
    for var_name in var_name_list:
        if var_name not in program.global_block().vars:
            var = origin_program.global_block().vars[var_name]
            program.global_block()._clone_variable(var)


def get_varlist_from_op_map(var_map):
    var_list = []
    for key, varlist in six.iteritems(var_map):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for i in range(len(varlist)):
            var = varlist[i]
            var_list.append(var.name)
    return var_list


def find_ops_list_input_output(program, ops_list):
    input_var_list = []
    output_var_list = []
    for op in ops_list:
        inputs = _get_input_map_from_op(program.global_block().vars, op)
        input_var_list += get_varlist_from_op_map(inputs)
        outputs = _get_output_map_from_op(program.global_block().vars, op)
        output_var_list += get_varlist_from_op_map(outputs)

    input_var_list = list(set(input_var_list))
    output_var_list = list(set(output_var_list))
    return input_var_list, output_var_list


def find_op_input_output(program, block, op):
    input_var_list = []
    output_var_list = []
    inputs = _get_input_map_from_op(block.vars, op)
    input_var_list += get_varlist_from_op_map(inputs)
    outputs = _get_output_map_from_op(block.vars, op)
    output_var_list += get_varlist_from_op_map(outputs)
    input_var_list = list(set(input_var_list))
    output_var_list = list(set(output_var_list))
    return input_var_list, output_var_list


def get_vars_name_in_block(block):
    vars_list = block.vars.keys()
    vars_name_list = [var_name for var_name in vars_list]
    return vars_name_list


def is_same_op(op1, op2):
    if str(op1) != str(op2):
        return False
    return True


def _get_input_map_from_op(varmap, op):
    """Returns a dict from op input name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.input_names:
        vars = []
        for varname in op.input(key):
            if varname == "@EMPTY@":
                continue
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def _get_output_map_from_op(varmap, op):
    """Returns a dict from op output name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.output_names:
        vars = []
        for varname in op.output(key):
            if varname == "@EMPTY@":
                continue
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def delete_same_ops(block, ops):
    for op in ops:
        try:
            for origin_op in block.ops:
                if is_same_op(origin_op, op):
                    idx = list(block.ops).index(origin_op)
                    block._remove_op(idx)
                    break
        except Exception as e:
            print(e)


def delete_same_ops_in_list(op_list, ops):
    for op in ops:
        try:
            for origin_op in op_list:
                if is_same_op(origin_op, op):
                    op_list.remove(origin_op)
                    break
        except Exception as e:
            print(e)


class MergedVariable:
    def __init__(self, merged, ordered, offsets):
        self.merged_var = merged
        self.ordered_vars = ordered
        self.offsets = offsets


class CompileTimeStrategy(object):
    def __init__(self, main_program, startup_program, strategy, role_maker):

        self.min_block_size = 8192

        self.origin_main_program = main_program
        self.origin_startup_program = startup_program

        self.strategy = strategy
        self.role_maker = role_maker

        self.origin_sparse_pairs = []
        self.origin_dense_pairs = []

        self.merged_variables_pairs = []
        self.merged_dense_pairs = []
        self.merged_sparse_pairs = []

        self.merged_variable_map = {}
        self.param_name_to_grad_name = {}
        self.grad_name_to_param_name = {}

        self.param_grad_ep_mapping = collections.OrderedDict()
        self.grad_param_mapping = collections.OrderedDict()

        self._build_var_distributed()

    def get_distributed_mode(self):
        trainer = self.strategy.get_trainer_runtime_config()
        return trainer.mode

    def is_sync_mode(self):
        trainer = self.strategy.get_trainer_runtime_config()
        return trainer.mode == DistributedMode.SYNC

    def is_geo_mode(self):
        trainer = self.strategy.get_trainer_runtime_config()
        return trainer.mode == DistributedMode.GEO

    def is_async_mode(self):
        trainer = self.strategy.get_trainer_runtime_config()
        return trainer.mode == DistributedMode.ASYNC

    def get_role_id(self):
        return self.role_maker.role_id()

    def get_trainers(self):
        return self.role_maker.worker_num()

    def get_ps_endpoint(self):
        return self.role_maker.get_pserver_endpoints()[self.get_role_id()]

    def get_ps_endpoints(self):
        return self.role_maker.get_pserver_endpoints()

    def get_heter_worker_endpoints(self):
        return self.role_maker._get_heter_worker_endpoints()

    def get_heter_worker_endpoint(self):
        return self.role_maker._get_heter_worker_endpoint()

    def get_origin_programs(self):
        return self.origin_main_program, self.origin_startup_program

    def get_origin_main_program(self):
        return self.origin_main_program

    def get_origin_startup_program(self):
        return self.origin_startup_program

    def get_sparse_varname_on_ps(self, is_distributed, endpoint=None):
        if not endpoint:
            endpoint = self.get_ps_endpoint()

        varnames = get_sparse_tablenames(self.get_origin_main_program(),
                                         is_distributed)
        ps_sparse_varnames = []
        for varname in varnames:
            tables = self.get_var_distributed(varname, True)
            for i in range(len(tables)):
                table, ep, _ = tables[i]
                if ep == endpoint:
                    ps_sparse_varnames.append(table)
        return ps_sparse_varnames

    def build_ctx(self,
                  vars,
                  mapping,
                  is_grad,
                  is_sparse,
                  is_send,
                  is_distributed=False):
        def get_grad_var_ep(slices):
            names = []
            eps = []
            sections = []

            for slice in slices:
                if self.is_geo_mode():
                    if is_send:
                        names.append("{}.delta".format(slice.name))
                    else:
                        names.append(slice.name)
                elif is_grad and self.is_sync_mode() and self.get_trainers(
                ) > 1:
                    names.append("{}.trainer_{}".format(slice.name,
                                                        self.get_role_id()))
                else:
                    names.append(slice.name)

                sections.append(slice.shape[0])

                for ep, pairs in self.param_grad_ep_mapping.items():
                    params, grads = pairs["params"], pairs["grads"]

                    for var in params + grads:
                        if slice.name == var.name:
                            eps.append(ep)
                            break
            return names, eps, sections

        if isinstance(vars, MergedVariable):
            name = vars.merged_var.name
            slices = mapping[name]
            names, eps, sections = get_grad_var_ep(slices)
            origin_varnames = [var.name for var in vars.ordered_vars]
        else:
            name = vars.name
            slices = mapping[name]
            names, eps, sections = get_grad_var_ep(slices)
            origin_varnames = [vars.name]

        trainer_id = self.get_role_id()
        aggregate = True
        ctx = CommContext(name, names, eps, sections, origin_varnames,
                          trainer_id, aggregate, is_sparse, is_distributed)
        return ctx

    def get_trainer_send_context(self):
        send_ctx = {}
        distibuted_varnames = get_sparse_tablenames(self.origin_main_program,
                                                    True)

        if not self.is_geo_mode():
            for merged in self.merged_dense_pairs:
                grad = merged[1]
                ctx = self.build_ctx(grad, self.grad_var_mapping, True, False,
                                     True)
                send_ctx[ctx.var_name()] = ctx

            for merged in self.merged_sparse_pairs:
                param = merged[0]
                grad = merged[1]

                param_name = param.merged_var.name

                is_distributed = True if param_name in distibuted_varnames else False

                ctx = self.build_ctx(grad, self.grad_var_mapping, True, True,
                                     True, is_distributed)
                send_ctx[ctx.var_name()] = ctx

            if self.is_async_mode():
                name, ctx = self._step_ctx()
                send_ctx[name] = ctx
        else:
            for pairs in self.origin_sparse_pairs:
                param, grad = pairs
                param_name = param.name
                is_distributed = True if param_name in distibuted_varnames else False

                param_ctx = self.build_ctx(param, self.param_var_mapping, False,
                                           True, True, is_distributed)
                grad_ctx = self.build_ctx(grad, self.grad_var_mapping, True,
                                          True, True, is_distributed)

                ctx = CommContext(param_ctx.var_name(),
                                  param_ctx.split_varnames(),
                                  param_ctx.split_endpoints(),
                                  param_ctx.sections(),
                                  grad_ctx.origin_varnames(),
                                  param_ctx.trainer_id(),
                                  param_ctx.aggregate(),
                                  param_ctx.is_sparse(),
                                  param_ctx.is_distributed())

                send_ctx[ctx.var_name()] = ctx
            name, ctx = self._step_ctx()
            send_ctx[name] = ctx
        return send_ctx

    def get_communicator_send_context(self):
        send_ctx = {}
        distibuted_varnames = get_sparse_tablenames(self.origin_main_program,
                                                    True)

        if self.is_geo_mode():
            for pairs in self.merged_dense_pairs:
                param = pairs[0]
                ctx = self.build_ctx(param, self.param_var_mapping, False,
                                     False, True)
                send_ctx[ctx.var_name()] = ctx

            for pairs in self.merged_sparse_pairs:
                param = pairs[0]
                param_name = param.merged_var.name
                is_distributed = True if param_name in distibuted_varnames else False

                ctx = self.build_ctx(param, self.param_var_mapping, False, True,
                                     True, is_distributed)
                send_ctx[ctx.var_name()] = ctx
            name, ctx = self._step_ctx()
            send_ctx[name] = ctx
        else:
            for merged in self.merged_dense_pairs:
                grad = merged[1]
                ctx = self.build_ctx(grad, self.grad_var_mapping, True, False,
                                     True)
                send_ctx[ctx.var_name()] = ctx

            for merged in self.merged_sparse_pairs:
                param, grad = merged
                param_name = param.merged_var.name

                is_distributed = True if param_name in distibuted_varnames else False

                ctx = self.build_ctx(grad, self.grad_var_mapping, True, True,
                                     True, is_distributed)
                send_ctx[ctx.var_name()] = ctx

            name, ctx = self._step_ctx()
            send_ctx[name] = ctx
        return send_ctx

    def get_communicator_recv_context(self, recv_type=1):
        # recv_type
        # 1 : DENSE 2. SPARSE 3. DISTRIBUTED 4. ALL
        distibuted_varnames = get_sparse_tablenames(self.origin_main_program,
                                                    True)
        sparse_varnames = []
        for pairs in self.origin_sparse_pairs:
            param, grad = pairs
            sparse_varnames.append(param.name)

        dense_recv_ctx = {}
        sparse_recv_ctx = {}
        distributed_recv_ctx = {}

        for merged in self.merged_variables_pairs:
            params = merged[0]
            if params.merged_var.name in sparse_varnames:
                continue

            ctx = self.build_ctx(params, self.param_var_mapping, False, False,
                                 False)
            dense_recv_ctx[ctx.var_name()] = ctx

        for pairs in self.origin_sparse_pairs:
            param, grad = pairs

            if param.name in distibuted_varnames:
                ctx = self.build_ctx(param, self.param_var_mapping, False, True,
                                     False, True)
                distributed_recv_ctx[ctx.var_name()] = ctx
            else:
                ctx = self.build_ctx(param, self.param_var_mapping, False, True,
                                     False, False)
                sparse_recv_ctx[ctx.var_name()] = ctx

        if recv_type == 1:
            return dense_recv_ctx
        if recv_type == 2:
            return sparse_recv_ctx
        if recv_type == 3:
            return distributed_recv_ctx
        if recv_type == 4:
            dense_recv_ctx.update(sparse_recv_ctx)
            dense_recv_ctx.update(distributed_recv_ctx)
            return dense_recv_ctx
        assert ValueError(
            "recv_type can only be 1/2/3/4, 1 : DENSE 2. SPARSE 3. DISTRIBUTED 4. ALL"
        )

    def get_server_runtime_config(self):
        return self.strategy.get_server_runtime_config()

    def get_var_distributed(self, varname, is_param):
        var_distributed = []
        offset = 0
        if is_param:
            params = self.param_var_mapping[varname]
            param_varnames = [var.name for var in params]
            for ep, pairs in self.param_grad_ep_mapping.items():
                for p in pairs["params"]:
                    if p.name in param_varnames:
                        offset += p.shape[0]
                        var_distributed.append((p.name, ep, p.shape[0]))
        else:
            grads = self.grad_var_mapping[varname]
            grad_varnames = [var.name for var in grads]
            for ep, pairs in self.param_grad_ep_mapping.items():
                for g in pairs["grads"]:
                    if g.name in grad_varnames:
                        var_distributed.append((g.name, ep, g.shape[0]))
        return var_distributed

    def _step_ctx(self):
        name = STEP_COUNTER
        trainer_id = self.get_role_id()
        endpoints = self.get_ps_endpoints()
        sections = [1] * len(endpoints)
        names = [name] * len(endpoints)
        ctx = CommContext(name, names, endpoints, sections, [name], trainer_id,
                          True, False, False)
        return name, ctx

    def _create_vars_from_blocklist(self, block_list):
        """
        Create vars for each split.
        NOTE: only grads need to be named for different trainers, use
              add_trainer_suffix to rename the grad vars.
        Args:
            block_list (list[(varname, block_id, block_size)]): List of gradient blocks.
            add_trainer_suffix (Bool): Add trainer suffix to new variable's name if set True.
        Returns:
            var_mapping (collections.OrderedDict(varname->[new_varname_variable])):A dict mapping
                from original var name to each var split.
        """

        # varname->[(block_id, current_block_size)]
        block_map = collections.OrderedDict()
        var_mapping = collections.OrderedDict()

        for block_str in block_list:
            varname, offset, size = block_str.split(":")
            if varname not in block_map:
                block_map[varname] = []
            block_map[varname].append((int(offset), int(size)))

        for varname, split in six.iteritems(block_map):
            orig_var = self.merged_variable_map[varname]

            if len(split) == 1:
                var_mapping[varname] = [orig_var]
                self.var_distributed.add_distributed_var(
                    origin_var=orig_var,
                    slice_var=orig_var,
                    block_id=0,
                    offset=0,
                    is_slice=False,
                    vtype="Param")
            else:
                var_mapping[varname] = []
                orig_shape = orig_var.shape
                orig_dim1_flatten = 1

                if len(orig_shape) >= 2:
                    orig_dim1_flatten = reduce(lambda x, y: x * y,
                                               orig_shape[1:])

                for i, block in enumerate(split):
                    size = block[1]
                    rows = size // orig_dim1_flatten
                    splited_shape = [rows]
                    if len(orig_shape) >= 2:
                        splited_shape.extend(orig_shape[1:])

                    new_var_name = "%s.block%d" % (varname, i)
                    slice_var = vars_metatools.VarStruct(
                        name=new_var_name,
                        shape=splited_shape,
                        dtype=orig_var.dtype,
                        type=orig_var.type,
                        lod_level=orig_var.lod_level,
                        persistable=False)
                    var_mapping[varname].append(slice_var)

                    self.var_distributed.add_distributed_var(
                        origin_var=orig_var,
                        slice_var=slice_var,
                        block_id=i,
                        offset=-1,
                        is_slice=False,
                        vtype="Param")

        return var_mapping

    def _dispatcher(self):
        ps_dispatcher = RoundRobin(self.get_ps_endpoints())
        ps_dispatcher.reset()
        grad_var_mapping_items = list(six.iteritems(self.grad_var_mapping))

        sparse_gradnames = [grad.name for _, grad in self.origin_sparse_pairs]

        for grad_varname, splited_vars in grad_var_mapping_items:
            if grad_varname in sparse_gradnames:
                continue

            send_vars = []
            for _, var in enumerate(splited_vars):
                send_vars.append(var)

            recv_vars = []
            for _, var in enumerate(send_vars):
                recv_vars.append(self.grad_param_mapping[var])

            eps = ps_dispatcher.dispatch(recv_vars)

            for i, ep in enumerate(eps):
                self.param_grad_ep_mapping[ep]["params"].append(recv_vars[i])
                self.param_grad_ep_mapping[ep]["grads"].append(send_vars[i])

        for grad_varname, splited_vars in grad_var_mapping_items:
            if grad_varname not in sparse_gradnames:
                continue

            ps_dispatcher.reset()

            send_vars = []
            for _, var in enumerate(splited_vars):
                send_vars.append(var)

            recv_vars = []
            for _, var in enumerate(send_vars):
                recv_vars.append(self.grad_param_mapping[var])

            eps = ps_dispatcher.dispatch(recv_vars)

            for i, ep in enumerate(eps):
                self.param_grad_ep_mapping[ep]["params"].append(recv_vars[i])
                self.param_grad_ep_mapping[ep]["grads"].append(send_vars[i])

    def _slice_variable(self,
                        var_list,
                        slice_count,
                        min_block_size,
                        uniform=False):
        """
        We may need to split dense tensor to one or more blocks and put
        them equally onto parameter server. One block is a sub-tensor
        aligned by dim[0] of the tensor.

        We need to have a minimal block size so that the calculations in
        the parameter server side can gain better performance. By default
        minimum block size 8K elements (maybe 16bit or 32bit or 64bit).

        Args:
            var_list (list): List of variables.
            slice_count (int): Numel of count that variables will be sliced, which
                could be the pserver services' count.
            min_block_size (int): Minimum split block size.
        Returns:
            blocks (list[(varname, block_id, current_block_size)]): A list
                of VarBlocks. Each VarBlock specifies a shard of the var.
        """
        blocks = []
        for var in var_list:
            if not uniform:
                var_numel = reduce(lambda x, y: x * y, var.shape)

                split_count = 1

                # if min_block_size == -1:
                #     split_count = 1
                # else:
                #     split_count = slice_count
                #     max_pserver_count = int(
                #         math.floor(var_numel / float(min_block_size)))
                #     if max_pserver_count == 0:
                #         max_pserver_count = 1
                #     if max_pserver_count < slice_count:
                #         split_count = max_pserver_count
                block_size = int(math.ceil(var_numel / float(split_count)))

                if len(var.shape) >= 2:
                    # align by dim1(width)
                    dim1 = reduce(lambda x, y: x * y, var.shape[1:])
                    remains = block_size % dim1
                    if remains != 0:
                        block_size += dim1 - remains
                        # update split_count after aligning
                split_count = int(math.ceil(var_numel / float(block_size)))
                for block_id in range(split_count):
                    curr_block_size = min(block_size, var_numel - (
                        (block_id) * block_size))
                    block = vars_metatools.VarBlock(var.name, block_id,
                                                    curr_block_size)
                    blocks.append(str(block))
            else:
                block_size = var.shape[0] / slice_count
                remainder = var.shape[0] % slice_count

                if block_size == 0:
                    dim0s = [block_size] * remainder
                else:
                    dim0s = [block_size] * slice_count
                for i in range(remainder):
                    dim0s[i] = dim0s[i] + 1

                dim1 = reduce(lambda x, y: x * y, var.shape[1:])

                for block_id in range(len(dim0s)):
                    numel = dim0s[block_id] * dim1
                    block = vars_metatools.VarBlock(var.name, block_id, numel)
                    blocks.append(str(block))
        return blocks

    def _get_param_grad_blocks(self, pairs, min_block_size, uniform=False):
        param_list = []
        grad_list = []
        param_grad_set = set()
        for p, g in pairs:
            # todo(tangwei12) skip parameter marked not trainable
            # if type(p) == Parameter and p.trainable == False:
            # continue
            p = p.merged_var
            g = g.merged_var

            if p.name not in param_grad_set:
                param_list.append(p)
                param_grad_set.add(p.name)
            if g.name not in param_grad_set:
                grad_list.append(g)
                param_grad_set.add(g.name)

                # when we slice var up into blocks, we will slice the var according to
                # pserver services' count. A pserver may have two or more listening ports.
        grad_blocks = self._slice_variable(grad_list,
                                           len(self.get_ps_endpoints()),
                                           min_block_size, uniform)

        param_blocks = self._slice_variable(param_list,
                                            len(self.get_ps_endpoints()),
                                            min_block_size, uniform)
        return param_blocks, grad_blocks

    def _var_slice_and_distribute(self):
        # update these mappings for further transpile:
        # 1. param_var_mapping : param var name->[split params vars]
        # 2. grad_var_mapping : grad var name->[split grads vars]
        # 3. grad_param_mapping : grad.blockx->param.blockx
        # 4. param_grad_ep_mapping : ep->{"params" : [], "grads" : [] }

        dps, dgs = self._get_param_grad_blocks(self.merged_dense_pairs, -1,
                                               False)
        sps, sgs = self._get_param_grad_blocks(self.merged_sparse_pairs,
                                               self.min_block_size, True)

        param_blocks = dps + sps
        grad_blocks = dgs + sgs

        assert (len(grad_blocks) == len(param_blocks))

        # origin_param_name->[splited_param_vars]
        self.param_var_mapping = self._create_vars_from_blocklist(param_blocks)
        self.grad_var_mapping = self._create_vars_from_blocklist(grad_blocks)

        # dict(grad_splited_var->param_splited_var)
        self.grad_param_mapping = collections.OrderedDict()
        for g, p in zip(grad_blocks, param_blocks):
            g_name, g_bid, _ = g.split(":")
            p_name, p_bid, _ = p.split(":")
            self.grad_param_mapping[self.grad_var_mapping[g_name][int(g_bid)]] = \
                self.param_var_mapping[p_name][int(p_bid)]

        print_maps = {}
        for k, v in self.grad_param_mapping.items():
            print_maps[str(k)] = str(v)

        # create mapping of endpoint->split var to create pserver side program
        self.param_grad_ep_mapping = collections.OrderedDict()
        [
            self.param_grad_ep_mapping.update({
                ep: {
                    "params": [],
                    "grads": []
                }
            }) for ep in self.get_ps_endpoints()
        ]

    def _build_var_distributed(self):
        self.var_distributed = vars_metatools.VarsDistributed()

        sparse_pairs, dense_pairs = self.get_param_grads()
        origin_for_sparse = []
        origin_for_dense = []
        param_name_grad_name = dict()
        grad_name_to_param_name = dict()

        for param, grad in sparse_pairs:
            param = vars_metatools.create_var_struct(param)
            grad = vars_metatools.create_var_struct(grad)
            origin_for_sparse.append((param, grad))

        for param, grad in dense_pairs:
            param = vars_metatools.create_var_struct(param)
            grad = vars_metatools.create_var_struct(grad)
            origin_for_dense.append((param, grad))

        for dense_pair in origin_for_dense:
            param, grad = dense_pair

            m_param = MergedVariable(param, [param], [0])
            m_grad = MergedVariable(grad, [grad], [0])
            self.merged_variables_pairs.append((m_param, m_grad))
            self.merged_dense_pairs.append((m_param, m_grad))

        for sparse_pair in origin_for_sparse:
            param, grad = sparse_pair

            m_param = MergedVariable(param, [param], [0])
            m_grad = MergedVariable(grad, [grad], [0])
            self.merged_variables_pairs.append((m_param, m_grad))
            self.merged_sparse_pairs.append((m_param, m_grad))

        for merged in self.merged_variables_pairs:
            m_param, m_grad = merged
            self.merged_variable_map[
                m_param.merged_var.name] = m_param.merged_var
            self.merged_variable_map[m_grad.merged_var.name] = m_grad.merged_var

        param_merges = []
        param_merges.extend(origin_for_sparse)
        param_merges.extend(origin_for_dense)

        for param, grad in param_merges:
            param_name_grad_name[param.name] = grad.name
            grad_name_to_param_name[grad.name] = param.name

        self.origin_sparse_pairs = origin_for_sparse
        self.origin_dense_pairs = origin_for_dense
        self.param_name_to_grad_name = param_name_grad_name
        self.grad_name_to_param_name = grad_name_to_param_name

        sparse_pair_map = collections.OrderedDict()

        for pair in self.origin_sparse_pairs + self.origin_dense_pairs:
            param, grad = pair
            sparse_pair_map[param.name] = str(param)
            sparse_pair_map[grad.name] = str(grad)

        self._var_slice_and_distribute()
        self._dispatcher()

    def get_param_grads(self):
        origin_program = self.origin_main_program

        def _get_params_grads(sparse_varnames):
            block = origin_program.global_block()

            dense_param_grads = []
            sparse_param_grads = []

            optimize_params = set()
            origin_var_dict = origin_program.global_block().vars
            role_id = int(core.op_proto_and_checker_maker.OpRole.Backward)
            for op in block.ops:
                if _is_opt_role_op(op):
                    # delete clip op from opt_ops when run in Parameter Server mode
                    if OP_NAME_SCOPE in op.all_attrs() \
                            and CLIP_OP_NAME_SCOPE in op.attr(OP_NAME_SCOPE):
                        op._set_attr("op_role", role_id)
                        continue
                    if op.attr(OP_ROLE_VAR_ATTR_NAME):
                        param_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[0]
                        grad_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[1]
                        if param_name not in optimize_params:
                            optimize_params.add(param_name)
                            param_grad = (origin_var_dict[param_name],
                                          origin_var_dict[grad_name])

                            if param_name in sparse_varnames:
                                sparse_param_grads.append(param_grad)
                            else:
                                dense_param_grads.append(param_grad)
            return sparse_param_grads, dense_param_grads

        def _get_sparse_varnames():
            varnames = []
            op_types = {"lookup_table": "W"}
            for op in origin_program.global_block().ops:
                if op.type in op_types.keys() \
                        and op.attr('remote_prefetch') is True:
                    param_name = op.input(op_types[op.type])[0]
                    varnames.append(param_name)

            return list(set(varnames))

        sparse_varnames = _get_sparse_varnames()
        sparse_param_grads, dense_param_grads = _get_params_grads(
            sparse_varnames)

        return sparse_param_grads, dense_param_grads

    def remove_var_pair_by_grad(self, var_name):

        for index, pair in enumerate(self.merged_variables_pairs):
            var = pair[0]
            var_grad = pair[1]
            if var_grad.merged_var.name == var_name:
                del self.merged_variables_pairs[index]

        for index, pair in enumerate(self.merged_dense_pairs):
            var = pair[0]
            var_grad = pair[1]
            if var_grad.merged_var.name == var_name:
                del self.merged_dense_pairs[index]
                return

        for index, pair in enumerate(self.merged_sparse_pairs):
            var = pair[0]
            var_grad = pair[1]
            if var_grad.merged_var.name == var_name:
                del self.merged_sparse_pairs[index]
                return

        print("Not find {} in self.merge_pairs".format(var_name))


def _is_opt_role_op(op):
    # NOTE : depend on oprole to find out whether this op is for
    # optimize
    op_maker = core.op_proto_and_checker_maker
    optimize_role = core.op_proto_and_checker_maker.OpRole.Optimize
    if op_maker.kOpRoleAttrName() in op.attr_names and \
            int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(optimize_role):
        return True
    return False


def _get_optimize_ops(_program):
    block = _program.global_block()
    opt_ops = []
    for op in block.ops:
        if _is_opt_role_op(op):
            # delete clip op from opt_ops when run in Parameter Server mode
            if OP_NAME_SCOPE in op.all_attrs() \
                    and CLIP_OP_NAME_SCOPE in op.attr(OP_NAME_SCOPE):
                op._set_attr(
                    "op_role",
                    int(core.op_proto_and_checker_maker.OpRole.Backward))
                continue
            opt_ops.append(op)
    return opt_ops


def _get_varname_parts(varname):
    # returns origin, blockid, trainerid
    orig_var_name = ""
    trainer_part = ""
    block_part = ""
    trainer_idx = varname.find(".trainer_")
    if trainer_idx >= 0:
        trainer_part = varname[trainer_idx + 1:]
    else:
        trainer_idx = len(varname)
    block_index = varname.find(".block")
    if block_index >= 0:
        block_part = varname[block_index + 1:trainer_idx]
    else:
        block_index = len(varname)
    orig_var_name = varname[0:min(block_index, trainer_idx)]
    return orig_var_name, block_part, trainer_part


def _orig_varname(varname):
    orig, _, _ = _get_varname_parts(varname)
    return orig
