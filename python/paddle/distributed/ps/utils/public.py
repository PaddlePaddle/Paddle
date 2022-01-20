# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import logging
import six
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.core import CommContext
import paddle.fluid.framework as framework
from paddle.fluid.incubate.fleet.parameter_server.ir import vars_metatools
from paddle.fluid.incubate.fleet.parameter_server.ir.ps_dispatcher import RoundRobin, PSDispatcher
from paddle.fluid.transpiler.details.program_utils import delete_ops
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "gradient_clip"
STEP_COUNTER = "@PS_STEP_COUNTER@"
LEARNING_RATE_DECAY_COUNTER = "@LR_DECAY_COUNTER@"

OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize

SPARSE_OP_LIST = ["lookup_table", "lookup_table_v2"]
SPARSE_OP_TYPE_DICT = {"lookup_table": "W", "lookup_table_v2": "W"}


class DistributedMode:
    SYNC = 0
    ASYNC = 1
    HALF_ASYNC = 2
    GEO = 3
    FL = 4


class TrainerRuntimeConfig(object):
    def __init__(self, valid_strategy):

        k_steps = valid_strategy.a_sync_configs["k_steps"]
        if not valid_strategy.a_sync and k_steps == 0:
            self.mode = DistributedMode.SYNC

        if valid_strategy.a_sync and k_steps == 0:
            self.mode = DistributedMode.ASYNC

        if valid_strategy.a_sync and k_steps > 0:
            self.mode = DistributedMode.GEO

        self.mode = None
        num_threads = os.getenv("CPU_NUM", "1")

        self.runtime_configs = {}
        self.runtime_configs['communicator_max_merge_var_num'] = os.getenv(
            "FLAGS_communicator_max_merge_var_num", num_threads)
        self.runtime_configs['communicator_send_queue_size'] = os.getenv(
            "FLAGS_communicator_send_queue_size", num_threads)
        self.runtime_configs[
            'communicator_independent_recv_thread'] = os.getenv(
                "FLAGS_communicator_independent_recv_thread", "1")
        self.runtime_configs[
            'communicator_min_send_grad_num_before_recv'] = os.getenv(
                "FLAGS_communicator_min_send_grad_num_before_recv", num_threads)
        self.runtime_configs['communicator_thread_pool_size'] = os.getenv(
            "FLAGS_communicator_thread_pool_size", "5")
        self.runtime_configs['communicator_send_wait_times'] = os.getenv(
            "FLAGS_communicator_send_wait_times", "5")
        self.runtime_configs['communicator_is_sgd_optimizer'] = os.getenv(
            "FLAGS_communicator_is_sgd_optimizer", "1")


def get_lr_ops(program):
    lr_ops = []
    for index, op in enumerate(program.global_block().ops):
        role_id = int(op.attr(RPC_OP_ROLE_ATTR_NAME))
        if role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) or \
                role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) | \
                int(OPT_OP_ROLE_ATTR_VALUE):
            lr_ops.append(op)
    return lr_ops


def get_optimize_ops(_program):
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


def get_dist_env():
    trainer_id = int(os.getenv('PADDLE_TRAINER_ID', '0'))
    trainer_endpoints = ''
    current_endpoint = ''
    num_trainers = 0
    if os.getenv('PADDLE_TRAINER_ENDPOINTS'):
        trainer_endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS')
        current_endpoint = trainer_endpoints.split(',')[trainer_id]
        num_trainers = len(trainer_endpoints.split(','))

    return {
        'trainer_id': trainer_id,
        'num_trainers': num_trainers,
        'current_endpoint': current_endpoint,
        'trainer_endpoints': trainer_endpoints
    }


def get_ps_endpoint(role_maker):
    try:
        return role_maker._get_pserver_endpoints()[get_role_id(role_maker)]
    except Exception:
        return role_maker.get_pserver_endpoints()[get_role_id(role_maker)]


def get_heter_worker_endpoint(role_maker):
    try:
        return role_maker._get_heter_worker_endpoint()
    except Exception:
        return role_maker.get_heter_worker_endpoint()


def get_trainer_endpoint(role_maker):
    try:
        return role_maker._get_trainer_endpoint()
    except Exception:
        return role_maker.get_trainer_endpoint()


def get_previous_stage_trainers(role_maker):
    try:
        return role_maker_get_previous_trainers()
    except Exception:
        return role_maker.get_previous_trainers()


def is_distributed_sparse_op(op):
    if op.type in SPARSE_OP_LIST and op.attr('is_distributed') is True:
        return True

    if op.type == "distributed_lookup_table" and op.attr(
            'is_distributed') is True:
        return True

    return False


def get_sparse_tablename(op):
    return op.input("W")[0]


def is_sparse_op(op):
    if op.type in SPARSE_OP_LIST and op.attr('is_sparse') is True and op.attr(
            'is_distributed') is False:
        return True

    if op.type == "distributed_lookup_table" and op.attr(
            'is_distributed') is False:
        return True

    return False


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


def get_role_id(role_maker):
    try:
        return role_maker._role_id()
    except Exception:
        return role_maker.role_id()


def get_ps_endpoints(role_maker):
    try:
        return role_maker._get_pserver_endpoints()[get_role_id(role_maker)]
    except Exception:
        return role_maker.get_pserver_endpoints()[get_role_id(role_maker)]


def get_trainers(role_maker):
    try:
        return role_maker._worker_num()
    except Exception:
        return role_maker.worker_num()


def get_dense_send_context(context,
                           send_ctx,
                           idx,
                           merged_dense_pairs,
                           trainer_id,
                           split_dense_table=False):
    if len(merged_dense_pairs) < 1:
        return idx
    if not split_dense_table:
        origin_varnames = []
        var_numel = 0
        for merged in merged_dense_pairs:
            grad = merged[1]
            origin_varnames.append(grad.merged_var.name)
            var = context['origin_main_program'].global_block().vars[
                grad.merged_var.name]
            var_numel += reduce(lambda x, y: x * y, var.shape)
        grad_name = "Dense@Grad"
        trainer_id = get_role_id(context['role_maker'])
        aggregate = True
        dense_ctx = CommContext(grad_name, [grad_name], ["127.0.0.1:6071"],
                                [var_numel], origin_varnames, trainer_id,
                                aggregate, False, False, idx, False)
        send_ctx[grad_name] = dense_ctx
        idx += 1
    else:
        for merged in merged_dense_pairs:
            grad = merged[1]
            origin_varname = grad.merged_var.name
            var = context['origin_main_program'].global_block().vars[
                origin_varname]
            var_numel = reduce(lambda x, y: x * y, var.shape)
            grad_name = origin_varname
            aggregate = True
            dense_ctx = CommContext(grad_name, [grad_name], ["127.0.0.1:6071"],
                                    [var_numel], [origin_varname], trainer_id,
                                    aggregate, False, False, idx, False)
            send_ctx[grad_name] = dense_ctx
            idx += 1
    return idx


def get_geo_trainer_send_context(context):
    if context['ps_mode'] != DistributedMode.GEO:
        raise ValueError("ps mode: {} not matched {}",
                         format(ps_mode, "get_geo_trainer_send_context"))

    send_ctx = {}
    return send_ctx


def _step_ctx(idx, role_maker):
    name = STEP_COUNTER
    trainer_id = get_role_id(role_maker)
    endpoints = get_ps_endpoints(role_maker)
    sections = [1] * len(endpoints)
    names = [name] * len(endpoints)
    ctx = CommContext(name, names, endpoints, sections, [name], trainer_id,
                      True, False, False, idx, True)
    return name, ctx


def get_the_one_send_context(context,
                             split_dense_table=False,
                             use_origin_program=False,
                             ep_list=None):
    send_ctx = {}
    idx = 0

    if len(context['tensor_table']) > 0 and context['is_worker']:
        name, ctx = _step_ctx(idx, context['role_maker'])
        send_ctx[name] = ctx

    return send_ctx


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
            # for distributed communciate ops: send & recv & barrier etc.
            # Todo: need update this method
            #op._set_attr('op_device', current_heter_device)
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
    '''
       re-place sum op to fix bug for union forward backward op
    '''
    var2idx = {}
    op_list = list(block.ops)
    op_size = len(op_list)

    for i in range(op_size - 1, -1, -1):
        op_list = list(block.ops)
        op = op_list[i]
        if "_grad" in op.type:
            forward_op_type = op.type.split("_grad")[0]
            if forward_op_type in SPARSE_OP_TYPE_DICT.keys() \
                and op.attr('remote_prefetch') is True:
                param_name = op.input(SPARSE_OP_TYPE_DICT[forward_op_type])[0]
                if param_name in var2idx:
                    ## insert sum op & remove sum op from var2idx and origin place
                    op_list = list(block.ops)
                    sum_op = op_list[var2idx[param_name]]
                    sum_op_inputs = {
                        sum_op.input_names[0]: [
                            block.vars[input]
                            for input in sum_op.input_arg_names
                        ]
                    }
                    sum_op_outputs = {
                        sum_op.output_names[0]: [
                            block.vars[output]
                            for output in sum_op.output_arg_names
                        ]
                    }
                    block._insert_op(
                        index=i + 1,
                        type=sum_op.type,
                        inputs=sum_op_inputs,
                        outputs=sum_op_outputs,
                        attrs=sum_op.all_attrs())
                    block._remove_op(var2idx[param_name] + 1)
                    var2idx.pop(param_name)
                    for var_ in var2idx:
                        var2idx[var_] += 1
            elif forward_op_type == "elementwise_mul":
                """
                get output varname of pre op

                """
                output_vars_no_grad = []
                for key in op.output_names:
                    for varname in op.output(key):
                        if varname == "@EMPTY@":
                            continue
                        if "lod_tensor_blocking_queue" in varname:
                            continue
                        output_vars_no_grad.append(varname.split("@GRAD")[0])
                for no_grad_var in output_vars_no_grad:
                    if no_grad_var in var2idx:
                        """
                       insert sum op & remove sum op from var2idx and origin place
  
                       """
                        op_list = list(block.ops)
                        sum_op = op_list[var2idx[no_grad_var]]
                        sum_op_inputs = {
                            sum_op.input_names[0]: [
                                block.vars[input]
                                for input in sum_op.input_arg_names
                            ]
                        }
                        sum_op_outputs = {
                            sum_op.output_names[0]: [
                                block.vars[output]
                                for output in sum_op.output_arg_names
                            ]
                        }
                        block._insert_op(
                            index=i + 1,
                            type=sum_op.type,
                            inputs=sum_op_inputs,
                            outputs=sum_op_outputs,
                            attrs=sum_op.all_attrs())
                        block._remove_op(var2idx[no_grad_var] + 1)
                        var2idx.pop(no_grad_var)
                        for var_ in var2idx:
                            var2idx[var_] += 1
        else:
            if op.type == "sum":
                var = op.output("Out")[0]
                if "@GRAD" in var:
                    origin_var = var.split("@GRAD")[0]
                    pre_op = op_list[i - 1]
                    if "_grad" in pre_op.type:
                        forward_op_type = pre_op.type.split("_grad")[0]
                        if forward_op_type in SPARSE_OP_TYPE_DICT.keys() \
                            and pre_op.attr('remote_prefetch') is True:
                            param_name = pre_op.input(SPARSE_OP_TYPE_DICT[
                                forward_op_type])[0]
                            if param_name == origin_var and op.attr(
                                    "op_device") == pre_op.attr("op_device"):
                                continue
                            else:
                                var2idx[origin_var] = i
                        elif forward_op_type == "elementwise_mul":
                            output_vars = []
                            for key in pre_op.output_names:
                                for varname in pre_op.output(key):
                                    if varname == "@EMPTY@":
                                        continue
                                    if "lod_tensor_blocking_queue" in varname:
                                        continue
                                    output_vars.append(varname)
                            input_vars = []
                            for key in op.input_names:
                                for varname in op.input(key):
                                    if varname == "@EMPTY@":
                                        continue
                                    if "lod_tensor_blocking_queue" in varname:
                                        continue
                                    input_vars.append(varname)
                            is_match = False
                            for varname in output_vars:
                                if varname in input_vars:
                                    is_match = True
                                    break
                            if is_match:
                                continue
                            else:
                                var2idx[origin_var] = i
                    else:
                        var2idx[origin_var] = i

    origin_porgram = program.clone()
    block = program.global_block()

    program_block_ops = []
    default_ops = {default_device: {}}
    heter_ops = {}
    block_index = 0

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


def union_forward_gradient_op(program_block_ops_list):
    """
    before analyzing the input & output of each block in program_block_list, we should
    union the forward op and corresponding gradient op to elimincate the uneccessary variable
    transmit
    """
    """
    fix for 2emb model, re-place sum op

    """
    block_length = len(program_block_ops_list)
    union_program_block_ops_list = []
    assert block_length % 2 != 0, "the length of program_block_ops_list should be odd"
    for i in range(0, block_length // 2):
        block_op_list = {"forward": program_block_ops_list[i]}
        block_op_list.update({
            "backward": program_block_ops_list[block_length - 1 - i]
        })
        union_program_block_ops_list.append(block_op_list)

    block_op_list = {"forward": [], "backward": []}
    for op in program_block_ops_list[block_length // 2]:
        if not "_grad" in op.type and not (op.type == "sum"):
            block_op_list["forward"].append(op)
        else:
            block_op_list["backward"].append(op)
    union_program_block_ops_list.append(block_op_list)
    return union_program_block_ops_list


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
        ## forward
        block_input, block_output = find_ops_list_input_output(
            program, block_op_list["forward"])
        persistables = screen_persistables(
            program, block_input) + screen_persistables(program, block_output)
        # find entrance & exit
        block_private_vars = list(set(block_input) & set(block_output))
        block_entrance = list(set(block_input) - set(block_private_vars))
        block_exit = list(set(block_output) - set(block_private_vars))
        detail = {
            "forward": {
                "entrance": block_entrance,
                "exit": block_exit,
                "private": block_private_vars,
                "persistables": persistables
            }
        }

        ## backward
        bp_block_input, bp_block_output = find_ops_list_input_output(
            program, block_op_list["backward"])
        bp_persistables = screen_persistables(
            program, bp_block_input) + screen_persistables(program,
                                                           bp_block_output)
        # find entrance & exit
        bp_block_private_vars = list(set(bp_block_input) & set(bp_block_output))
        bp_block_entrance = list(
            set(bp_block_input) - set(bp_block_private_vars))
        bp_block_exit = list(set(bp_block_output) - set(bp_block_private_vars))
        detail.update({
            "backward": {
                "entrance": bp_block_entrance,
                "exit": bp_block_exit,
                "private": bp_block_private_vars,
                "persistables": bp_persistables
            }
        })
        block_var_detail.append(detail)
    return block_var_detail


def entrance_exit_check(program, program_block_ops_list, block_var_detail,
                        heter_ops):
    for index in range(len(block_var_detail) - 1, -1, -1):
        if index - 1 < 0:
            break
        previous_block_exit = block_var_detail[index - 1]["forward"]["exit"]
        previous_block_exit.sort()
        current_block_entrance = block_var_detail[index]["forward"]["entrance"]

        backward_entrance = block_var_detail[index]["backward"]["entrance"]

        forward_all = block_var_detail[index]["forward"][
            "entrance"] + block_var_detail[index]["forward"][
                "private"] + block_var_detail[index]["forward"]["exit"]

        for var in backward_entrance:
            if not ("@GRAD" in var) and not (var in forward_all):
                current_block_entrance.append(var)

        current_block_entrance.sort()

        if previous_block_exit == current_block_entrance:
            continue
        exist_vars = list(
            set(previous_block_exit) & set(current_block_entrance))
        need_add_vars = list(set(current_block_entrance) - set(exist_vars))
        # var in different stage should not be ignored, since they are not placed in the same program & device
        #need_add_vars = find_need_var_from_previous_block(
        #    need_add_vars, block_var_detail, index, heter_ops)

        previous_block_private = block_var_detail[index - 1]["forward"][
            "private"]
        previous_block_entrance = block_var_detail[index - 1]["forward"][
            "entrance"]
        for var in need_add_vars:
            if var not in previous_block_private and var not in previous_block_entrance:
                previous_block_entrance.append(var)
            previous_block_exit.append(var)
            if not var in current_block_entrance:
                current_block_entrance.append(var)

    for index in range(0, len(block_var_detail) - 1, 1):
        previous_block_exit = block_var_detail[index + 1]["backward"]["exit"]
        previous_block_exit.sort()
        current_block_entrance = block_var_detail[index]["backward"]["entrance"]

        current_block_entrance.sort()

        if previous_block_exit == current_block_entrance:
            continue
        exist_vars = list(
            set(previous_block_exit) & set(current_block_entrance))
        need_add_vars = list(set(current_block_entrance) - set(exist_vars))
        need_ignore_vars = []
        for var in need_add_vars:
            if not "@GRAD" in var:
                need_ignore_vars.append(var)
        need_add_vars = list(
            set(need_add_vars).difference(set(need_ignore_vars)))
        previous_block_private = block_var_detail[index + 1]["backward"][
            "private"]
        previous_block_entrance = block_var_detail[index + 1]["backward"][
            "entrance"]
        for var in need_add_vars:
            if var not in previous_block_private and var not in previous_block_entrance:
                previous_block_entrance.append(var)
            previous_block_exit.append(var)
    return block_var_detail


def delete_block_useless_exit(program, program_block_ops_list,
                              block_var_detail):
    ## forward
    for index in range(len(block_var_detail)):
        if index == len(block_var_detail) - 1:
            break
        current_block_exit = block_var_detail[index]["forward"]["exit"]
        next_block_entrance = block_var_detail[index + 1]["forward"]["entrance"]
        need_delete_var = []
        for var in current_block_exit:
            if var not in next_block_entrance:
                need_delete_var.append(var)

        for var in need_delete_var:
            current_block_exit.remove(var)
    ## backward
    for index in range(len(block_var_detail) - 1, -1, -1):
        if index - 1 < 0:
            break
        current_block_exit = block_var_detail[index]["backward"]["exit"]
        next_block_entrance = block_var_detail[index - 1]["backward"][
            "entrance"]
        need_delete_var = []
        for var in current_block_exit:
            if var not in next_block_entrance:
                need_delete_var.append(var)
        for var in need_delete_var:
            current_block_exit.remove(var)

    return block_var_detail


def get_communicate_var_info(program,
                             block_index,
                             entrance_var_list,
                             type="forward"):
    input_var_reshape_dim = []
    input_var_reshape_name = []

    if type == "forward":
        block_input_var_name = "forward_joint_{}_{}@Heter".format(
            block_index - 1, block_index)
    else:
        block_input_var_name = "backward_joint_{}_{}@Heter".format(
            block_index + 1, block_index)

    entrance_var_list.sort()
    # input
    # Heter_SERVER_BLOCK_index@JOINT_VAR -> slice -> var@Heter_SERVER_BLOCK@INPUT_RESHAPE_VAR -> reshape -> var
    for name in entrance_var_list:
        var = program.global_block().vars[name]
        shape = var.shape
        recv_var_dim = -1 * reduce(lambda x, y: x * y, shape)
        input_var_reshape_dim.append(recv_var_dim)
        input_var_reshape_name.append("{}.input_reshape@Heter".format(name))

    info = {
        "input_var_reshape_dim": input_var_reshape_dim,
        "input_var_reshape_name": input_var_reshape_name,
        "block_input_var_name": block_input_var_name,
    }

    return info


def add_vars_by_var_list(var_name_list, origin_program, program, block):
    for var_name in var_name_list:
        if var_name not in program.global_block(
        ).vars and var_name not in block.vars:
            var = origin_program.global_block().vars[var_name]
            if var.persistable:
                program.global_block()._clone_variable(
                    var, force_persistable=False)
            else:
                block._clone_variable(var, force_persistable=False)


def _get_output_map_from_op(varmap, op):
    """Returns a dict from op output name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.output_names:
        vars = []
        for varname in op.output(key):
            if varname == "@EMPTY@":
                continue
            if "lod_tensor_blocking_queue" in varname:
                continue
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def block_append_op(program, origin_program, block, op):
    merge_ordereddict = origin_program.global_block().vars.copy()
    merge_ordereddict.update(block.vars)
    inputs = _get_input_map_from_op(merge_ordereddict, op)
    for key, varlist in six.iteritems(inputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if var.name not in program.global_block(
            ).vars and var.name not in block.vars:
                if var.persistable:
                    program.global_block()._clone_variable(
                        var, force_persistable=False)
                else:
                    block._clone_variable(var, force_persistable=False)

    outputs = _get_output_map_from_op(origin_program.global_block().vars, op)
    for key, varlist in six.iteritems(outputs):
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if var.name not in program.global_block(
            ).vars and var.name not in block.vars:
                if var.persistable:
                    program.global_block()._clone_variable(
                        var, force_persistable=False)
                else:
                    block._clone_variable(var, force_persistable=False)

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


def get_next_stage_trainers(role_maker):
    try:
        return role_maker._get_next_trainers()
    except Exception:
        return role_maker.get_next_trainers()


def insert_communicate_op(orign_program,
                          role_maker,
                          heter_block,
                          stage_id,
                          first_op_index,
                          block_var_detail,
                          device,
                          is_forward=True):

    if is_forward:
        next_heter_worker_endpoints = get_next_stage_trainers(role_maker)
        previous_heter_worker_endpoints = get_previous_stage_trainers(
            role_maker)
        entrance_var = block_var_detail[stage_id]["forward"]["entrance"]
        comm_info = get_communicate_var_info(orign_program, stage_id + 1,
                                             entrance_var)

    else:
        next_heter_worker_endpoints = get_next_stage_trainers(role_maker)
        previous_heter_worker_endpoints = get_previous_stage_trainers(
            role_maker)
        entrance_var = block_var_detail[stage_id - 1]["backward"]["exit"]
        comm_info = get_communicate_var_info(orign_program, stage_id - 1,
                                             entrance_var, "backward")

    heter_block._insert_op(
        index=first_op_index,
        type="send_and_recv",
        inputs={"X": heter_block.vars[entrance_var[0]]},
        outputs={"Out": []},
        attrs={
            "mode": "forward" if is_forward else "backward",
            "send_var_name": entrance_var + ["microbatch_id"],
            "recv_var_name": [],
            "message_name": comm_info["block_input_var_name"],
            "next_endpoints": next_heter_worker_endpoints,
            "previous_endpoints": previous_heter_worker_endpoints,
            "trainer_id": get_role_id(role_maker),
            "op_device": device,
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
        })

    return entrance_var


def get_the_one_recv_context(context,
                             is_dense=True,
                             split_dense_table=False,
                             use_origin_program=False):
    recv_id_maps = {}
    grad_name_to_param_name = {}
    if is_dense:
        send_ctx = get_the_one_send_context(
            context,
            split_dense_table=split_dense_table,
            use_origin_program=use_origin_program)
        for idx, (name, ctx) in enumerate(send_ctx.items()):
            if ctx.is_sparse():
                continue
            if ctx.is_tensor_table():
                continue

            origin_grad_varnames = ctx.origin_varnames()

            param_names = []
            for grad_varname in origin_grad_varnames:
                param_name = grad_name_to_param_name[grad_varname]
                param_names.append(param_name)
            recv_id_maps[ctx.table_id()] = param_names
    else:
        send_ctx = get_the_one_send_context(
            context,
            split_dense_table=False,
            use_origin_program=False,
            ep_list=None)
        for idx, (name, ctx) in enumerate(send_ctx.items()):
            if not ctx.is_sparse():
                continue

            origin_grad_varnames = ctx.origin_varnames()

            param_names = []
            for grad_varname in origin_grad_varnames:
                param_name = grad_name_to_param_name[grad_varname]
                param_names.append(param_name)
            recv_id_maps[ctx.table_id()] = param_names
    return recv_id_maps


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
