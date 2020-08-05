# Copyright(c) 2020 PaddlePaddle Authors.All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
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
from paddle.fluid.incubate.fleet.parameter_server.ir import ops_metatools
from paddle.fluid.incubate.fleet.parameter_server.ir.ps_dispatcher import RoundRobin, PSDispatcher
from paddle.fluid.transpiler.details.program_utils import delete_ops


DEVICE_LIST = ["cpu", "gpu", "xpu"]
COMMUNICATE_OPS_TYPE = ["send", "recv", "fetch_barrier", "send_barrier"]


def find_heter_ops(program, default_device="cpu"):
    if default_device not in DEVICE_LIST:
        raise ValueError("Given device {} is not in device list {}".format(
            default_device, DEVICE_LIST))

    def _is_heter_op(op, current_heter_device, default_device="cpu"):
        heter_devices = list(DEVICE_LIST)
        heter_devices.remove(default_device)
        op_device = op.attr("op_device")
        if op_device in heter_devices:
            return True
        elif op_device == "" or op_device == default_device:
            op._set_attr('op_device', default_device)
            return False
        return False

    def _is_same_device(op, pre_device, default_device="cpu"):
        op_device = op.attr("op_device")
        if op_device == pre_device:
            return True
        if pre_device == default_device:
            # for cpu-op , xpu-op
            return True
        return False

    def _record_op(op, current_heter_block_ops, heter_ops=None):
        op_device = op.attr("op_device")
        if op_device != "" and op_device != "cpu":
            if op_device not in heter_ops:
                heter_ops[op_device] = {}
        op_struct = ops_metatools.create_op_struct(op)
        current_heter_block_ops.append(op_struct)

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

            # for cpu-op block end
            if len(current_default_block_ops) > 1:
                _record_op(op, current_default_block_ops)
                default_ops[default_device][block_index] = current_default_block_ops
                program_block_ops.append(current_default_block_ops)
                current_default_block_ops = []
                block_index += 1

            if _is_same_device(op, current_heter_device, default_device):
                # for gpu-op, gpu-op -> gpu-op,...
                current_heter_device = op.attr("op_device")
                _record_op(op, current_heter_block_ops, heter_ops)

            else:
                # for gpu-op -> xpu-op, ...
                op_device = current_heter_block_ops[0].attr("op_device")
                heter_ops[op_device][block_index] = current_heter_block_ops
                program_block_ops.append(current_heter_block_ops)
                block_index += 1
                current_heter_block_ops = []
                current_heter_device = op.attr("op_device")
                _record_op(op, current_heter_block_ops, heter_ops)

        elif is_heter:
            # for gpu/xpu-op -> cpu-op
            op_device = current_heter_block_ops[0].attr("op_device")
            heter_ops[op_device][block_index] = current_heter_block_ops
            program_block_ops.append(current_heter_block_ops)
            block_index += 1
            current_heter_block_ops = []
            current_heter_device = default_device
            is_heter = False
            _record_op(op, current_default_block_ops)
        else:
            # for cpu-op
             _record_op(op, current_default_block_ops)

    if current_heter_block_ops != []:
        op_device = current_heter_block_ops[0].attr("op_device")
        heter_ops[op_device][block_index] = current_heter_block_ops
        program_block_ops.append(current_heter_block_ops)

    if current_default_block_ops != []:
        default_ops[default_device][block_index] = current_default_block_ops
        program_block_ops.append(current_default_block_ops)

    #remove_communicate_op(origin_porgram, heter_ops, default_ops, program_block_ops)

    if len(heter_ops) == 0:
        warnings.warn(
            "No heterogeneous OP was found in your program , "
            " please using fluid.device_guard() to run OPs on different device.")

    total_heter_ops = 0
    heter_blocks=0 
    for device in heter_ops.keys():
        heter_block_dict = heter_ops[device]
        heter_blocks += len(heter_block_dict)
        for _,heter_block in heter_block_dict.items():
            total_heter_ops += len(heter_block)
    print(
        "There are {} OPs in your main_program, and contains {} heter-OPs which is made up of {} heter-blocks.".format(len(block.ops), total_heter_ops, heter_blocks))
    return origin_porgram, heter_ops, default_ops, program_block_ops

def remove_communicate_op(program, heter_ops, default_ops, program_block_ops):
    send_op_list = get_send_op_list(program)
    send_op_input = []
    for send_op in send_op_list:
        # fail when send op has multi input var
        inputs = _get_input_map_from_op(program.global_block().vars, send_op)
        send_op_input += get_varlist_from_op_map(inputs)

    need_delete_op = []
    for device in heter_ops:
        for index, op_list in heter_ops[device].items():
            _, block_output = find_ops_list_input_output(program, op_list)
            send_var = list(set(send_op_input) & set(block_output))
            if send_var != None:
                for var in send_var:
                    send_op_index = list(send_op_input).index(var)
                    # 1. append send op in heter_block
                    heter_ops[device][index].append(send_op_list[send_op_index])
                    program_block_ops[index].append(send_op_list[send_op_index])
                    # 2. delete send op in origin block 
                    need_delete_op.append(send_op_list[send_op_index])

    for device in default_ops:
        for index, op_list in default_ops[device].items():
            block_input, _ = find_ops_list_input_output(program, op_list)
            send_var = list(set(send_op_input) & set(block_input))
            if send_var != None:
                for var in send_var:
                    # 3. delete send op in ops_list
                    send_op_index = list(send_op_input).index(var)
                    delete_same_ops_in_list(default_ops[device][index], [send_op_list[send_op_index]])
                    delete_same_ops_in_list(program_block_ops[index], [send_op_list[send_op_index]])

    delete_ops(program.global_block(), need_delete_op)
    program.global_block()._sync_with_cpp()

def get_send_op_list(program):
    send_op_list = []
    block = program.global_block()
    for op in block.ops:
        if op.type == "send":
            send_op_list.append(op)
    return send_op_list