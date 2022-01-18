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
from paddle.fluid.incubate.fleet.parameter_server.mode import DistributedMode
from paddle.fluid.incubate.fleet.parameter_server.ir import vars_metatools
from paddle.fluid.incubate.fleet.parameter_server.ir.ps_dispatcher import RoundRobin, PSDispatcher
from paddle.fluid.transpiler.details.program_utils import delete_ops

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


def get_dist_env(self):
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


def get_distributed_strategy(dist_strategy):
    from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler.distributed_strategy import StrategyFactory

    k_steps = dist_strategy.a_sync_configs["k_steps"]
    strategy = None

    if not dist_strategy.a_sync and k_steps == 0:
        strategy = StrategyFactory.create_sync_strategy()

    if dist_strategy.a_sync and k_steps == 0:
        strategy = StrategyFactory.create_async_strategy()

    if dist_strategy.a_sync and k_steps > 0:
        strategy = StrategyFactory.create_geo_strategy(k_steps)

    if not strategy:
        raise ValueError("k_steps must be invalid value, please check")

    return strategy


def is_distributed_sparse_op(op):
    if op.type in SPARSE_OP_LIST and op.attr('is_distributed') is True:
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
