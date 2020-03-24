# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import six

from paddle.fluid import core

OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "@CLIP"
OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()


class DistributedMode:
    SYNC = 0
    ASYNC = 1
    HALF_ASYNC = 2
    GEO = 3


def _is_opt_role_op(op):
    # NOTE: depend on oprole to find out whether this op is for
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


def clone_variable(block, var, persistable=True):
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=persistable)


def get_param_grads(origin_program):
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
        sparse_varnames = []
        op_types = {"lookup_table": "W"}
        for op in origin_program.global_block().ops:
            if op.type in op_types.keys() \
                    and op.attr('remote_prefetch') is True:
                param_name = op.input(op.input_names[op_types[op.type]])[0]
                sparse_varnames.append(param_name)

        return list(set(sparse_varnames))

    sparse_varnames = _get_sparse_varnames()
    sparse_param_grads, dense_param_grads = _get_params_grads(sparse_varnames)

    return sparse_varnames, dense_param_grads
