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

import os

from paddle.fluid import core
from paddle.fluid.incubate.fleet.parameter.ir import vars_distributed

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


class ServerRuntimeConfig(object):
    def __init__(self):
        self._rpc_send_thread_num = int(
            os.getenv("FLAGS_rpc_send_thread_num", "12"))
        self._rpc_get_thread_num = int(
            os.getenv("FLAGS_rpc_get_thread_num", "12"))
        self._rpc_prefetch_thread_num = int(
            os.getenv("FLAGS_rpc_prefetch_thread_num", "12"))


class MergedVariable:
    def __init__(self, merged, ordered, offsets):
        self.merged_var = merged
        self.ordered_vars = ordered
        self.offsets = offsets


class CompileTimeStrategy(object):
    def __init__(self, main_program, startup_program, strategy, role_maker):
        self.origin_main_program = main_program
        self.origin_startup_program = startup_program

        self.strategy = strategy
        self.role_maker = role_maker

        self.origin_sparse_pairs = []
        self.origin_dense_pairs = []

        self.merged_denses = []

        self.param_grad_map = {}
        self.grad_param_map = {}

        self._build_var_distributed()

    def get_distributed_mode(self):
        return self.strategy.distributed_mode

    def get_role_id(self):
        return self.role_maker.role_id()

    def get_trainers(self):
        return self.role_maker.worker_num()

    def get_ps_endpoint(self):
        return self.role_maker.get_pserver_endpoints()[self.get_role_id()]

    def get_ps_endpoints(self):
        return self.role_maker.get_pserver_endpoints()

    def get_origin_programs(self):
        return self.origin_main_program, self.origin_startup_program

    def get_origin_main_program(self):
        return self.origin_main_program

    def get_origin_startup_program(self):
        return self.origin_startup_program

    def get_communicator_context(self):
        pass

    def get_server_runtime_config(self):
        return self.strategy.get_server_runtime_config()

    def _build_var_distributed(self):
        sparse_pairs, dense_pairs = self.get_param_grads()

        origin_for_sparse = []
        origin_for_dense = []
        param_grad_map = dict()
        grad_param_map = dict()

        for param, grad in sparse_pairs:
            param = vars_distributed.create_var_struct(param)
            grad = vars_distributed.create_var_struct(grad)
            origin_for_sparse.append((param, grad))

        for param, grad in dense_pairs:
            param = vars_distributed.create_var_struct(param)
            grad = vars_distributed.create_var_struct(grad)
            origin_for_dense.append((param, grad))

        ordered_dense, ordered_dense_offsets, merged_param, merged_grad = self.dense_var_merge(
            origin_for_dense)

        param = MergedVariable(merged_param, ordered_dense_offsets,
                               ordered_dense)
        grad = MergedVariable(merged_grad, ordered_dense_offsets, ordered_dense)

        self.merged_denses.append((param, grad))

        param_merges = []
        param_merges.extend(origin_for_sparse)
        param_merges.append((merged_param, merged_grad))

        for param, grad in param_merges:
            param_grad_map[param.name] = grad.name
            grad_param_map[grad.name] = param.name

        self.origin_sparse_pairs = origin_for_sparse
        self.origin_dense_pairs = origin_for_dense
        self.param_grad_map = param_grad_map
        self.grad_param_map = grad_param_map

    def dense_var_merge(self, denses):
        if not denses:
            return [], [], None

        ordered_dense = []
        ordered_dense_offsets = []
        flatten_dims = 0

        for dense in denses:
            param, grad = dense

            if grad.type == core.VarDesc.VarType.SELECTED_ROWS:
                raise TypeError(
                    "{} may not Dense Param, need check `build_var_distributed`".
                    format(param.name))

            ordered_dense.append(dense)
            ordered_dense_offsets.append(flatten_dims)
            flatten_dims += reduce(lambda x, y: x * y, param.shape)

        merged_param = vars_distributed.VarStruct(
            "merged.dense_0", (flatten_dims, ), denses[0].dtype, denses[0].type,
            denses[0].lod_level, denses[0].persistable)
        merged_grad = vars_distributed.VarStruct(
            "merged.dense_0@GRAD", (flatten_dims, ), denses[0].dtype,
            denses[0].type, denses[0].lod_level, denses[0].persistable)

        return ordered_dense, ordered_dense_offsets, merged_param, merged_grad

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
