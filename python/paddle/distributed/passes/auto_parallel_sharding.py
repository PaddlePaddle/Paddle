# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from functools import reduce
from collections import OrderedDict
import numpy as np

import paddle
from paddle.framework import core
from paddle.fluid import unique_name
from .pass_base import PassBase, register_pass
from paddle.distributed.fleet.meta_optimizers.common import is_backward_op, is_optimizer_op
from paddle.distributed.auto_parallel.process_group import new_process_group
from paddle.distributed.auto_parallel.operators.common import is_parameter_related, is_data_parallel_reduce_op
from paddle.distributed.auto_parallel.utils import _get_comm_group, naive_set_dist_op_attr_for_program_by_mesh_and_mapping, set_var_dist_attr

OpRole = core.op_proto_and_checker_maker.OpRole
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
_skip_ops = [
    'create_py_reader', 'create_double_buffer_reader', 'read', 'slice', 'split',
    'assign', "send_v2"
]
# update here to support new optimizers
_supported_optimizer_type = [
    "adam", "adamax", "adamw", "decayed_adagrad", "momentum", "dgc_momentum",
    "lars_momentum", "merged_momentum", "lamb", "sgd"
]


def _is_reshard_op(op):
    return op.desc.has_attr("op_namescope") and \
        "/auto_parallel/reshard" in op.desc.attr('op_namescope')


# NOTE we add the "auto_parallel" prefix to the pass in order to
# indicate that this pass should obey some constrains by auto_parallel
# for example all ops and vars should has dist attr before and after pass
# should use dist op instead of custom comm op
@register_pass("auto_parallel_sharding")
class ShardingPass(PassBase):

    def __init__(self):
        super(ShardingPass, self).__init__()
        self.set_attr("dist_context", None)
        self.set_attr("stage", None)
        self.set_attr("sharding_degree", None)  # for parallelizer
        self.set_attr("degree", None)  # for parallelizer_v2
        self.set_attr("params_grads", [])
        self.set_attr("global_rank", -1)
        self.dp_groups = set()
        self.sharding_infos = []
        self.varname_to_sharding_info = {}
        self.partial_sharding = False
        self.outer_dp_group = None
        self.shared_params_grads = []

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False

        if self.get_attr("stage") not in [1, 2, 3]:
            return False
        if self.get_attr("sharding_degree") is not None:
            if (not isinstance(self.get_attr("sharding_degree"), int)) \
                or self.get_attr("sharding_degree") <= 1:
                return False
        elif self.get_attr("degree") is not None:
            if (not isinstance(self.get_attr("degree"), int)) \
                or self.get_attr("degree") <= 1:
                return False
        else:
            return False
        if len(self.get_attr("params_grads")) <= 0:
            return False
        if (not isinstance(self.get_attr("global_rank"),
                           int)) or self.get_attr("global_rank") < 0:
            return False

        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self._dist_context = self.get_attr("dist_context")
        self.sharding_world_size = int(
            self.get_attr("sharding_degree") or self.get_attr("degree"))
        self.stage = int(self.get_attr("stage"))
        self.global_rank = int(self.get_attr("global_rank"))
        params_grads = self.get_attr("params_grads")
        main_block, startup_block = main_program.global_block(
        ), startup_program.global_block()

        self._build_sharding_groups(main_block, params_grads)
        self._shard_optimizer(main_block, startup_block, params_grads, context)
        self._shard_gradient_synchronization(main_block)
        self._shard_parameter(main_block, startup_block)

        context.set_attr("params_grads", self.shared_params_grads)

    def _build_sharding_groups(self, main_block, params_grads):
        self._collective_data_parallel_groups(main_block)
        self._build_sharding_infos(params_grads)

    def _collective_data_parallel_groups(self, main_block):
        for op in main_block.ops:
            if not _is_forward_op(op) or op.type in _skip_ops:
                continue
            # NOTE: there aren't dist_attr in the ops which reshard insert,
            # and should be skip in sharding.
            if _is_reshard_op(op):
                continue
            group = _inference_data_parallel_group_for_operator(
                self.global_rank, op, self._dist_context)
            if group is not None:
                self.dp_groups.add(group)

        # TODO(JZ-LIANG) allow more than one dp groups in network, support more general distribution
        # genetated by auto search
        if len(self.dp_groups) != 1:
            raise NotImplementedError(
                "So far Only and Exactly one data parallel group in network are supported, but got [{}] different data parallel groups"
                .format(len(self.dp_groups)))

    def _build_sharding_infos(self, params_grads):

        for dp_group in self.dp_groups:

            assert dp_group.nranks >= self.sharding_world_size, "sharding world size [{}] should not larger than dp world size [{}]".format(
                self.sharding_world_size, dp_group.nranks)
            assert dp_group.nranks % self.sharding_world_size == 0, "sharding world size [{}] should be divisible by dp world size [{}]".format(
                self.sharding_world_size, dp_group.nranks)
            assert self.global_rank in dp_group.ranks, "current ranks [{}] does NOT belong to the data parallel group [{}]".format(
                self.global_rank, dp_group.ranks)
            assert len(
                params_grads
            ) >= self.sharding_world_size, "number of parameters [{}] is not enough to be shard among [{}] ranks".format(
                len(params_grads), self.sharding_world_size)

            # sharding hybrid data parallel: partial sharding param within
            if dp_group.nranks > self.sharding_world_size:
                self.partial_sharding = True
                assert len(
                    self.dp_groups
                ) == 1, "hybrid sharding and data parallelism are supported only when there is excatly one data parallel group in the network"
                outer_dp_group, sharding_group = _get_dp_and_sharding_groups(
                    dp_group.ranks, self.sharding_world_size, self.global_rank)
                sharding_group = new_process_group(sharding_group)
                self.outer_dp_group = new_process_group(outer_dp_group)
            else:
                sharding_group = dp_group

            self._dist_context._sharding_group = sharding_group
            # TODO(JZ-LIANG) when support multiple dp groups in future, should group param and bind them to corresponding dp group
            sharding_info = ShardingInfo(sharding_group, self.global_rank,
                                         params_grads)
            self.sharding_infos.append(sharding_info)
            for param in sharding_info.params:
                self.varname_to_sharding_info[param.name] = sharding_info

    def _shard_optimizer(self, main_block, startup_block, params_grads,
                         pass_context):
        """
        sharding all optimizer related ops and vars, include:
        gradient clip ops & vars
        weight decay ops & vars
        optimizer ops and states
        """
        self._shard_amp_related_op_and_vars(main_block, pass_context)
        self._shard_weight_decay(main_block)
        # self._shard_gradient_clip(main_block)
        self._shard_optimizer_ops_and_states(main_block, startup_block)
        self._insert_optimizer_broadcasts(main_block, startup_block)

    def _shard_amp_related_op_and_vars(self, main_block, pass_context):

        if self.stage < 2:
            return

        for idx, op in reversed(list(enumerate(main_block.ops))):
            # shard amp related param_grad cast
            if _is_param_grad_fp32_cast_op(main_block, op):
                output_name = op.output_arg_names[0]
                param_name = output_name[:output_name.find("@")]
                if not self._is_parameter_in_local_shard(param_name):
                    main_block._remove_op(idx, sync=False)
                    main_block._remove_var(output_name, sync=False)

            # shard check nan inf
            elif op.type in ["check_finite_and_unscale", "update_loss_scaling"]:
                reversed_x = []
                for input_name in op.desc.input('X'):
                    param_name = input_name[:input_name.find("@")]

                    if self._is_parameter_in_local_shard(param_name):
                        reversed_x.append(input_name)

                # NOTE: When `reversed_x` is [], check_finite_and_unscale will be replaced by `fill_constant` op.
                # The output of check_finite_and_unscale is be set False
                if reversed_x:
                    op.desc.set_input('X', reversed_x)
                    op.desc.set_output('Out', reversed_x)
                else:
                    if op.type == "check_finite_and_unscale":
                        op_role = op.attr('op_role')
                        out_name = op.output_arg_names[0]
                        out_var = main_block.vars[out_name]
                        main_block._remove_op(idx, sync=False)
                        main_block._insert_op_without_sync(
                            idx,
                            type="fill_constant",
                            outputs={"Out": out_var},
                            attrs={
                                "shape": out_var.shape,
                                "dtype": out_var.dtype,
                                "value": 0,
                                OP_ROLE_KEY: op_role,
                            })
                    else:
                        main_block._remove_op(idx, sync=False)

        main_block._sync_with_cpp()

    def _shard_gradient_clip(self, main_block):

        if self.stage < 2:
            return

        # TODO (JZ-LIANG) support calculate global norm with tensor parallelism
        removed_op_type = ['elementwise_mul', 'squared_l2_norm', 'clip_by_norm']
        removed_op_idx = set()
        removed_tmp_var = set()

        for idx, op in list(enumerate(main_block.ops)):
            if not _is_gradient_clip_op(op):
                continue

            if op.type in removed_op_type:
                input_name = op.input("X")[0]
                param_name = input_name[:input_name.find("@GRAD")]
                if not self._is_parameter_in_local_shard(param_name):
                    removed_op_idx.add(idx)
                    if op.type in ['squared_l2_norm', 'clip_by_norm']:
                        for output_name in op.output_arg_names:
                            removed_tmp_var.add(output_name)

        for idx, op in reversed(list(enumerate(main_block.ops))):
            if not _is_gradient_clip_op(op):
                continue
            if idx in removed_op_idx:
                main_block._remove_op(idx, sync=False)

        for varname in removed_tmp_var:
            main_block._remove_var(varname, sync=False)

        for idx, op in list(enumerate(main_block.ops)):
            if not _is_gradient_clip_op(op):
                continue
            if op.type == 'sum':
                reserved_vars = []
                for input_name in op.input_arg_names:
                    if input_name not in removed_tmp_var:
                        reserved_vars.append(input_name)
                op.desc.set_input("X", reserved_vars)

                sum_op_output = op.desc.output_arg_names()[0]
                for i, sharding_info in enumerate(self.sharding_infos):
                    new_op = main_block._insert_op(
                        idx + i + 1,
                        type='c_allreduce_sum',
                        inputs={'X': [sum_op_output]},
                        outputs={'Out': [sum_op_output]},
                        attrs={
                            'ring_id': sharding_info.group.id,
                            'op_namescope': "/gradient_clip_model_parallelism",
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Optimize,
                        })
                    dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
                        main_block.var(sum_op_output))
                    # assert dist_attr is not None
                    # naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    #     new_op, dist_attr.process_mesh, dist_attr.dims_mapping,
                    #     self._dist_context)
                break

        main_block._sync_with_cpp()

    def _shard_weight_decay(self, main_block):

        if self.stage < 2:
            return

        for idx, op in reversed(list(enumerate(main_block.ops))):
            if not _is_weight_decay_op(op):
                continue
            else:
                raise NotImplementedError(
                    "weight decay is NOT supported by now")
        main_block._sync_with_cpp()

    def _shard_optimizer_ops_and_states(self, main_block, startup_block):

        should_removed_optimizer_states = []
        for idx, op in reversed(list(enumerate(main_block.ops))):
            if not is_optimizer_op(op):
                break

            if op.type in _supported_optimizer_type:
                assert "Param" in op.input_names
                assert len(op.input("Param")) == 1
                param_name = op.input("Param")[0]
                if not self._is_parameter_in_local_shard(param_name):
                    should_removed_optimizer_states.extend([
                        varname for varname in op.output_arg_names
                        if varname != param_name
                    ])
                    main_block._remove_op(idx, sync=False)
                else:
                    self.shared_params_grads.append(
                        self._get_param_grad(param_name))

        for idx, op in reversed(list(enumerate(startup_block.ops))):
            if len(op.output_arg_names) == 1 and op.output_arg_names[
                    0] in should_removed_optimizer_states:
                startup_block._remove_op(idx, sync=False)

        for varname in should_removed_optimizer_states:
            if main_block.has_var(varname):
                main_block._remove_var(varname, sync=False)
            if startup_block.has_var(varname):
                startup_block._remove_var(varname, sync=False)

        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()

    def _insert_optimizer_broadcasts(self, main_block, startup_block):

        if self.stage > 2:
            return

        for sharding_info in self.sharding_infos:
            for param in sharding_info.params:
                assert main_block.has_var(param.name)
                assert startup_block.has_var(param.name)

                new_op = main_block.append_op(type='c_broadcast',
                                              inputs={'X': param},
                                              outputs={'Out': param},
                                              attrs={
                                                  'ring_id':
                                                  sharding_info.group.id,
                                                  'root':
                                                  sharding_info.get_var_rank(
                                                      param.name),
                                                  'use_calc_stream':
                                                  True,
                                                  OP_ROLE_KEY:
                                                  OpRole.Optimize
                                              })
                param_dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
                    param)
                assert param_dist_attr is not None
                naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
                    new_op, param_dist_attr.process_mesh,
                    param_dist_attr.dims_mapping, self._dist_context)
        main_block._sync_with_cpp()

    def _is_parameter_in_local_shard(self, param_name):
        assert param_name in self.varname_to_sharding_info
        sharding_info = self.varname_to_sharding_info[param_name]
        return sharding_info.is_in_local_shard(param_name)

    def _get_param_grad(self, param_name):
        assert param_name in self.varname_to_sharding_info
        sharding_info = self.varname_to_sharding_info[param_name]
        p_g = sharding_info.get_param_grad(param_name)
        assert p_g is not None
        return p_g

    def _shard_gradient_synchronization(self, main_block):

        if self.stage < 2:
            return

        dp_ring_ids = [group.id for group in self.dp_groups]
        for idx, op in reversed(list(enumerate(main_block.ops))):
            if is_data_parallel_reduce_op(op):
                input_name = op.input_arg_names[0]
                base_name = _get_base_name_from_grad_name(input_name)
                sharding_info = self.varname_to_sharding_info[base_name]
                _insert_reduce_op(main_block, idx, input_name,
                                  sharding_info.group.id,
                                  sharding_info.get_var_rank(base_name),
                                  self._dist_context)
                if not self.partial_sharding or not sharding_info.is_in_local_shard(
                        base_name):
                    main_block._remove_op(idx + 1, sync=False)
                else:
                    op._set_attr("ring_id", self.outer_dp_group.id)

            # NOTE:
            # var@GRAD = sum(var@GRAD@RENAME@0, var@GRAD@RENAME@1)
            # If the var is not in local rank and it is output of many ops, or the var is renamed in another words,
            # the sum op should be removed.
            if _is_param_grad_sum_op(op, main_block):
                out_name = op.output_arg_names[0]
                base_name = _get_base_name_from_grad_name(out_name)
                sharding_info = self.varname_to_sharding_info[base_name]
                if not sharding_info.is_in_local_shard(base_name):
                    main_block._remove_op(idx, sync=False)

        main_block._sync_with_cpp()

    def _shard_parameter(self, main_block, startup_block):

        if self.stage < 3:
            return

        dp_ring_ids = [group.id for group in self.dp_groups]
        for sharding_info in self.sharding_infos:
            need_broadcast_vars, param_usage = sharding_info.get_broadcast_vars_and_param_usage(
                main_block)
            not_used_param_nane = []
            for param_name in param_usage:
                if param_usage[param_name] == 0 and sharding_info.get_var_rank(
                        param_name) != sharding_info.local_rank:
                    not_used_param_nane.append(param_name)

            for idx, op in reversed(list(enumerate(main_block.ops))):
                if is_optimizer_op(op):
                    continue

                for input_name in op.desc.input_arg_names():
                    # NOTE hack for embedding op when AMP 02-3
                    # paddle amp force embedding (lookup table) to be run on fp32
                    if _is_param_fp16_cast_op(main_block, op,
                                              sharding_info.param_names):
                        continue
                    if input_name not in need_broadcast_vars:
                        continue
                    root_rank = sharding_info.get_var_rank(input_name)
                    if root_rank == sharding_info.local_rank:
                        broadcast_varname = input_name
                    else:
                        broadcast_varname = unique_name.generate(input_name +
                                                                 "@BroadCast")
                        input_var = main_block.var(input_name)
                        new_var = main_block.create_var(name=broadcast_varname,
                                                        shape=input_var.shape,
                                                        dtype=input_var.dtype,
                                                        persistable=False)
                        ref_dist_attr = self._dist_context.get_tensor_dist_attr_for_program(
                            input_var)
                        out_var_dist_attr = set_var_dist_attr(
                            self._dist_context, new_var,
                            ref_dist_attr.dims_mapping,
                            ref_dist_attr.process_mesh)
                        op._rename_input(input_name, broadcast_varname)

                    _insert_init_and_broadcast_op(main_block, idx,
                                                  broadcast_varname,
                                                  sharding_info.local_rank,
                                                  root_rank,
                                                  sharding_info.group.id,
                                                  op.attr('op_role'),
                                                  self._dist_context)

            for idx, op in reversed(list(enumerate(main_block.ops))):
                if op.type != "cast":
                    continue
                input_name = op.input_arg_names[0]
                output_name = op.output_arg_names[0]
                if input_name in not_used_param_nane:
                    main_block._remove_op(idx, sync=False)
                    main_block._remove_var(output_name, sync=False)

            for idx, op in reversed(list(enumerate(startup_block.ops))):
                assert len(op.output_arg_names) == 1
                output_name = op.output_arg_names[0]

                if op.type == "c_broadcast" and op.attr(
                        "ring_id") in dp_ring_ids:
                    if self.outer_dp_group and sharding_info.get_var_rank(
                            output_name) == sharding_info.local_rank:
                        op._set_attr("ring_id", self.outer_dp_group.id)
                    else:
                        startup_block._remove_op(idx, sync=False)
                    continue

                if op.type != "c_broadcast" and output_name in param_usage and sharding_info.get_var_rank(
                        output_name) != sharding_info.local_rank:
                    startup_block._remove_op(idx, sync=False)

            for param_name in param_usage:
                if sharding_info.get_var_rank(
                        param_name) != sharding_info.local_rank:
                    main_block._remove_var(param_name, sync=False)
                    startup_block._remove_var(param_name, sync=False)

        main_block._sync_with_cpp()
        startup_block._sync_with_cpp()


def _insert_init_and_broadcast_op(block, insert_idx, varname, local_rank,
                                  root_rank, ring_id, op_role, dist_context):
    """
    empty op for initialization
    """
    broadcast_var = block.var(varname)
    broadcast_var_dist_attr = dist_context.get_tensor_dist_attr_for_program(
        broadcast_var)

    new_op = block._insert_op_without_sync(insert_idx,
                                           type='c_broadcast',
                                           inputs={'X': varname},
                                           outputs={'Out': varname},
                                           attrs={
                                               'ring_id': ring_id,
                                               'root': root_rank,
                                               'use_calc_stream': True,
                                               OP_ROLE_KEY: op_role
                                           })
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
        new_op, broadcast_var_dist_attr.process_mesh,
        broadcast_var_dist_attr.dims_mapping, dist_context)
    if local_rank != root_rank:

        new_op = block._insert_op_without_sync(
            insert_idx,
            type="empty",
            outputs={"Out": broadcast_var.name},
            attrs={
                "shape": broadcast_var.shape,
                "dtype": broadcast_var.dtype,
                OP_ROLE_KEY: op_role
            })
        naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
            new_op, broadcast_var_dist_attr.process_mesh,
            broadcast_var_dist_attr.dims_mapping, dist_context)
    return


def _insert_reduce_op(block,
                      insert_idx,
                      reduce_var,
                      ring_id,
                      root_id,
                      dist_context,
                      op_role=OpRole.Backward,
                      use_calc_stream=True):
    assert root_id >= 0, "root id should be a positive int, but now root id is {}".format(
        root_id)
    new_op = block._insert_op_without_sync(insert_idx,
                                           type='c_reduce_sum',
                                           inputs={'X': [reduce_var]},
                                           outputs={'Out': [reduce_var]},
                                           attrs={
                                               'ring_id': ring_id,
                                               'root_id': root_id,
                                               'use_calc_stream':
                                               use_calc_stream,
                                               OP_ROLE_KEY: op_role
                                           })

    dist_attr = dist_context.get_tensor_dist_attr_for_program(
        block.var(reduce_var))
    naive_set_dist_op_attr_for_program_by_mesh_and_mapping(
        new_op, dist_attr.process_mesh, dist_attr.dims_mapping, dist_context)


def _get_dp_and_sharding_groups(origin_group, sharding_group_size, rank):
    dp_axis = 0
    sharding_axis = 1
    shape = [len(origin_group) // sharding_group_size, sharding_group_size]

    dp_group = _get_comm_group(origin_group, shape, dp_axis, rank)
    sharding_group = _get_comm_group(origin_group, shape, sharding_axis, rank)

    return dp_group, sharding_group


def _is_gradient_clip_op(op):
    return op.desc.has_attr("op_namescope") \
        and op.desc.attr("op_namescope").startswith("/gradient_clip")


def _is_weight_decay_op(op):
    return op.desc.has_attr("op_namescope") \
        and op.desc.attr("op_namescope").startswith("/regularization")


def _is_param_grad_fp32_cast_op(block, op):
    if not is_backward_op(op):
        return False
    if not _is_desired_cast_op(block, op, core.VarDesc.VarType.FP16,
                               core.VarDesc.VarType.FP32):
        return False
    output_name = op.desc.output_arg_names()[0]
    base_name = output_name[:output_name.find("@")]
    if not block.has_var(base_name):
        return False
    return block.var(base_name).is_parameter


def _is_param_fp16_cast_op(block, op, params):

    if is_optimizer_op(op):
        return False
    if not _is_desired_cast_op(block, op):
        return False
    input_name = op.desc.input_arg_names()[0]
    if input_name not in params:
        return False
    return True


def _is_desired_cast_op(block,
                        op,
                        src_var_type=core.VarDesc.VarType.FP32,
                        dst_var_type=core.VarDesc.VarType.FP16):
    if op.type != "cast":
        return False
    assert (len(op.desc.input_arg_names()) == 1)
    assert (len(op.desc.output_arg_names()) == 1)
    input_var = block.var(op.desc.input_arg_names()[0])
    output_var = block.var(op.desc.output_arg_names()[0])

    if input_var.dtype != src_var_type or \
        output_var.dtype != dst_var_type:
        return False

    return True


def _get_base_name_from_grad_name(grad_name):
    base_name = None
    if ".cast_fp16@GRAD" in grad_name:
        base_name = grad_name[:grad_name.find(".cast_fp16@GRAD")]
    elif "@GRAD" in grad_name:
        base_name = grad_name[:grad_name.find("@GRAD")]
    return base_name


def _is_param_grad_sum_op(op, block):

    if not is_backward_op(op):
        return False
    if op.type != "sum":
        return False

    output_name = op.output_arg_names[0]
    base_name = _get_base_name_from_grad_name(output_name)

    if not block.has_var(base_name):
        return False

    return block.var(base_name).is_parameter


def _is_forward_op(op):
    return op.attr("op_role") == 0


def _inference_data_parallel_group_for_operator(rank_id, op, dist_context):

    dp_group = None
    for input_name in op.input_arg_names:
        if not is_parameter_related(input_name, op.block):
            dist_attr = dist_context.get_op_dist_attr_for_program(op)
            process_mesh = dist_attr.process_mesh
            input_dim_mapping = dist_attr.get_input_dims_mapping(input_name)
            mesh_shape = process_mesh.topology
            # TODO(JZ-LIANG) replace with specific batch size dimension
            batch_size_axis = input_dim_mapping[0]
            if batch_size_axis > -1 and mesh_shape[batch_size_axis] > 1:
                group_ranks = _get_comm_group(process_mesh.processes,
                                              process_mesh.topology,
                                              batch_size_axis, rank_id)
                dp_group = new_process_group(group_ranks)
                break

    return dp_group


def shard_parameters(params, group_size):
    # TODO(JZ-LIANG) support multiple partition methods
    # method1: greedy even but unorder
    # method2: roughly even with oreder
    mapping = {}
    for rank_ in range(group_size):
        mapping[rank_] = []
    sizes = [0] * group_size
    for param in params:
        rank = sizes.index(min(sizes))
        mapping[rank].append(param)
        numel = reduce(lambda x, y: x * y, param.shape)
        assert numel > 0, "param [{}] should larger than 0, but it is [{}]".format(
            param.name, numel)
        sizes[rank] += numel

    return mapping


class ShardingInfo(object):

    def __init__(self, group, rank, params_grads):
        self.group = group
        self.params_grads = dict([(p.name, (p, g)) for p, g in params_grads])
        assert len(self.params_grads) == len(set(
            self.params_grads)), "found duplicated param in params_grads"

        self.params = [p for p, _ in params_grads]
        self.param_names = [p.name for p in self.params]
        self.group_size = group.nranks
        self.global_rank = rank
        self.local_rank = group.ranks.index(self.global_rank)
        # rank in below mapping are local rank in this sharding group
        self.rank_to_params = shard_parameters(self.params, self.group_size)
        # include fp32 and fp16 param
        self.param_to_rank = dict()
        self._map_param_to_rank()

    def _map_param_to_rank(self):
        """
        mapping parameters to the rank which holds it.
        """
        for rank, params in self.rank_to_params.items():
            for param in params:
                self.param_to_rank[param.name] = rank

    def get_var_rank(self, varname):
        if varname in self.param_to_rank:
            return self.param_to_rank[varname]
        return -1

    # determine fp32 and fp16 (cast) param
    def is_in_local_shard(self, param_name):
        return self.get_var_rank(param_name) == self.local_rank

    # NOTE the follwo logic is designed for supporting AMP O1 when
    # the param would be cast to fp16 before used for caculation.
    # and sharding should only broadcast the casted fp16 param
    # instead of the origin fp32 version param.
    def get_broadcast_vars_and_param_usage(self, block):
        broadcast_vars = set([])
        fp16_params = set([])
        fp16_to_fp32 = {}

        param_usage = {x: 0 for x in self.param_names}
        for op in block.ops:
            if is_optimizer_op(op):
                continue
            for input_name in op.desc.input_arg_names():
                if input_name in self.param_names:
                    param_usage[input_name] += 1

        for op in block.ops:
            if not _is_param_fp16_cast_op(block, op, self.param_names):
                continue
            input_name = op.input_arg_names[0]
            output_name = op.output_arg_names[0]
            broadcast_vars.add(output_name)
            fp16_params.add(output_name)
            fp16_to_fp32[output_name] = input_name
            param_usage[input_name] -= 1
            self.param_to_rank[output_name] = self.param_to_rank[input_name]

        for param, usage in param_usage.items():
            if usage > 0:
                broadcast_vars.add(param)
        return broadcast_vars, param_usage

    def get_param_grad(self, param_name):
        if not self.is_in_local_shard(param_name):
            raise ValueError(
                "param[{}] not in current rank.".format(param_name))
        if param_name not in self.params_grads:
            raise ValueError('param[{}] not in params_grads'.format(param_name))
        return self.params_grads.get(param_name, None)
