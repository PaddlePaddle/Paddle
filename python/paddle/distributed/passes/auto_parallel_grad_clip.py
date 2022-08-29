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

import numpy as np
from functools import reduce

import paddle

from paddle.fluid import core
from .pass_base import PassBase, register_pass
from ..auto_parallel.reshard import Resharder
from ..auto_parallel.utils import is_gradient_clip_op, is_optimize_op, OP_ROLE_KEY, OpRole, _get_comm_group


def _get_params_grads(block):
    params_grads = []
    for op in reversed(block.ops):
        if not is_optimize_op(op):
            break
        if "Param" in op.input_names and "Grad" in op.input_names:
            param_name = op.input("Param")[0]
            grad_name = op.input("Grad")[0]
            param = block.var(param_name)
            grad = block.var(grad_name)
            params_grads.append((param, grad))
    return params_grads


def _get_dpmp_process_mesh(rank_id, topology, processes, sharding_group):
    if sharding_group is None:
        return topology, processes

    sharding_axis = 1
    dp_sharding_topology = [
        topology[0] // sharding_group.nranks, sharding_group.nranks
    ]
    if dp_sharding_topology[0] == 1:
        sharding_axis = 0
        dp_sharding_topology = dp_sharding_topology[1:]
    product_dp_sharding = reduce(lambda x, y: x * y, dp_sharding_topology)
    product_topology = reduce(lambda x, y: x * y, topology)

    if product_topology == product_dp_sharding:
        new_topology = dp_sharding_topology
    else:
        assert product_topology % product_dp_sharding == 0
        mp_degree = product_topology // product_dp_sharding
        new_topology = dp_sharding_topology + [mp_degree]

    sharding_groups = []
    for rank in processes:
        group = _get_comm_group(processes, new_topology, sharding_axis, rank)
        if group not in sharding_groups:
            sharding_groups.append(group)

    sharding_groups = np.array(sharding_groups)
    dpmp_processes_in_sharding = None
    for i in range(sharding_groups.shape[-1]):
        if rank_id in sharding_groups[:, i]:
            dpmp_processes_in_sharding = sharding_groups[:, i]

    assert dpmp_processes_in_sharding is not None
    return new_topology, list(dpmp_processes_in_sharding)


def _is_belong_to_global_norm(rank_id, tensor_shape, topology, processes,
                              dims_mapping, sharding_group):

    new_topology, new_processes = _get_dpmp_process_mesh(
        rank_id, topology, processes, sharding_group)

    complete_shape = Resharder.compute_complete_shape(tensor_shape,
                                                      new_topology,
                                                      dims_mapping)

    complete_partitions = []
    complete_param_ranks = []
    for process in new_processes:
        partition_index = Resharder.compute_partition_index(
            process, complete_shape, dims_mapping, new_topology, new_processes)
        if partition_index not in complete_partitions:
            complete_partitions.append(partition_index)
            complete_param_ranks.append(process)

    return rank_id in complete_param_ranks


class ClipHelper(object):

    def __init__(self, params_grads, rank_id, block, dist_context):
        params, _ = zip(*params_grads)
        self.params = list(params)
        self.params_name = [p.name for p in self.params]
        self.rank_id = rank_id
        self.block = block
        self.dist_context = dist_context
        self.sharding_group = None
        if hasattr(dist_context, '_sharding_group'):
            self.sharding_group = dist_context._sharding_group

    def _is_calcuate_norm(self, name):
        if not self._is_local_param(name):
            return False, []

        param = self.params[self.params_name.index(name)]
        dist_attr = self._get_dist_attr(name)
        topology = dist_attr.process_mesh.topology
        processes = dist_attr.process_mesh.processes
        dims_mapping = dist_attr.dims_mapping
        return _is_belong_to_global_norm(self.rank_id, param.shape, topology,
                                         processes, dims_mapping,
                                         self.sharding_group)

    def _get_dist_attr(self, name):
        var = self.block.vars[name]
        return self.dist_context.get_tensor_dist_attr_for_program(var)

    def _is_local_param(self, name):
        if name not in self.params_name:
            return False
        return True

    def _is_local_var(self, name):
        dist_attr = self._get_dist_attr(name)
        assert dist_attr is not None
        return self.rank_id in dist_attr.process_mesh.processes


@register_pass("auto_parallel_grad_clip")
class ClipGradByGloblNormPass(PassBase):

    def __init__(self):
        super(ClipGradByGloblNormPass, self).__init__()
        self.set_attr("rank_id", None)
        self.set_attr("dist_context", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        dist_context = self.get_attr("dist_context", None)
        rank_id = self.get_attr("rank_id", None)
        block = main_program.global_block()
        dist_params_grads = _get_params_grads(block)

        self.clip_helper = ClipHelper(dist_params_grads, rank_id, block,
                                      dist_context)
        self._remove_no_need_ops_vars(block)

    def _remove_no_need_ops_vars(self, block):

        removed_op_out_type = [
            'clip_by_norm', 'squared_l2_norm', 'square', 'reduce_sum'
        ]

        removed_op_idx = set()
        removed_tmp_var = set()
        for idx, op in enumerate(block.ops):
            if not is_gradient_clip_op(op):
                continue

            if op.type in removed_op_out_type:
                input_name = op.input("X")[0]
                if input_name.find("@GRAD") != -1:
                    #'clip_by_norm', 'squared_l2_norm', 'square'
                    param_name = input_name[:input_name.find("@GRAD")]
                    is_local = self.clip_helper._is_local_param(param_name)
                    is_calculate = self.clip_helper._is_calcuate_norm(
                        param_name)
                    if not is_local or (not is_calculate
                                        and op.type != 'clip_by_norm'):
                        removed_op_idx.add(idx)
                        removed_tmp_var.update(set(op.output_arg_names))
                else:
                    # 'reduce_sum'
                    if idx - 1 in removed_op_idx:
                        removed_op_idx.add(idx)
                        removed_tmp_var.update(set(op.output_arg_names))

            elif op.type == 'elementwise_mul':
                input_name = op.input("X")[0]
                if input_name.find("@GRAD") != -1:
                    param_name = input_name[:input_name.find("@GRAD")]
                    is_local = self.clip_helper._is_local_param(param_name)
                    if not is_local:
                        removed_op_idx.add(idx)
                        if block.ops[idx - 1].type == 'cast':
                            removed_op_idx.add(idx - 1)
                            removed_tmp_var.update(
                                set(block.ops[idx - 1].output_arg_names))

            elif op.type == 'sum':
                reserved_vars = []
                for input_name in op.input_arg_names:
                    if input_name not in removed_tmp_var and \
                        self.clip_helper._is_local_var(input_name):
                        reserved_vars.append(input_name)
                if not reserved_vars:
                    removed_op_idx.add(idx)
                    removed_tmp_var.update(set(op.output_arg_names))
                    if block.ops[idx + 1].type == 'cast':
                        removed_op_idx.add(idx + 1)
                        removed_tmp_var.update(
                            set(block.ops[idx + 1].output_arg_names))
                else:
                    op.desc.set_input("X", reserved_vars)

        for idx, op in reversed(list(enumerate(block.ops))):
            if not is_optimize_op(op):
                break
            if not is_gradient_clip_op(op):
                continue
            if idx in removed_op_idx:
                block._remove_op(idx, sync=False)

        for idx, op in reversed(list(enumerate(block.ops))):
            if not is_optimize_op(op):
                break
            if not is_gradient_clip_op(op):
                continue
            if op.type == 'sqrt':
                input_name = op.input("X")[0]
                input_var = block.vars[input_name]
                if paddle.distributed.get_world_size() > 1:
                    offset = 0
                    if input_name in removed_tmp_var:
                        removed_tmp_var.remove(input_name)
                        fill_constant_op = block._insert_op(
                            idx,
                            type='fill_constant',
                            inputs={},
                            outputs={'Out': [input_var]},
                            attrs={
                                'shape': [1],
                                'dtype': input_var.dtype,
                                'value': 0,
                                'force_cpu': False,
                                OP_ROLE_KEY: OpRole.Forward
                            })
                        fill_constant_op._set_attr('op_namescope',
                                                   "/gradient_clip_pass")
                        offset += 1

                    allreduce_op = block._insert_op(
                        idx + offset,
                        type='c_allreduce_sum',
                        inputs={'X': [input_var]},
                        outputs={'Out': [input_var]},
                        attrs={
                            'ring_id': 0,
                            'use_calc_stream': True,
                            OP_ROLE_KEY: OpRole.Optimize,
                        })
                    allreduce_op._set_attr('op_namescope',
                                           "/gradient_clip_pass")

        for varname in removed_tmp_var:
            block._remove_var(varname, sync=False)

        block._sync_with_cpp()
