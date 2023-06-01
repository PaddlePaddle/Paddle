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

from ..process_group import new_process_group
from ..utils import (
    _get_comm_group,
    _get_corresponding_rank,
    compute_compatible_and_update_dim_mapping,
    is_dim_replicate,
    is_dim_shard,
)
from .common import (
    DistributedOperatorImpl,
    DistributedOperatorImplContainer,
    register_distributed_operator_impl,
    register_distributed_operator_impl_container,
)
from .dist_default import DistributedDefaultImpl0


class DistributedFusedFeedForward(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super().__init__(op_type)


register_distributed_operator_impl_container(
    DistributedFusedFeedForward("fused_feedforward")
)


class DistributedFusedFeedForwardImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super().__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        linear1_weight = op_desc.input('Linear1Weight')[0]
        linear1_bias = op_desc.input('Linear1Bias')[0]
        linear2_weight = op_desc.input('Linear2Weight')[0]
        linear2_bias = op_desc.input('Linear2Bias')[0]

        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        linear1_weight_dims_mapping = op_dist_attr.get_input_dims_mapping(
            linear1_weight
        )
        linear1_bias_dims_mapping = op_dist_attr.get_input_dims_mapping(
            linear1_bias
        )
        linear2_weight_dims_mapping = op_dist_attr.get_input_dims_mapping(
            linear2_weight
        )
        linear2_bias_dims_mapping = op_dist_attr.get_input_dims_mapping(
            linear2_bias
        )

        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        if is_dim_shard(linear1_weight_dims_mapping[-2]) or is_dim_replicate(
            linear1_weight_dims_mapping[-1]
        ):
            return False
        if is_dim_replicate(linear1_bias_dims_mapping[-1]):
            return False
        if is_dim_replicate(linear2_weight_dims_mapping[-2]) or is_dim_shard(
            linear2_weight_dims_mapping[-1]
        ):
            return False
        if is_dim_shard(linear2_bias_dims_mapping[-1]):
            return False
        if linear1_weight_dims_mapping[-1] != linear2_weight_dims_mapping[-2]:
            return False

        return True

    def is_output_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr

        # none of output should be sharded
        for out_name in op_desc.output_names():
            out = op_desc.output(out_name)[0]
            out_dims_mapping = op_dist_attr.get_output_dims_mapping(out)
            for mapping in out_dims_mapping[1:-1]:
                if is_dim_shard(mapping):
                    return False
        return True

    def is_auto_compatible(self, dist_op):
        if (not self.is_input_compatible(dist_op)) or (
            not self.is_output_compatible(dist_op)
        ):
            return False

        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_names = op_desc.output('Out')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        for out_name in out_names:
            out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
            if x_dims_mapping != out_dims_mapping:
                return False

        return True

    def update_dims_mapping(self, dist_op):
        changed = False
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_names = op_desc.output('Out')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)

        for out_name in out_names:
            out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
            for i in range(len(x_dims_mapping)):
                dim_changed = compute_compatible_and_update_dim_mapping(
                    [x_dims_mapping, out_dims_mapping], [i, i]
                )
                if dim_changed:
                    changed = True
                    op_dist_attr.set_output_dims_mapping(
                        out_name, out_dims_mapping
                    )

        if changed:
            op_dist_attr.set_input_dims_mapping(x_name, x_dims_mapping)

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)

        if rank_id not in op_dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )

        # infer logic comm presentation
        linear1_weight = src_op.input('Linear1Weight')[0]
        linear1_weight_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
            linear1_weight
        )[-1]
        assert (
            linear1_weight_col_dim_mapping >= 0
        ), "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            linear1_weight_col_dim_mapping
        )
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        parallel_axis = linear1_weight_col_dim_mapping
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
        group = new_process_group(group_ranks)

        # insert op
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

        # setting comm id
        new_op = main_block.ops[-1]
        assert new_op.type == "fused_feedforward"
        new_op._set_attr("ring_id", int(group.id))

    @staticmethod
    def backward(ctx, *args, **kwargs):

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)

        if rank_id not in op_dist_attr.process_mesh.process_ids:
            rank_id = _get_corresponding_rank(
                ctx, op_dist_attr.process_mesh, rank_id
            )

        # infer logic comm presentation
        linear2_weight = src_op.input('Linear2Weight')[0]
        linear2_weight_col_dim_mapping = op_dist_attr.get_input_dims_mapping(
            linear2_weight
        )[-1]
        assert (
            linear2_weight_col_dim_mapping >= 0
        ), "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            linear2_weight_col_dim_mapping
        )
        process_mesh_shape = op_dist_attr.process_mesh.shape
        process_mesh_group = op_dist_attr.process_mesh.process_ids

        parallel_axis = linear2_weight_col_dim_mapping
        group_ranks = _get_comm_group(
            process_mesh_group, process_mesh_shape, parallel_axis, rank_id
        )
        group = new_process_group(group_ranks)

        # insert op
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)

        # setting comm id
        new_op = main_block.ops[-1]
        assert new_op.type == "fused_feedforward_grad"
        new_op._set_attr("ring_id", int(group.id))


register_distributed_operator_impl(
    "fused_feedforward", DistributedFusedFeedForwardImpl("tensor_parallel")
)
