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

from .common import DistributedOperatorImplContainer
from .common import DistributedOperatorImpl
from .common import register_distributed_operator_impl_container
from .common import register_distributed_operator_impl
from ..utils import is_dim_shard, is_dim_replicate
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping
from .dist_default import DistributedDefaultImpl0
from ..utils import _get_comm_group, _get_corresponding_rank
from ..process_group import new_process_group


class DistributedFusedAttention(DistributedOperatorImplContainer):
    def __init__(self, op_type):
        super(DistributedFusedAttention, self).__init__(op_type)


register_distributed_operator_impl_container(
    DistributedFusedAttention("fused_attention"))


class DistributedFusedAttentionImpl(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedFusedAttentionImpl, self).__init__(name)
        self._forward_implemented = True
        self._backward_implemented = True

    def is_input_compatible(self, dist_op):
        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        qkv_w = op_desc.input('QKVW')[0]
        qkv_bias = op_desc.input('QKVBias')[0]
        out_w = op_desc.input('OutLinearW')[0]
        out_bias = op_desc.input('OutLinearBias')[0]

        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        qkv_w_dims_mapping = op_dist_attr.get_input_dims_mapping(qkv_w)
        qkv_bias_dims_mapping = op_dist_attr.get_input_dims_mapping(qkv_bias)
        out_w_dims_mapping = op_dist_attr.get_input_dims_mapping(out_w)
        out_bias_dims_mapping = op_dist_attr.get_input_dims_mapping(out_bias)

        head_axis = 1
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        if len(qkv_w_dims_mapping) != 4 or is_dim_replicate(qkv_w_dims_mapping[
                head_axis]):
            return False
        if len(qkv_bias_dims_mapping) != 3 or is_dim_replicate(
                qkv_bias_dims_mapping[head_axis]):
            return False
        if is_dim_replicate(out_w_dims_mapping[0]):
            return False
        if is_dim_shard(out_bias_dims_mapping[-1]):
            return False

        replicated_dims = [
            qkv_w_dims_mapping[0], qkv_w_dims_mapping[-2],
            qkv_w_dims_mapping[-1], qkv_bias_dims_mapping[0],
            qkv_bias_dims_mapping[-1], out_w_dims_mapping[-1],
            out_bias_dims_mapping[-1]
        ]
        for mapping in replicated_dims:
            if is_dim_shard(mapping):
                return False
        if qkv_bias_dims_mapping[head_axis] != qkv_w_dims_mapping[head_axis]:
            return False
        if qkv_bias_dims_mapping[head_axis] != out_w_dims_mapping[0]:
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
        if (not self.is_input_compatible(dist_op)) or \
            (not self.is_output_compatible(dist_op)):
            return False

        op_desc = dist_op.serial_op.desc
        op_dist_attr = dist_op.dist_attr
        x_name = op_desc.input('X')[0]
        out_names = op_desc.output('Y')
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
        out_names = op_desc.output('Y')
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)

        for out_name in out_names:
            out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
            for i in range(len(x_dims_mapping)):
                dim_changed = compute_compatible_and_update_dim_mapping(
                    [x_dims_mapping, out_dims_mapping], [i, i])
                if dim_changed:
                    changed = True

        return changed

    @staticmethod
    def forward(ctx, *args, **kwargs):

        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)

        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)

        # infer logic comm presentation
        head_axis = 1
        qkv_w = src_op.input('QKVW')[0]
        qkv_w_col_dim_mapping = op_dist_attr.get_input_dims_mapping(qkv_w)[
            head_axis]
        assert qkv_w_col_dim_mapping >= 0, "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            qkv_w_col_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        parallel_axis = qkv_w_col_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
        group = new_process_group(group_ranks)

        # insert op
        DistributedDefaultImpl0.forward(ctx, *args, **kwargs)

        # setting comm id
        new_op = main_block.ops[-1]
        assert new_op.type == "fused_attention"
        new_op._set_attr("ring_id", int(group.id))

    @staticmethod
    def backward(ctx, *args, **kwargs):
        dist_op_context = ctx.dist_op_context
        main_block = dist_op_context.work_block
        startup_block = dist_op_context.startup_block
        src_op = dist_op_context.cur_src_op
        rank_id = dist_op_context.rank_id
        op_dist_attr = ctx.get_op_dist_attr_for_program(src_op)

        if rank_id not in op_dist_attr.process_mesh.processes:
            rank_id = _get_corresponding_rank(ctx, op_dist_attr.process_mesh,
                                              rank_id)

        # infer logic comm presentation
        out_w = src_op.input('OutLinearW')[0]
        out_w_col_dim_mapping = op_dist_attr.get_input_dims_mapping(out_w)[-1]
        assert out_w_col_dim_mapping >= 0, "col_parallel_matmul's row should be divided by a specific mesh axis, but got [{}]".format(
            out_w_col_dim_mapping)
        process_mesh_shape = op_dist_attr.process_mesh.topology
        process_mesh_group = op_dist_attr.process_mesh.processes

        parallel_axis = out_w_col_dim_mapping
        group_ranks = _get_comm_group(process_mesh_group, process_mesh_shape,
                                      parallel_axis, rank_id)
        group = new_process_group(group_ranks)

        # insert op
        DistributedDefaultImpl0.backward(ctx, *args, **kwargs)

        # setting comm id
        new_op = main_block.ops[-1]
        assert new_op.type == "fused_attention_grad"
        new_op._set_attr("ring_id", int(group.id))


register_distributed_operator_impl(
    "fused_attention", DistributedFusedAttentionImpl("tensor_parallel"))
