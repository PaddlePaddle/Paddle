# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy

import paddle

from ..process_group import new_process_group
from .base_reshard_func import ReshardFunction, copy_dist_attr_with_new_member


class SubToGlobalMeshFunction(ReshardFunction):
    """
    Reshard from sub-mesh to global mesh, now only supports
    both input and output values are replicated, e.g.
    1. input: mesh:[0], placements:[Replicate()]
       output: mesh:[0,1], placements:[Replicate()]
    2. input: mesh:[0,1], placements:[Replicate()]
       output: mesh:[[0,1],[2,3]], placements:[Replicate(), Replicate()]
    """

    def is_suitable(self, src_dist_attr, dst_dist_attr):
        # only supports replicated input and output
        if 0 in src_dist_attr.dims_mapping or 0 in src_dist_attr.partial_status:
            return False
        if 0 in dst_dist_attr.dims_mapping or 0 in dst_dist_attr.partial_status:
            return False
        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh
        if out_mesh.ndim > in_mesh.ndim + 1:
            return False
        if out_mesh.ndim == in_mesh.ndim:
            return set(in_mesh.process_ids) < set(out_mesh.process_ids)
        else:
            sub_meshes = paddle.base.libpaddle.pir.get_sub_meshes(in_mesh)
            return out_mesh in sub_meshes

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        src_mesh = src_dist_attr.process_mesh
        dst_mesh = dst_dist_attr.process_mesh

        root_rank = src_mesh.process_ids[0]
        other_ranks = copy.copy(dst_mesh.process_ids)
        for rank in other_ranks:
            if rank in src_mesh.process_ids:
                other_ranks.remove(rank)

        cur_rank = paddle.distributed.get_rank()

        if cur_rank in src_mesh.process_ids:
            # the root rank will broadcast the src_value to other ranks
            chunk_id = -1
            if src_value.get_defining_op().dist_attr:
                chunk_id = src_value.get_defining_op().dist_attr.chunk_id
            tmp_value = paddle._C_ops.share_data_(src_value)
            value_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                src_value.type(), src_value.dist_attr()
            )
            tmp_value.set_type(value_type)
            op = tmp_value.get_defining_op()
            op.dist_attr = paddle.base.libpaddle.pir.create_op_dist_attribute(
                src_mesh, [src_dist_attr], [src_dist_attr], chunk_id
            )
        else:
            # create the buffer on other ranks for receiving the data
            tmp_value = paddle.zeros(dst_type.shape, dst_type.dtype)
            op = tmp_value.get_defining_op()
            mesh = paddle.distributed.ProcessMesh(other_ranks)
            value_dist_attr = copy_dist_attr_with_new_member(
                dst_dist_attr, new_process_mesh=mesh
            )
            value_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                dst_type, value_dist_attr
            )
            tmp_value.set_type(value_type)
            op.dist_attr = paddle.base.libpaddle.pir.create_op_dist_attribute(
                mesh, [], [value_dist_attr]
            )

        group = new_process_group(sorted(dst_mesh.process_ids))
        broadcast_value = paddle._C_ops.broadcast(
            tmp_value, group.id, root_rank
        )
        broadcast_value.set_type(dst_type)

        broadcast_op = broadcast_value.get_defining_op()
        broadcast_op.dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                dst_mesh, [src_dist_attr], [dst_dist_attr]
            )
        )

        return broadcast_value
