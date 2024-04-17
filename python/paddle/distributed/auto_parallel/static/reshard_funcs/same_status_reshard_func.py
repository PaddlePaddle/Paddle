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

import paddle

from ..process_group import new_process_group
from .base_reshard_func import ReshardFunction


class SameStatusReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if src_dist_attr.dims_mapping == dst_dist_attr.dims_mapping:
            return False
        if src_dist_attr.partial_dims == dst_dist_attr.partial_dims:
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh != out_mesh:
            return False
        if in_mesh.shape == out_mesh.shape:
            return False
        return True

    def reshard(self, program, op, src_dist_attr, dst_dist_attr):
        src_mesh = src_dist_attr.process_mesh
        dst_mesh = dst_dist_attr.process_mesh

        all_process_ids = set(src_mesh.process_ids) | set(dst_mesh.process_ids)

        dtype = op.operand_source(0).dtype

        def get_local_rank(all_process_ids, global_rank=-1):
            if global_rank == -1:
                global_rank = paddle.distributed.get_rank()
            for idx, val in enumerate(all_process_ids):
                if global_rank == val:
                    return idx
            return -1

        local_rank_map = {}
        for src, dst in zip(src_mesh.process_ids, dst_mesh.process_ids):
            curr_global_rank = paddle.distributed.get_rank()
            if src == curr_global_rank:
                dst_local_rank = get_local_rank(all_process_ids, dst)
                local_rank_map["dst_local_rank"] = dst_local_rank
            elif dst == curr_global_rank:
                src_local_rank = get_local_rank(all_process_ids, src)
                local_rank_map["src_local_rank"] = src_local_rank

        paddle.pir.set_insertion_point(op)
        group = new_process_group(src_mesh.process_ids)
        paddle._pir_ops.send_v2(
            op.operand_source(0),
            group.id,
            local_rank_map['dst_local_rank'],
            False,
            True,
        )
        recv_value = paddle._pir_ops.recv_v2(
            [], dtype, local_rank_map['src_local_rank'], group.id, False, True
        )

        recv_value.set_type(op.result(0).type())
        op.result(0).replace_all_uses_with(recv_value)
        program.global_block().remove_op(op)

        for op in program.global_block().ops:
            if op.name() == "pd_op.send_v2":
                op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        src_mesh, [src_dist_attr], []
                    )
                )
            elif op.name() == "pd_op.recv_v2":
                op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        dst_mesh, [], [dst_dist_attr]
                    )
                )

        return recv_value.get_defining_op(), dst_dist_attr
