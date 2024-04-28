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

from .base_reshard_func import ReshardFunction, is_replicated, is_shard


class RToSReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_replicated(src_dist_attr):
            return False

        if not is_shard(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh.ndim != 1:
            return False
        if out_mesh.ndim != 1:
            return False
        if in_mesh != out_mesh:
            return False
        return True

    def reshard(
        self, program, op, src_dist_attr, dst_dist_attr, remove_op=True
    ):
        split_axis = -1
        mesh_axis = -1
        for idx, v in enumerate(dst_dist_attr.dims_mapping):
            if v != -1:
                split_axis = idx
                mesh_axis = v

        mesh = src_dist_attr.process_mesh
        curr_global_rank = paddle.distributed.get_rank()
        if curr_global_rank in mesh.process_ids:
            total_nums = op.operand_source(0).shape[split_axis]
            num_of_pieces = mesh.shape[mesh_axis]
            piece_len = (total_nums + num_of_pieces - 1) // num_of_pieces
            rank_relative = mesh.process_ids.index(curr_global_rank)
            start = rank_relative * piece_len
            end = start + piece_len
            if curr_global_rank == mesh.process_ids[-1]:
                end = total_nums

            paddle.pir.set_insertion_point(op)
            out_value = paddle.slice(
                op.operand_source(0), [split_axis], [start], [end]
            )
            op.result(0).replace_all_uses_with(out_value)
            op.get_parent_block().remove_op(op)
