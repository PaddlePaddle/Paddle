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


class SToRReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not self.is_shard(src_dist_attr):
            return False

        if not self.is_replicated(dst_dist_attr):
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
        def get_split_axis_with_dims_mapping(dims_mapping):
            split_axis = {}
            for idx, v in enumerate(dims_mapping):
                if v != -1:
                    split_axis[idx] = v
            return split_axis

        split_axis_map = get_split_axis_with_dims_mapping(
            src_dist_attr.dims_mapping
        )

        split_axis = -1
        for k, v in split_axis_map.items():
            split_axis = k
            break

        op_operand_value = op.operand_source(0)
        num_of_padding = (
            op_operand_value.shape[split_axis] % src_dist_attr.process_mesh.size
        )
        is_balanced_split = num_of_padding == 0

        if is_balanced_split:
            new_value = self.reshard_s_to_r_with_padding(
                program, op, split_axis, src_dist_attr.process_mesh.process_ids
            )
            return new_value, dst_dist_attr
        else:
            # TODO(ywt01) support unbalanced split
            pass

    def reshard_s_to_r_with_padding(
        self, program, op, split_axis, process_ids, padding_num=0
    ):
        num_of_process = len(process_ids)
        dtype = op.operand_source(0).dtype

        paddle.pir.set_insertion_point(op)
        group = new_process_group(process_ids)
        op_value = op.operand_source(0)
        allgather_value = paddle._pir_ops.c_allgather(
            op_value, group.id, num_of_process, False
        )
        op.result(0).replace_all_uses_with(allgather_value)
        program.global_block().remove_op(op)

        if split_axis != 0 or padding_num != 0:
            sections = num_of_process * [op_value.shape[0]]
            allgather_op = allgather_value.get_defining_op()
            paddle.pir.set_insertion_point_after(allgather_op)
            split_value = paddle._pir_ops.split(
                allgather_op.result(0), sections, 0
            )
            concat_value = paddle._pir_ops.concat(split_value, split_axis)
            return concat_value.get_defining_op()
        return allgather_value.get_defining_op()
