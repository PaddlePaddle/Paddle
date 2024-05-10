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
import numpy as np

from ..process_group import new_process_group
from .base_reshard_func import ReshardFunction, is_replicated, is_shard
from .same_status_reshard_func import SameStatusReshardFunction
from paddle.distributed.auto_parallel.placement_type import (
    to_placements,
)
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh

class SToRReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_shard(src_dist_attr):
            return False

        if not is_replicated(dst_dist_attr):
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
        self, program, op, src_dist_attr, dst_dist_attr, reshard_op=True
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

        print(f'debug program: {program}')
        print(f'debug op: {op}')
        print(f'debug is reshard_op: {reshard_op}')
        op_value = op.result(0)
        if reshard_op:
            op_value = op.operand_source(0)

        num_of_padding = (
            op_value.shape[split_axis] % src_dist_attr.process_mesh.size
        )
        is_balanced_split = num_of_padding == 0

        print(f'debug is_balanced_split: {is_balanced_split}')
        if is_balanced_split:
            new_value = self.reshard_s_to_r_with_padding(
                program, op, op_value, split_axis, src_dist_attr, dst_dist_attr, num_of_padding, reshard_op
            )
            return new_value, dst_dist_attr
        else:
            # TODO(ywt01) support unbalanced split
            pass

    def reshard_s_to_r_with_padding(
        self, program, op, op_value, split_axis, src_dist_attr, dst_dist_attr, padding_num=0, reshard_op=True
    ):
        print(f'debug op_value: {op_value}')
        print(f'debug reshard_op: {reshard_op}')
        src_mesh = src_dist_attr.process_mesh
        num_of_process = len(src_mesh.process_ids)
        dtype = op_value.dtype

        if reshard_op:
            paddle.pir.set_insertion_point(op)
        else:
            paddle.pir.set_insertion_point_after(op)
        group = new_process_group(src_mesh.process_ids)
        allgather_value = paddle._pir_ops.c_allgather(
            op_value, group.id, num_of_process, False
        )

        dst_mesh = dst_dist_attr.process_mesh
        mesh = ProcessMesh(shape=dst_mesh.shape, process_ids=dst_mesh.process_ids)
        placements = to_placements(dst_dist_attr.dims_mapping, mesh)
        allgather_value = paddle.distributed.auto_parallel.api.dtensor_from_local(allgather_value, mesh, placements)

        # set op_dist_attr
        new_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh,
            [-1] * len(dst_dist_attr.dims_mapping),
            dst_dist_attr.partial_status,
        )
        allgather_value.get_defining_op().dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                src_mesh, [src_dist_attr], [new_dist_attr]
            )
        )

        if reshard_op:
            op.result(0).replace_all_uses_with(allgather_value)
            program.global_block().remove_op(op)

        print(f'debug split_axis: {split_axis}, padding_num: {padding_num}')
        if split_axis != 0 or padding_num != 0:
            allgather_op = allgather_value.get_defining_op()
            paddle.pir.set_insertion_point_after(allgather_op)
            split_value = paddle._pir_ops.split_with_num(
                allgather_op.result(0), num_of_process, 0
            )
            concat_value = paddle._pir_ops.concat(split_value, split_axis)
            print(f'debug 2')
            return concat_value.get_defining_op()
        else: 
            print(f'debug 1')
        print(f'debug finish s_to_r reshard')
        return allgather_value.get_defining_op()
        
class SToRReshardFunctionCrossMesh(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_shard(src_dist_attr):
            return False

        if not is_replicated(dst_dist_attr):
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if (
            in_mesh.ndim != 1
            or out_mesh.ndim != 1
            or in_mesh.shape != out_mesh.shape
        ):
            return False

        if in_mesh == out_mesh:
            return False

        return True

    def reshard(self, program, op, src_dist_attr, dst_dist_attr):
        same_status_func = SameStatusReshardFunction()
        tmp_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh,
            src_dist_attr.dims_mapping,
            src_dist_attr.partial_status,
        )
        pre_op, out_dist_attr = same_status_func.reshard(
            program, op, src_dist_attr, tmp_dist_attr
        )

        if pre_op is None:
            return None, out_dist_attr

        curr_global_rank = paddle.distributed.get_rank()
        if curr_global_rank in dst_dist_attr.process_mesh.process_ids:
            s_to_r_func = SToRReshardFunction()
            assert s_to_r_func.is_suitable(
                out_dist_attr, dst_dist_attr
            ), f"Invoke the p to r reshard function is not valid from {pre_op.dist_attr()} to {dst_dist_attr}"
            s_to_r_func.reshard(
                program, pre_op, out_dist_attr, dst_dist_attr, False
            )
