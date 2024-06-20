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
        if src_dist_attr.dims_mapping != dst_dist_attr.dims_mapping:
            return False
        if src_dist_attr.partial_dims != dst_dist_attr.partial_dims:
            return False

        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh == out_mesh:
            return False
        if in_mesh.shape != out_mesh.shape:
            return False
        return True

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        src_mesh = src_dist_attr.process_mesh
        dst_mesh = dst_dist_attr.process_mesh

        all_process_ids = list(
            set(src_mesh.process_ids) | set(dst_mesh.process_ids)
        )
        all_process_ids = sorted(all_process_ids)

        cur_global_rank = paddle.distributed.get_rank()
        comm_group = new_process_group(all_process_ids)

        is_send = True
        for src, dst in zip(src_mesh.process_ids, dst_mesh.process_ids):
            if src == cur_global_rank:
                dst_local_rank = all_process_ids.index(dst)
                paddle._C_ops.send_v2(
                    src_value,
                    comm_group.id,
                    dst_local_rank,
                    True,
                    False,
                )
                point = paddle.base.libpaddle.pir.get_current_insertion_point()
                point.prev()
                new_op = point.get_operation()
                assert new_op.name() == "pd_op.send_v2"
                new_op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        src_mesh, [src_dist_attr], []
                    )
                )
                break

            elif dst == cur_global_rank:
                src_local_rank = all_process_ids.index(src)
                assert (
                    -1 not in dst_type.shape
                ), "dynamic shape is not supported by pir-auto parallel yet."
                recv_value = paddle._C_ops.recv_v2(
                    dst_type._local_shape,
                    dst_type.dtype,
                    src_local_rank,
                    comm_group.id,
                    True,
                    False,
                )
                new_op = recv_value.get_defining_op()
                new_op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        dst_mesh, [], [dst_dist_attr]
                    )
                )
                recv_value.set_type(dst_type)
                is_send = False
                break

        if is_send:
            # fake var will be removed in remove_other_rank_op_pass.
            fake_var = paddle._C_ops.reshard_v2(src_value, dst_dist_attr)
            fake_var.set_type(dst_type)
            return fake_var
        else:
            return recv_value
