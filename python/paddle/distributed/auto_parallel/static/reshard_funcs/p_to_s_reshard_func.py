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
from .base_reshard_func import (
    ReshardFunction,
    copy_dist_attr_with_new_member,
    is_partial,
    is_shard,
)


class PToSReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_partial(src_dist_attr):
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

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        src_mesh = src_dist_attr.process_mesh
        src_reduce_type = src_dist_attr.partial_status[0]
        assert (
            src_reduce_type == paddle.base.core.ReduceType.kRedSum
        ), f"The p to s reshard func only support sum op, but received {src_reduce_type}"

        split_axis = dst_dist_attr.dims_mapping.index(0)
        if split_axis != 0:
            perm = list(range(0, len(src_value.shape)))
            perm[0] = split_axis
            perm[split_axis] = 0
            src_value = paddle._C_ops.transpose(src_value, perm)
            tmp_dims_mapping = dst_dist_attr.dims_mapping
            tmp_dims_mapping[split_axis] = -1
            tmp_dims_mapping[0] = 0
            dst_dist_attr = copy_dist_attr_with_new_member(
                dst_dist_attr, new_dims_mapping=tmp_dims_mapping
            )

            global_dst_attr = dst_type.as_dist_type().dist_attr()
            global_dims_mapping = global_dst_attr.dims_mapping
            axis = global_dims_mapping[0]
            global_dims_mapping[0] = global_dims_mapping[split_axis]
            global_dims_mapping[split_axis] = axis
            global_dist_attr = copy_dist_attr_with_new_member(
                global_dst_attr, new_dims_mapping=global_dims_mapping
            )
            dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                src_value.type(), global_dist_attr
            )

        num_of_process = len(src_mesh.process_ids)
        group = new_process_group(sorted(src_mesh.process_ids))
        dst_value = paddle._C_ops.c_reducescatter(
            src_value, group.id, num_of_process, True
        )

        # set dist type and dist attr
        dst_value.set_type(dst_type)
        dst_value.get_defining_op().dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                src_mesh, [src_dist_attr], [dst_dist_attr]
            )
        )

        if split_axis != 0:
            dst_value = paddle._C_ops.transpose(dst_value, perm)
        return dst_value
