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
from .base_reshard_func import ReshardFunction, is_partial, is_replicated
from .same_status_reshard_func import SameStatusReshardFunction


class PToRReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_partial(src_dist_attr):
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

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        src_mesh = src_dist_attr.process_mesh
        src_reduce_type = src_dist_attr.partial_status[0]
        # reduce_mean = False
        # if src_reduce_type == paddle.base.core.ReduceType.kRedAvg:
        #     src_reduce_type = paddle.base.core.ReduceType.kRedSum
        #     reduce_mean = True

        group = new_process_group(sorted(src_mesh.process_ids))
        reduced_value = paddle._C_ops.all_reduce(
            src_value, group.id, int(src_reduce_type)
        )
        # set dist type and dist attr
        reduced_value.set_type(dst_type)
        chunk_id = -1
        if src_value.get_defining_op().dist_attr:
            chunk_id = src_value.get_defining_op().dist_attr.chunk_id

        reduced_value.get_defining_op().dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                src_mesh,
                [src_dist_attr],
                [dst_dist_attr],
                chunk_id,
            )
        )
        return reduced_value


class PToRReshardFunctionCrossMesh(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_partial(src_dist_attr):
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

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        same_status_func = SameStatusReshardFunction()
        tmp_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            dst_dist_attr.process_mesh,
            src_dist_attr.dims_mapping,
            src_dist_attr.partial_status,
        )
        tmp_dst_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            src_value.type(), tmp_dist_attr
        )
        src_value = same_status_func.reshard(
            src_dist_attr, tmp_dist_attr, src_value, tmp_dst_type
        )

        p_to_r_func = PToRReshardFunction()
        assert p_to_r_func.is_suitable(
            tmp_dist_attr, dst_dist_attr
        ), f"Invoke the p to r reshard function is not valid from {tmp_dist_attr} to {dst_dist_attr}"
        return p_to_r_func.reshard(
            tmp_dist_attr, dst_dist_attr, src_value, dst_type
        )
