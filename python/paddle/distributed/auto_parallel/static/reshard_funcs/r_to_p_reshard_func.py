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

from .base_reshard_func import (
    ReshardFunction,
    is_partial,
    is_replicated,
    is_shard,
)


class RToPReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_replicated(src_dist_attr):
            return False

        if not is_partial(dst_dist_attr) or is_shard(dst_dist_attr):
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
        dst_mesh = dst_dist_attr.process_mesh
        dst_reduce_type = dst_dist_attr.partial_status[0]
        local_rank = paddle.distributed.get_rank()

        assert dst_reduce_type in [
            paddle.base.core.ReduceType.kRedSum,
            paddle.distributed.ReduceType.kRedAvg,
            paddle.distributed.ReduceType.kRedMax,
        ], f"Unsupported reduce type {dst_reduce_type}"

        if (
            dst_reduce_type == paddle.distributed.ReduceType.kRedSum
            and local_rank != 0
        ):
            dst_value = paddle.full(src_value.shape, 0, dtype=src_value.dtype)
        else:
            dst_value = paddle.assign(src_value)

        chunk_id = -1
        if src_value.get_defining_op().dist_attr:
            chunk_id = src_value.get_defining_op().dist_attr.chunk_id
        dst_value.set_type(dst_type)
        dst_value.get_defining_op().dist_attr = (
            paddle.base.libpaddle.pir.create_op_dist_attribute(
                dst_mesh, [src_dist_attr], [dst_dist_attr], chunk_id
            )
        )

        return dst_value
