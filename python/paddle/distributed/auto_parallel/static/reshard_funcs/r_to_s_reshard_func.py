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
from .same_status_reshard_func import SameStatusReshardFunction


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

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        split_axis = -1
        mesh_axis = -1
        for idx, v in enumerate(dst_dist_attr.dims_mapping):
            if v != -1:
                split_axis = idx
                mesh_axis = v

        mesh = src_dist_attr.process_mesh
        curr_global_rank = paddle.distributed.get_rank()
        if curr_global_rank in mesh.process_ids:
            total_nums = src_value.shape[split_axis]
            num_of_pieces = mesh.shape[mesh_axis]
            piece_len = (total_nums + num_of_pieces - 1) // num_of_pieces
            rank_relative = mesh.process_ids.index(curr_global_rank)
            start = rank_relative * piece_len
            end = start + piece_len
            if curr_global_rank == mesh.process_ids[-1]:
                end = total_nums

            out_value = paddle.slice(src_value, [split_axis], [start], [end])

            out_value.set_type(dst_type)
            out_value.get_defining_op().dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    mesh, [src_dist_attr], [dst_dist_attr]
                )
            )
            return out_value
        # fake var will be removed in remove_other_rank_op_pass.
        fake_var = paddle._C_ops.reshard_v2(src_value, dst_dist_attr)
        fake_var.set_type(dst_type)
        return fake_var


class RToSReshardFunctionCrossMesh(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if not is_replicated(src_dist_attr):
            return False

        if not is_shard(dst_dist_attr):
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
        out_value = same_status_func.reshard(
            src_dist_attr, tmp_dist_attr, src_value, tmp_dst_type
        )

        if out_value is None:
            return None

        curr_global_rank = paddle.distributed.get_rank()
        if curr_global_rank in dst_dist_attr.process_mesh.process_ids:
            r_to_s_func = RToSReshardFunction()
            assert r_to_s_func.is_suitable(
                tmp_dist_attr, dst_dist_attr
            ), f"Invoke the r to s reshard function is not valid from {tmp_dist_attr} to {dst_dist_attr}"
            return r_to_s_func.reshard(
                tmp_dist_attr, dst_dist_attr, out_value, dst_type
            )
        return None
