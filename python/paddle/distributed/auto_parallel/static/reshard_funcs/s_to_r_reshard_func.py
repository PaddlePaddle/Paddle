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
from .base_reshard_func import ReshardFunction, is_replicated, is_shard
from .same_status_reshard_func import SameStatusReshardFunction


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

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
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

        num_of_padding = (
            src_value.shape[split_axis] % src_dist_attr.process_mesh.size
        )
        is_balanced_split = num_of_padding == 0

        if is_balanced_split:
            new_value = self.reshard_s_to_r_with_padding(
                src_value,
                split_axis,
                src_dist_attr,
                dst_dist_attr,
                dst_type,
                num_of_padding,
            )
            return new_value
        else:
            # TODO(ywt01) support unbalanced split
            pass
        return None

    def reshard_s_to_r_with_padding(
        self,
        src_value,
        split_axis,
        src_dist_attr,
        dst_dist_attr,
        dst_type,
        padding_num=0,
    ):
        src_mesh = src_dist_attr.process_mesh
        num_of_process = len(src_mesh.process_ids)
        dtype = src_value.dtype
        group = new_process_group(src_mesh.process_ids)
        allgather_value = paddle._C_ops.c_allgather(
            src_value, group.id, num_of_process, True
        )
        allgather_value.set_type(dst_type)

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

        if split_axis != 0 or padding_num != 0:
            allgather_op = allgather_value.get_defining_op()
            paddle.pir.set_insertion_point_after(allgather_op)
            split_value = paddle._C_ops.split_with_num(
                allgather_op.result(0), num_of_process, 0
            )
            concat_value = paddle._C_ops.concat(split_value, split_axis)
            return concat_value
        return allgather_value


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
            s_to_r_func = SToRReshardFunction()
            assert s_to_r_func.is_suitable(
                tmp_dist_attr, dst_dist_attr
            ), f"Invoke the p to r reshard function is not valid from {tmp_dist_attr} to {dst_dist_attr}"
            return s_to_r_func.reshard(
                tmp_dist_attr, dst_dist_attr, out_value, dst_type
            )
        return None
