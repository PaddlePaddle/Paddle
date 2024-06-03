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

    def infer_allgather_dist_type(self, in_value, split_axis):
        tensor_ndim = len(in_value.shape)
        in_dist_attr = in_value.dist_attr()
        split_mesh_dim = in_dist_attr.dims_mapping[split_axis]
        mesh = in_dist_attr.process_mesh

        # Calculate local shape. In nd_mesh_reshard, multiple tensor axis
        # may be shard and it will call this 1-D s_to_r function on each
        # axis. In this case, we should recompute the local and global shape.
        out_local_shape = list(in_value.shape)
        out_local_shape[split_axis] = (
            in_value.shape[split_axis] // mesh.shape[split_mesh_dim]
        )
        out_global_shape = list(out_local_shape)
        out_global_shape[0] *= mesh.shape[split_mesh_dim]
        out_type = paddle.pir.create_shaped_type(
            in_value.type(), out_global_shape
        )

        out_dims_mapping = list(in_dist_attr.dims_mapping)
        out_dims_mapping[split_axis] = -1
        out_dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
            mesh, out_dims_mapping, {}
        )
        out_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
            out_type, out_dist_attr
        )
        return out_type

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
            raise NotImplementedError("unbalanced split is not implemented")

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

        group = new_process_group(sorted(src_mesh.process_ids))
        allgather_value = paddle._C_ops.c_allgather(
            src_value, group.id, num_of_process, True
        )
        allgather_type = self.infer_allgather_dist_type(src_value, split_axis)
        allgather_value.set_type(allgather_type)

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
            split_values = paddle._C_ops.split_with_num(
                allgather_op.result(0), num_of_process, 0
            )
            builtin_split_op = split_values[0].get_defining_op()
            pd_splite_op = builtin_split_op.operand_source(0).get_defining_op()

            # fix the split_with_num dist attribtue.
            new_inner_types = []
            for sub_value in split_values:
                new_inner_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    sub_value.type(), allgather_value.dist_attr()
                )
                new_inner_types.append(new_inner_type)
                sub_value.set_type(new_inner_type)
            vec_type = paddle.base.libpaddle.pir.create_vec_type(
                new_inner_types
            )
            pd_splite_op.result(0).set_type(vec_type)

            concat_value = paddle._C_ops.concat(split_values, split_axis)
            # fold builtin.split op and builtin.combine op
            concat_op = concat_value.get_defining_op()
            builtin_combine_op = concat_op.operand_source(0).get_defining_op()
            concat_op.operand(0).set_source(pd_splite_op.result(0))
            builtin_combine_op.erase()
            builtin_split_op.erase()
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

        s_to_r_func = SToRReshardFunction()
        assert s_to_r_func.is_suitable(
            tmp_dist_attr, dst_dist_attr
        ), f"Invoke the p to r reshard function is not valid from {tmp_dist_attr} to {dst_dist_attr}"
        return s_to_r_func.reshard(
            tmp_dist_attr, dst_dist_attr, out_value, dst_type
        )
