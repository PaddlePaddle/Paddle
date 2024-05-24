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

import copy

import paddle
import paddle.distributed as dist

from .base_reshard_func import (
    ReshardFunction,
    copy_dist_attr_with_new_member,
    is_partial,
)
from .p_to_r_reshard_func import PToRReshardFunction
from .r_to_s_reshard_func import RToSReshardFunction
from .s_to_r_reshard_func import SToRReshardFunction
from .same_status_reshard_func import SameStatusReshardFunction


def find_first_diff_shard_axis(src_dist_attr, dst_dist_attr):
    src_dims_mapping = src_dist_attr.dims_mapping
    dst_dims_mapping = dst_dist_attr.dims_mapping
    ndim = len(src_dims_mapping)
    for i in range(ndim - 1, -1, -1):
        if src_dims_mapping[i] != dst_dims_mapping[i]:
            return i
    return -1


def get_1D_sub_process_mesh(process_mesh, mesh_dim):
    """
    Get the 1-D sub process mesh on specific mesh_dim which:
      1) where the reshard should be performed.
      2) contains current process.

    Args:
      process_mesh (ProcessMesh): the global process mesh.
      mesh_dim (int): the mesh dimension where the dist_tensor is
        sharded or partial.

    e.g.
      1) process_mesh = [[0, 1, 2], [3, 4, 5]], axis = 0:
         process rank id      returned sub mesh
           0 or 3               [0, 3]
           1 or 4               [1, 4]
           2 or 5               [2, 5]
      2) process_mesh = [[0, 1, 2], [3, 4, 5]], axis = 1:
         process rank id      returned sub mesh
           0 or 1 or 2          [0, 1, 2]
           3 or 4 or 5          [3, 4, 5]
    """
    import numpy as np

    mesh_shape = process_mesh.shape
    dim_names = process_mesh.dim_names
    process_ids = np.array(process_mesh.process_ids).reshape(mesh_shape)

    rank_id = dist.get_rank()
    coord = list(np.where(process_ids == rank_id))
    coord[mesh_dim] = range(mesh_shape[mesh_dim])
    sub_process_ids = process_ids[tuple(coord)].flatten()
    sub_mesh_shape = sub_process_ids.shape
    sub_mesh_name = dim_names[mesh_dim]

    return dist.ProcessMesh(sub_process_ids, [sub_mesh_name])


class NdMeshReshardFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh != out_mesh:
            return False
        if out_mesh.ndim <= 1:
            return False
        # check dims_mapping and partial_status
        if src_dist_attr == dst_dist_attr:
            return False

        return True

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        """
        Reshard on N-d mesh:
          1. Find the tensor dimensions where the dims_mapping values
             differ between src_dist_attr and dst_dist_attr.
          2. From higher to lower, convert the non-replicated dimensions
             in step1 to replicated using corresponding 1-D mesh functions.
          3. Convert the replicated dimensions in step2 to the status in
             dst_dist_attr with corresponding 1-D mesh functions.
        """
        # Step1. find first dimension with different shard status in src_dist_attr
        # and dst_dist_attr.
        first_diff_axis = find_first_diff_shard_axis(
            src_dist_attr, dst_dist_attr
        )
        ori_dst_dist_attr = copy_dist_attr_with_new_member(dst_dist_attr)
        out_value = src_value  # intermediate result
        src_type = src_value.type()
        tensor_ndim = len(src_value.shape)
        process_mesh = dst_dist_attr.process_mesh

        # Step2. Convert the non-replicated dimensions to replicated.
        # Step2.1. convert partial status to replicated
        real_out_dist_attr = copy_dist_attr_with_new_member(src_dist_attr)
        if is_partial(src_dist_attr):
            in_partial_status = copy.deepcopy(src_dist_attr.partial_status)
            out_partial_status = dst_dist_attr.partial_status  # read-only
            # convert each partial dim to replicated with corresponding
            # 1-D mesh function
            for partial_dim, partial_type in in_partial_status.items():
                if (
                    partial_dim in out_partial_status
                    or partial_dim in ori_dst_dist_attr.dims_mapping
                ):
                    continue

                # get the partial status after converting
                real_out_partial_status = copy.deepcopy(
                    real_out_dist_attr.partial_status
                )
                real_out_partial_status.pop(partial_dim)
                real_out_dist_attr = copy_dist_attr_with_new_member(
                    real_out_dist_attr,
                    new_partial_status=real_out_partial_status,
                )

                # get the process_mesh on specific axis
                sub_mesh = get_1D_sub_process_mesh(process_mesh, partial_dim)

                # calculate corresponding 1-D dist_attr of src_dst_attr
                in_one_dim_partial_status = {0: partial_type}
                in_one_dim_dist_attr = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        sub_mesh,
                        [-1] * tensor_ndim,
                        in_one_dim_partial_status,
                    )
                )

                # calculate corresponding 1-D dist_attr of dst_dst_attr
                out_one_dim_dist_attr = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        sub_mesh,
                        [-1] * tensor_ndim,
                        {},
                    )
                )

                one_dim_func = PToRReshardFunction()
                out_value = one_dim_func.reshard(
                    in_one_dim_dist_attr,
                    out_one_dim_dist_attr,
                    out_value,
                    src_type,
                )

                out_value.update_dist_attr(real_out_dist_attr)

        # Step2.2 convert shard status to replicated
        for i in range(first_diff_axis, -1, -1):
            in_mesh_axis = real_out_dist_attr.dims_mapping[i]
            if in_mesh_axis == -1:
                continue

            # calculate the dist_attr after converting
            real_out_dims_mapping = copy.deepcopy(
                real_out_dist_attr.dims_mapping
            )
            real_out_dims_mapping[i] = -1
            real_out_dist_attr = copy_dist_attr_with_new_member(
                real_out_dist_attr, new_dims_mapping=real_out_dims_mapping
            )

            # get the process_mesh on specific axis
            sub_mesh = get_1D_sub_process_mesh(process_mesh, in_mesh_axis)

            # calculate corresponding 1-D dist_attr of src_dst_attr
            in_one_dim_dims_mapping = [-1] * tensor_ndim
            in_one_dim_dims_mapping[i] = 0
            in_one_dim_dist_attr = (
                paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                    sub_mesh, in_one_dim_dims_mapping, {}
                )
            )

            # calculate corresponding 1-D dist_attr of dst_dst_attr
            out_one_dim_dims_mapping = [-1] * tensor_ndim
            out_one_dim_dist_attr = (
                paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                    sub_mesh, out_one_dim_dims_mapping, {}
                )
            )

            one_dim_func = SToRReshardFunction()
            out_value = one_dim_func.reshard(
                in_one_dim_dist_attr, out_one_dim_dist_attr, out_value, src_type
            )

            out_value.update_dist_attr(real_out_dist_attr)

        # Step3. Convert the replicated status to the status in dst_dist_attr
        # Step3.1 convert replicated to partial
        if is_partial(ori_dst_dist_attr):
            in_partial_status = out_value.dist_attr.partial_status
            out_partial_status = ori_dst_dist_attr.partial_status
            for partial_dim, partial_type in out_partial_status.items():
                if partial_dim in in_partial_status:
                    continue

                raise NotImplementedError(
                    "RToPReshardFunction is not implemented"
                )

        # Step3.2 convert replicated/partial to shard
        for i in range(first_diff_axis, -1, -1):
            out_mesh_axis = ori_dst_dist_attr.dims_mapping[i]
            if out_mesh_axis == -1:
                continue
            in_partial_status = out_value.dist_attr().partial_status
            need_p2s = out_mesh_axis in in_partial_status
            dims_mapping = copy.deepcopy(real_out_dist_attr.dims_mapping)
            dims_mapping[i] = out_mesh_axis
            partial_status = None
            if out_mesh_axis in real_out_dist_attr.partial_status:
                partial_status = copy.deepcopy(
                    real_out_dist_attr.partial_status
                )
                partial_status.pop(out_mesh_axis)

            real_out_dist_attr = copy_dist_attr_with_new_member(
                real_out_dist_attr,
                new_dims_mapping=dims_mapping,
                new_partial_status=partial_status,
            )

            # get the process_mesh on specific axis
            sub_mesh = get_1D_sub_process_mesh(process_mesh, out_mesh_axis)

            # calculate the corresponding 1-D input dist attr
            in_one_dim_dims_mapping = [-1] * tensor_ndim
            in_one_dim_dist_attr = (
                paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                    sub_mesh, in_one_dim_dims_mapping, {}
                )
            )

            # calculate the corresponding 1-D output dist attr
            out_one_dim_dims_mapping = [-1] * tensor_ndim
            out_one_dim_dims_mapping[i] = 0
            out_one_dim_dist_attr = (
                paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                    sub_mesh, out_one_dim_dims_mapping, {}
                )
            )

            if need_p2s:
                raise NotImplementedError(
                    "PToSReshardFunction is not implemented"
                )
            else:
                one_dim_func = RToSReshardFunction()
                out_value = one_dim_func.reshard(
                    in_one_dim_dist_attr,
                    out_one_dim_dist_attr,
                    out_value,
                    dst_type,
                )
                out_value.update_dist_attr(real_out_dist_attr)

        out_value.set_type(dst_type)
        return out_value


class NdMeshReshardFunctionCrossMesh(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh

        if in_mesh == out_mesh:
            return False
        if in_mesh.shape != out_mesh.shape:
            return False
        if out_mesh.ndim <= 1:
            return False
        if src_dist_attr == dst_dist_attr:
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
            nd_mesh_func = NdMeshReshardFunction()
            assert nd_mesh_func.is_suitable(
                tmp_dist_attr, dst_dist_attr
            ), f"Invoke the p to r reshard function is not valid from {tmp_dist_attr} to {dst_dist_attr}"
            return nd_mesh_func.reshard(
                tmp_dist_attr, dst_dist_attr, out_value, dst_type
            )
        return None
