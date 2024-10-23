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


from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import Tensor
from paddle.autograd import PyLayer

from .placement_type import check_placements_equal
from .static.reshard_funcs.nd_mesh_reshard_func import get_1D_sub_process_mesh

if TYPE_CHECKING:
    from paddle.distributed import Placement
    from paddle.distributed.auto_parallel.process_mesh import ProcessMesh


def _specific_alltoall_dim(
    dist_tensor: Tensor, mesh: ProcessMesh, placements: list[Placement]
):
    """
    Get the specific dimension for alltoall communication in nd_mesh reshard.
    """
    mesh_dim = None
    src_mesh = dist_tensor.process_mesh
    src_placements = dist_tensor.placements

    if src_mesh != mesh or src_mesh.ndim == 1:
        return None

    if any(p.is_partial() for p in src_placements):
        return None
    if any(p.is_partial() for p in placements):
        return None

    for i in range(min(len(src_placements), len(placements))):
        src_p = src_placements[i]
        dst_p = placements[i]
        if src_p.is_shard() and dst_p.is_shard() and src_p != dst_p:
            # reshard from shard to shard, needs alltoall
            # now only supports reshard on one dimension
            src_dim = src_p.get_dim()
            dst_dim = dst_p.get_dim()
            if mesh_dim is not None or abs(src_dim - dst_dim) != 1:
                return None
            else:
                mesh_dim = i

    return mesh_dim


class _NdMeshAlltoAll(PyLayer):
    @staticmethod
    def forward(
        ctx,
        dist_tensor: Tensor,
        mesh: ProcessMesh,
        placements: list[Placement],
        dim: int,
    ):
        sub_mesh = get_1D_sub_process_mesh(mesh, dim)
        ctx.alltoall_dim = dim
        ctx.x_mesh = copy.deepcopy(dist_tensor.process_mesh)
        ctx.x_placements = copy.deepcopy(dist_tensor.placements)
        ctx.out_mesh = copy.deepcopy(mesh)
        ctx.out_placements = copy.deepcopy(placements)

        out = dist.auto_parallel.api.dtensor_from_local(
            dist_tensor._local_value(), sub_mesh, [dist_tensor.placements[dim]]
        )
        out = dist.reshard(out, sub_mesh, [placements[dim]])
        out = dist.auto_parallel.api.dtensor_from_local(
            out._local_value(), mesh, placements
        )
        out.stop_gradient = dist_tensor.stop_gradient
        return out

    @staticmethod
    def backward(ctx, out_grad):
        if not check_placements_equal(ctx.out_placements, out_grad.placements):
            out = dist.reshard(out_grad, ctx.out_mesh, ctx.out_placements)
        out = _NdMeshAlltoAll.apply(
            out_grad, ctx.x_mesh, ctx.x_placements, ctx.alltoall_dim
        )
        return out


def _cal_local_shape(global_shape, mesh, placements):
    local_shape = list(global_shape)
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = placement.get_dim()
            local_shape[shard_dim] = local_shape[shard_dim] // mesh.shape[idx]
    return local_shape


def infer_pos_shape(src_shape, tgt_shape):
    if isinstance(tgt_shape, (list, tuple)):
        ret_shape = np.array(tgt_shape)
    else:
        ret_shape = tgt_shape.copy()

    neg_one_idx = np.where(ret_shape == -1)[0]
    if neg_one_idx.size > 0:
        assert (
            neg_one_idx.size <= 1
        ), "At most one -1 is allowed in target shape."

        nelem = np.prod(src_shape)
        ret_shape[neg_one_idx[0]] = 1
        ret_shape[neg_one_idx[0]] = nelem // np.prod(ret_shape)

    return list(ret_shape)


class _local_reshape(PyLayer):
    @staticmethod
    def forward(
        ctx,
        dist_tensor: Tensor,
        global_shape: list,
        mesh: ProcessMesh,
        placements: list[Placement],
    ):
        place = paddle.framework._current_expected_place()
        place = paddle.framework._get_paddle_place(place)

        local_tensor = dist_tensor._local_value().clone()
        ctx.x_global_shape = copy.deepcopy(dist_tensor.shape)
        ctx.x_local_shape = copy.deepcopy(local_tensor.shape)
        ctx.x_mesh = copy.deepcopy(dist_tensor.process_mesh)
        ctx.x_placements = copy.deepcopy(dist_tensor.placements)

        local_shape = _cal_local_shape(global_shape, mesh, placements)
        assert np.prod(local_shape) == np.prod(
            local_tensor.shape
        ), f"The local shapes {local_shape} and {local_tensor.shape} are mismatched."
        local_tensor = local_tensor.reshape(local_shape)
        out = paddle.Tensor(
            local_tensor,
            dims=global_shape,
            process_mesh=mesh,
            placements=placements,
            place=place,
        )
        out.stop_gradient = dist_tensor.stop_gradient
        return out

    @staticmethod
    def backward(ctx, out_grad):
        place = paddle.framework._current_expected_place()
        place = paddle.framework._get_paddle_place(place)

        local_grad = out_grad._local_value().clone()
        local_grad = local_grad.reshape(ctx.x_local_shape)
        ret = paddle.Tensor(
            local_grad,
            dims=ctx.x_global_shape,
            process_mesh=ctx.x_mesh,
            placements=ctx.x_placements,
            place=place,
        )
        return ret


def _dist_reshape(
    dist_tensor: Tensor,
    global_shape: list,
    mesh: ProcessMesh,
    placements: list[Placement],
):
    """
    Reshape the local tensors of the dist tensor on each rank,
    and mannualy set the process_mesh and placements of the output.
    """
    tgt_global_shape = infer_pos_shape(dist_tensor.shape, global_shape)
    if paddle.in_dynamic_mode():
        return _local_reshape.apply(
            dist_tensor, tgt_global_shape, mesh, placements
        )
    else:
        raise NotImplementedError(
            "dist_reshape is only supported in dynamic mode."
        )


def _reshard_mesh_shape(
    dist_tensor: Tensor, mesh: ProcessMesh, placements: list[Placement]
):
    src_mesh = dist_tensor.process_mesh
    if src_mesh == mesh or src_mesh.process_ids != mesh.process_ids:
        return False

    # only the mesh shapes are different,
    # if the placements are all replicate,
    # then we can reshard the mesh shapes
    if not all(p.is_replicated() for p in dist_tensor.placements + placements):
        return False

    return True
