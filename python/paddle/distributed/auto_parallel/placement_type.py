#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from typing import cast

import paddle
from paddle.base.core import Partial, Replicate, Shard


def dims_mapping_to_placements(dim_map, mesh, partial_idx=[], split_factor={}):
    """
    convert dim_map to placements.

    Args:
        dim_map(List[int]): a list of integer that represents sharding on each tensor dimension.
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        partial_idx(List[int], Optional): a list of integer that represents the DTensor have pending sum on which device mesh dimension

    Returns:
        List[Placement]: a list contains some `paddle.distributed.Placement`.
    """
    placements = [Replicate() for _ in mesh.shape]

    for s in partial_idx:
        placements[s] = Partial()

    for tensor_dim, mesh_dims in enumerate(dim_map):
        if len(mesh_dims) <= 0:
            continue
        is_co_shard = len(mesh_dims) > 1
        for shard_order, mesh_dim in enumerate(mesh_dims):
            p = placements[mesh_dim]
            if p.is_shard():
                raise Exception(
                    f"ProcessMesh dimension can not be mapped to two dimension of same tensor: {tensor_dim} and {p.get_dim()}."
                )
            elif p.is_partial():
                raise Exception(
                    f"ProcessMesh dimension {mesh_dim} can not be both shard and partial!"
                )

            shard = (
                Shard(tensor_dim, co_shard_order=shard_order)
                if is_co_shard
                else Shard(tensor_dim)
            )
            placements[mesh_dim] = shard

    if len(split_factor) > 1:
        raise RuntimeError("At now only support to rearrange at one mesh dim.")

    for k, v in split_factor.items():
        placements[k].set_split_factor(v)

    return placements


def to_placements(dim_map, mesh, partial_idx=[]):
    """
    convert dim_map to placements.
    Args:
        dim_map(List[int]): a list of integer that represents sharding on each tensor dimension.
        mesh(paddle.distributed.ProcessMesh): The `ProcessMesh` object describes the Cartesian topology of the used processes.
        partial_idx(List[int], Optional): a list of integer that represents the DTensor have pending sum on which device mesh dimension
    Returns:
        List[Placement]: a list contains some `paddle.distributed.Placement`.
    """
    if isinstance(mesh, paddle.base.libpaddle.ProcessMesh):
        shape = mesh.shape
    else:
        shape = mesh.mesh.shape
    placements = [Replicate() for _ in range(len(shape))]

    for s in partial_idx:
        placements[s] = Partial()

    for i, m in enumerate(dim_map):
        if m >= 0:
            p = placements[m]
            if p.is_shard():
                p = cast("Shard", p)
                raise Exception(
                    f"ProcessMesh dimension can not be mapped to two dimension of same tensor: {i} and {p.get_dim()}."
                )
            elif p.is_partial():
                raise Exception(
                    f"ProcessMesh dimension {m} can not be both shard and partial!"
                )
            placements[m] = Shard(i)

    return placements


def check_placements_equal(this, that):
    assert isinstance(this, list) and isinstance(that, list)
    small_placements = this if len(this) < len(that) else that
    large_placements = that if len(this) < len(that) else this
    for i in range(len(large_placements)):
        if i < len(small_placements):
            if small_placements[i] != large_placements[i]:
                return False
        else:
            if large_placements[i] != Replicate():
                return False
    return True


def placemetns_to_dist_status(
    placements, tensor_dims, return_split_factor=False
):
    """
    convert placements to dim_map.

    Args:
        placements(List[Placement]): a list contains some `paddle.distributed.Placement`.
        tensor_dims(int): the dimension of dist_tensor.

    Returns:
        List[int]: a list of integer that represents sharding on each tensor dimension.
    """
    output_list = []
    dim_map = [[] for _ in range(tensor_dims)]
    partial_status = {}
    split_factor_map = {}
    for i, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = cast("Shard", placement).get_dim()
            dim_map[shard_dim].append(i)
            if cast("Shard", placement).get_split_factor() > 1:
                split_factor_map[i] = cast(
                    "Shard", placement
                ).get_split_factor()
                assert (
                    len(split_factor_map) == 1
                ), "only support to rerrange at one mesh dim."
        if placement.is_partial():
            partial_status[i] = cast("Partial", placement).reduce_type()

    for shard_dim in dim_map:
        if len(shard_dim) > 0:
            shard_dim.sort(key=lambda idx: placements[idx].get_co_shard_order())

    output_list.append(dim_map)
    output_list.append(partial_status)

    if return_split_factor:
        output_list.append(split_factor_map)

    return output_list


def to_dim_map(placements, tensor_dims):
    """
    convert placements to dim_map.

    Args:
        placements(List[Placement]): a list contains some `paddle.distributed.Placement`.
        tensor_dims(int): the dimension of dist_tensor.

    Returns:
        List[int]: a list of integer that represents sharding on each tensor dimension.
    """
    dim_map = [-1] * tensor_dims
    partial_status = {}
    for i, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = cast("Shard", placement).get_dim()
            if dim_map[shard_dim] > -1:
                import logging

                logging.warning(
                    f"Tensor dim {shard_dim} is already sharded on mesh dim {dim_map[shard_dim]}."
                )

            dim_map[shard_dim] = i
        if placement.is_partial():
            partial_status[i] = cast("Partial", placement).reduce_type()

    return dim_map, partial_status


# TODO(lfw): delete it in future.
def get_shard_spec(mesh, placements, tensor_dims):
    """to get shard_spec for construct DistAttr for static API."""
    dim_map = [-1] * tensor_dims
    for i, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = cast("Shard", placement).get_dim()
            if dim_map[shard_dim] > -1:
                import logging

                logging.warning(
                    f"Tensor dim {shard_dim} is already sharded on mesh dim {dim_map[shard_dim]}."
                )

            dim_map[shard_dim] = i

    mesh_dim_names = mesh.dim_names
    shard_spec = [None] * len(dim_map)
    for i, d in enumerate(dim_map):
        if d > -1:
            shard_spec[i] = mesh_dim_names[d]

    return shard_spec
