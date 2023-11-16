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

from paddle.base import core


class ReduceType:
    kRedSum = 0
    kRedMax = 1
    kRedMin = 2
    kRedProd = 3
    kRedAvg = 4
    kRedAny = 5
    kRedAll = 6


class Placement(core.Placement):
    # Placement (base class)
    def __init__(self):
        super().__init__()

    def is_replicated(self):
        return isinstance(self, core.Replicate)

    def is_shard(self, dim):
        if dim is not None and isinstance(self, core.Shard):
            return self.dim == dim
        else:
            return isinstance(self, core.Shard)

    def is_partial(self):
        return isinstance(self, core.Partial)

    def __repr__(self):
        return "Placement()"


class Replicate(core.Replicate):
    # Replicate placement
    def __init__(self):
        super().__init__()

    def __eq__(self, other):
        if not isinstance(other, core.Replicate):
            return False
        return True

    def __repr__(self):
        return "Replicate()"


class Shard(core.Shard):
    # Shard placement
    def __init__(self, dim):
        super().__init__(dim)
        self.dim = dim

    def get_dim(self):
        return self.dim

    def __eq__(self, other):
        if not isinstance(other, core.Shard):
            return False
        return self.dim == other.dim

    def __repr__(self):
        return f"Shard({self.dim})"


class Partial(core.Partial):
    # Partial placement
    def __init__(self, reduce_type=None):
        if reduce_type is None:
            self.reduce_type = core.ReduceType.kRedSum
        elif reduce_type == ReduceType.kRedSum:
            self.reduce_type = core.ReduceType.kRedSum
        elif reduce_type == ReduceType.ReduceType.kRedMax:
            self.reduce_type = core.ReduceType.kRedMax
        elif reduce_type == ReduceType.kRedMin:
            self.reduce_type = core.ReduceType.kRedMin
        elif reduce_type == ReduceType.kRedProd:
            self.reduce_type = core.ReduceType.kRedProd
        elif reduce_type == ReduceType.kRedAvg:
            self.reduce_type = core.ReduceType.kRedAvg
        elif reduce_type == ReduceType.kRedAny:
            self.reduce_type = core.ReduceType.kRedAny
        elif reduce_type == ReduceType.kRedAll:
            self.reduce_type = core.ReduceType.kRedAll
        else:
            raise Exception(
                "reduce_type is error! it should be dist.ReduceType."
            )
        super().__init__(self.reduce_type)

    def __eq__(self, other):
        if not isinstance(self, core.Partial):
            return False
        return self.reduce_type == other.reduce_type

    def __repr__(self):
        return f"Partial{self.reduce_type}"


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
    placements = [Replicate() for _ in range(len(mesh.mesh.shape))]

    for s in partial_idx:
        placements[s] = Partial()

    for i, m in enumerate(dim_map):
        if m >= 0:
            p = placements[m]
            if p.is_shard():
                p = cast(Shard, p)
                raise Exception(
                    f"ProcessMesh dimension can not be mapped to two dimension of same tensor: {i} and {p.get_dim()}."
                )
            elif p.is_partial():
                raise Exception(
                    f"ProcessMesh dimension {m} can not be both shard and partial!"
                )
            placements[m] = Shard(i)

    return placements


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
    for i, placement in enumerate(placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).get_dim()
            if dim_map[shard_dim] > -1:
                raise Exception(
                    "Tensor dim {shard_dim} is already sharded on mesh dim {dim_map[shard_dim]}"
                )

            dim_map[shard_dim] = i

    return dim_map


def get_shard_spec(mesh, placements, tensor_dims):
    """to get shard_spec for construct DistAttr for static API."""
    dim_map = to_dim_map(placements, tensor_dims)
    mesh_dim_names = mesh.dim_names
    shard_spec = [None] * len(dim_map)
    for i, d in enumerate(dim_map):
        if d > -1:
            shard_spec[i] = mesh_dim_names[d]

    return shard_spec
