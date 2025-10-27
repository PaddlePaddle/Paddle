# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from paddle import Tensor
    from paddle.distributed.communication.group import Group


@dataclass(frozen=True)
class ShardedWeightDesc:
    key: str
    local_shape: tuple[int, ...]
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    dtype: str | None = None


class ShardedWeight:
    """
    Represents a local shard of a distributed tensor parameter.

    Args:
        key (str): The name of the parameter.
        local_tensor (Tensor): The local shard of the parameter.
        local_shape (Tuple[int, ...]): The shape of the local shard.
        global_shape (Tuple[int, ...]): The global logical shape of the parameter.
        global_offset (Tuple[int, ...]): The offset of the local shard in the global parameter.
        is_flattened (bool, optional): Whether the parameter has been flattened (used in sharding_v2 scenarios). Default is False.
        flattened_range (slice, optional): If the parameter is flattened, this indicates the index range of the actual local shard within the local_tensor.
    """

    def __init__(
        self,
        key: str,
        local_tensor: Tensor,
        local_shape: tuple[int, ...],
        global_shape: tuple[int, ...],
        global_offset: tuple[int, ...],
        is_flattened: bool = False,
        flattened_range: slice | None = None,
    ) -> None:
        self.key = key
        self.local_tensor = local_tensor
        self.local_shape = local_shape
        self.global_shape = global_shape
        self.global_offset = global_offset
        self.is_flattened = is_flattened
        self.flattened_range = flattened_range

    def __str__(self) -> str:
        """Returns a formatted string representation of the sharded tensor."""
        return (
            f"ShardedWeight(\n"
            f"  key={self.key},\n"
            f"  local_tensor={type(self.local_tensor).__name__}(shape={self.local_tensor.shape}),\n"
            f"  local_shape={self.local_shape},\n"
            f"  global_shape={self.global_shape},\n"
            f"  global_offset={self.global_offset},\n"
            f"  flattened_range={self.flattened_range}\n"
            f")"
        )


ShardedStateDict = Union[
    dict[str, ShardedWeight], OrderedDict[str, ShardedWeight]
]


def shard_weight(
    key: str,
    weight: Tensor,
    axis: int,
    group: Group,
) -> ShardedWeight:
    """Creates a ShardedWeight by splitting the input tensor along a specified axis.

    Args:
        key: Unique identifier for the tensor.
        weight: The input tensor to be sharded.
        axis: The axis along which to shard the tensor.
        group: The process group used for distributed communication.

    Returns:
        A ShardedWeight representing the local portion of the global tensor.
    """
    if axis < 0 or axis >= len(weight.shape):
        raise ValueError(
            f"Shard axis {axis} is invalid for tensor with shape {weight.shape}"
        )

    # Get hybrid communication group and rank information
    current_rank = group.rank
    world_size = group.nranks

    # Calculate shapes and offsets
    local_shape = weight.shape
    global_shape = deepcopy(local_shape)
    global_shape[axis] = local_shape[axis] * world_size
    global_shape = tuple(global_shape)
    local_shape = tuple(local_shape)
    global_offset = [0] * len(global_shape)
    if world_size > 1:
        global_offset[axis] = current_rank * local_shape[axis]
    global_offset = tuple(global_offset)

    return ShardedWeight(
        key=key,
        local_tensor=weight,
        local_shape=local_shape,
        global_shape=global_shape,
        global_offset=global_offset,
    )


def make_tp_sharded_weight_for_checkpoint(
    key: str,
    tensor: Tensor,
    tensor_parallel_axis: int = 0,
) -> ShardedWeight:
    """Creates a tensor-parallel sharded tensor for checkpointing purposes.

    Args:
        key: Unique identifier for the tensor in the checkpoint.
        tensor: The local tensor portion to be sharded.
        tensor_parallel_axis: The axis along which tensor parallelism is applied.
                            Defaults to 0 (first dimension).

    Returns:
        A ShardedWeight configured for tensor parallel checkpointing.
    """
    from paddle.distributed.fleet import get_hybrid_communicate_group

    hcg = get_hybrid_communicate_group()
    tensor_parallel_group = hcg.get_model_parallel_group()

    return shard_weight(
        key=key,
        weight=tensor,
        axis=tensor_parallel_axis,
        group=tensor_parallel_group,
    )


def make_replicated_sharded_weight(
    key: str,
    tensor: Tensor,
) -> ShardedWeight:
    """
    Creates a ShardedWeight that represents a fully replicated tensor (each process holds a full copy).

    Args:
        key: Unique identifier for the tensor in the checkpoint.
        tensor: The local tensor (full copy).

    Returns:
        ShardedWeight: A ShardedWeight instance representing the replicated tensor.
    """
    zero_offset = tuple(0 for _ in tensor.shape)
    return ShardedWeight(
        key=key,
        local_tensor=tensor,
        local_shape=tensor.shape,
        global_shape=tensor.shape,
        global_offset=zero_offset,
    )


def build_sharded_state_dict(
    state_dict: dict[str, Tensor],
    shard_rules: dict[str, int] | None = None,
    prefix: str = "",
) -> dict[str, ShardedWeight]:
    """Converts a regular state dict to a sharded state dict based on sharding rules.

    Args:
        state_dict: The original state dictionary containing tensors
        shard_rules: Dictionary mapping tensor names to their sharding axes.
                    If None, treated as empty dict (no tensor parallelism).
        prefix: Optional prefix to prepend to all tensor keys

    Returns:
        Dictionary with the same keys as input but values converted to ShardedWeight
        or regular Tensor based on sharding rules.

    Note:
        Tensors not in shard_rules will be wrapped as non-sharded ShardedWeights.
    """
    shard_rules = shard_rules or {}
    sharded_state_dict = {}

    for key, tensor in state_dict.items():
        full_key = f"{prefix}{key}" if prefix else key

        if key in shard_rules:
            # Apply tensor parallelism sharding
            sharded_state_dict[full_key] = (
                make_tp_sharded_weight_for_checkpoint(
                    key=full_key,
                    tensor=tensor,
                    tensor_parallel_axis=shard_rules[key],
                )
            )
        else:
            # Create regular sharded tensor (non-tensor-parallel)
            sharded_state_dict[full_key] = make_replicated_sharded_weight(
                key=full_key,
                tensor=tensor,
            )

    return sharded_state_dict


def create_sharded_weight_with_new_local(
    new_key: str,
    new_local_tensor: Tensor,
    reference_tensor: ShardedWeight,
) -> ShardedWeight:
    """
    Creates a new ShardedWeight with a new local tensor while preserving the metadata from a reference ShardedWeight.

    Args:
        new_key (str): The new key for the ShardedWeight.
        new_local_tensor (Tensor): The new local tensor to use (must match reference_tensor.local_shape).
        reference_tensor (ShardedWeight): The reference ShardedWeight to copy metadata from.

    Returns:
        ShardedWeight: A new ShardedWeight with the new local tensor and copied metadata.

    """
    # Copy metadata from the reference tensor
    global_shape = deepcopy(reference_tensor.global_shape)
    local_shape = deepcopy(reference_tensor.local_shape)
    global_offset = deepcopy(reference_tensor.global_offset)

    # Input validation: Check if new_local_tensor's shape matches local_shape
    if tuple(new_local_tensor.shape) != tuple(local_shape):
        raise ValueError(
            f"Shape mismatch: new_local_tensor has shape {new_local_tensor.shape}, "
            f"but expected shape {local_shape} (from reference_tensor.local_shape)."
        )

    return ShardedWeight(
        key=new_key,
        local_tensor=new_local_tensor,
        local_shape=tuple(local_shape),
        global_shape=tuple(global_shape),
        global_offset=tuple(global_offset),
    )
