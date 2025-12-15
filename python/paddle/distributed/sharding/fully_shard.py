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

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from collections.abc import Iterable

    import paddle
    import paddle.distributed as dist
import paddle
from paddle.distributed.auto_parallel.fully_shard import FullyShardAuto
from paddle.distributed.fleet.meta_parallel.sharding.group_sharded_fully_shard import (
    FullyShard,
)


def in_auto_parallel_mode() -> bool:
    return getattr(
        paddle.base.framework.global_var, '_in_auto_parallel_', False
    )


# @dataclass
class MixedPrecisionPolicy:
    param_dtype: paddle.dtype | None = None
    reduce_dtype: paddle.dtype | None = None
    output_dtype: paddle.dtype | None = None
    cast_forward_inputs: bool = True


# @dataclass
class OffloadPolicy:
    pin_memory: bool = True


def _fully_shard_manual_parallel(
    module,
    mesh,
    reshard_after_forward,
    shard_placement_fn,
    mp_policy,
    offload_policy,
    ignored_params,
):
    return FullyShard(module)


def _fully_shard_auto_parallel(
    module,
    mesh,
    reshard_after_forward,
    shard_placement_fn,
    mp_policy,
    offload_policy,
    ignored_params,
):
    FullyShardAuto(module, mesh)


def fully_shard(
    module: paddle.nn.Layer,
    *,
    mesh: dist.ProcessMesh = None,
    reshard_after_forward: bool | int | None = None,
    shard_placement_fn: Callable[[paddle.Tensor], dist.Shard | None]
    | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    ignored_params: Iterable[paddle.Tensor] | None = None,
) -> paddle.nn.Layer:
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()
    if offload_policy is None:
        offload_policy = OffloadPolicy()
    ignored_params_set: set[paddle.Tensor] = (
        set(ignored_params) if ignored_params else set()
    )

    args = (
        module,
        mesh,
        reshard_after_forward,
        shard_placement_fn,
        mp_policy,
        offload_policy,
        ignored_params_set,
    )

    if in_auto_parallel_mode():
        return _fully_shard_auto_parallel(*args)
    else:
        return _fully_shard_manual_parallel(*args)
