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
    return module


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
    """
    Apply fully sharded data parallel (FSDP) to the given module.

    This function wraps the input module with fully sharded data parallelism, which shards
    model parameters, gradients, and optimizer states across multiple devices. It supports
    both auto_parallel mode and manual_parallel mode.

    Args:
        module (Layer): The neural network module to be wrapped with fully sharded data parallelism.
        mesh (dist.ProcessMesh, optional): The process mesh defining the device arrangement for sharding.
            Defaults to None, which uses the default mesh.
        reshard_after_forward (bool | int | None, optional): Controls when to reshard the parameters after forward pass.
            If True or 1, reshard after each forward pass. If False or 0, keep sharded.
            If None, use default strategy. Defaults to None.
        shard_placement_fn (Callable[[paddle.Tensor], dist.Shard | None] | None, optional):
            A function that determines how each tensor should be sharded. Takes a tensor as input
            and returns a Shard placement or None. If None, uses default sharding strategy.
            Defaults to None.
        mp_policy (MixedPrecisionPolicy | None, optional): Mixed precision policy configuration.
            If None, creates a default MixedPrecisionPolicy. Defaults to None.
        offload_policy (OffloadPolicy | None, optional): Offload policy configuration for CPU offloading.
            If None, creates a default OffloadPolicy. Defaults to None.
        ignored_params (Iterable[paddle.Tensor] | None, optional): Parameters that should not be sharded.
            These parameters will be kept in full precision and not distributed. Defaults to None.

    Returns:
        module: A wrapper module that applies FSDP to the input module.

    Examples:
        .. code-block:: python

            >>> # type: ignore
            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> # python -m paddle.distributed.launch --device=0,1 train.py
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle.distributed.fsdp import fully_shard

            >>> mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
            >>> model = paddle.nn.Linear(10, 10)
            >>> inputs = paddle.rand(shape=[10, 10])
            >>> inputs = dist.shard_tensor(inputs, mesh, [dist.Shard(0)])
            >>> opt = paddle.optimizer.AdamW(parameters=model.parameters())
            >>> model = fully_shard(model, mesh)
            >>> tr_loss = model(inputs)
            >>> tr_loss.backward()
            >>> opt.step()
            >>> opt.clear_grad()
    """
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
