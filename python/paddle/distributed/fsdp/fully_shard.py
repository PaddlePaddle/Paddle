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

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import paddle
    import paddle.distributed as dist
import paddle
from paddle.distributed.auto_parallel.fully_shard import FullyShardAuto
from paddle.distributed.fleet.meta_parallel.meta_parallel_base import (
    MetaParallelBase,
)

from .fully_shard_fusion import FullyShardFusion


def in_auto_parallel_mode() -> bool:
    return getattr(
        paddle.base.framework.global_var, '_in_auto_parallel_', False
    )


@dataclass
class MixedPrecisionPolicy:
    param_dtype: paddle.dtype | None = None
    reduce_dtype: paddle.dtype | None = None
    output_dtype: paddle.dtype | None = None
    cast_forward_inputs: bool = True


# @dataclass
class OffloadPolicy:
    pin_memory: bool = True


def _mark_params_for_optimizer(module, optimizer):
    decay_fun = getattr(optimizer, "_apply_decay_param_fun", None)
    muon_info_map = getattr(optimizer, "_muon_param_info_map", None)
    missing = []
    for name, param in module.named_parameters():
        param.no_weight_decay = decay_fun is not None and not decay_fun(
            param.name
        )
        if muon_info_map is None:
            continue
        info = muon_info_map.get(param.name)
        if info is None and not param.stop_gradient:
            missing.append(name)
        hyperball = info is not None and getattr(info, "use_hyperball", False)
        # AdamW+Hyperball runs on shards with all-reduced norms.
        shape = getattr(param, "original_shape", None) or param.shape
        param.use_hyperball_sharded = bool(
            hyperball
            and not info.use_muon
            and getattr(info, "split_concat_func", None) is None
            and len(shape) == 2
        )
        param.use_muon = bool(
            info is not None
            and (info.use_muon or hyperball)
            and not param.use_hyperball_sharded
        )
    if missing:
        raise ValueError(
            f"muon_param_info_map misses trainable params: {missing[:8]}"
        )


def _fully_shard_manual_parallel(
    module,
    mesh,
    reshard_after_forward,
    shard_placement_fn,
    mp_policy,
    offload_policy,
    ignored_params,
    enable_tensor_fusion_and_overlap,
    optimizer=None,
    grad_accum_steps=1,
    prefetch_units=1,
    prefetch_sizes=None,
):
    layers = module._layers if isinstance(module, MetaParallelBase) else module
    if optimizer is not None:
        _mark_params_for_optimizer(layers, optimizer)
    layers._fsdp_context = FullyShardFusion(
        layers,
        enable_tensor_fusion_and_overlap=enable_tensor_fusion_and_overlap,
        mp_policy=mp_policy,
        prefetch_units=prefetch_units,
        prefetch_sizes=prefetch_sizes,
    )
    layers._fsdp_context.set_grad_accum_steps(grad_accum_steps)
    if optimizer is not None:
        layers._fsdp_context.bind_optimizer(optimizer)
    module._fsdp_context = layers._fsdp_context
    return module


def _fully_shard_auto_parallel(
    module,
    mesh,
    reshard_after_forward,
    shard_placement_fn,
    mp_policy,
    offload_policy,
    ignored_params,
    enable_tensor_fusion_and_overlap,
    optimizer=None,
    grad_accum_steps=1,
    prefetch_units=1,
    prefetch_sizes=None,
):
    module._fsdp_context = FullyShardAuto(
        module,
        mesh,
        enable_tensor_fusion_and_overlap=enable_tensor_fusion_and_overlap,
    )
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
    enable_tensor_fusion_and_overlap: bool = True,
    optimizer: paddle.optimizer.Optimizer | None = None,
    grad_accum_steps: int = 1,
    prefetch_units: int = 1,
    prefetch_sizes: int | None = None,
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
        enable_tensor_fusion_and_overlap (bool, optional): Whether to enable tensor fusion and
            communication/computation overlap optimization. When True, parameters are fused into
            contiguous buffers for more efficient communication and memory access, and enables
            prefetching and asynchronous communication to overlap with computation for better
            performance. Defaults to True.
        optimizer (Optimizer | None, optional): Manual parallel mode only. The optimizer
            built on ``module``'s parameters. Defaults to None, in which case
            ``param.no_weight_decay`` and ``param.use_muon`` must be set beforehand
            if needed.
        grad_accum_steps (int, optional): Manual parallel mode only. Gradient
            accumulation steps. Defaults to 1.
        prefetch_units (int, optional): Manual parallel mode only. Number of
            FSDP units all_gathered ahead (prefetch by layer). 0 disables
            prefetch. Ignored when ``prefetch_sizes`` is set. Defaults to 1.
        prefetch_sizes (int | None, optional): Manual parallel mode only. If
            set, prefetch by size: params are all_gathered
            ``prefetch_sizes // 2`` elements ahead and at most
            ``prefetch_sizes`` grad elements stay in reduce_scatter. 0
            disables prefetch. Defaults to None (prefetch by layer).

    Returns:
        module: A wrapper module that applies FSDP to the input module.

    Examples:
        .. code-block:: pycon

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
        enable_tensor_fusion_and_overlap,
        optimizer,
        grad_accum_steps,
        prefetch_units,
        prefetch_sizes,
    )

    if in_auto_parallel_mode():
        return _fully_shard_auto_parallel(*args)
    else:
        return _fully_shard_manual_parallel(*args)
