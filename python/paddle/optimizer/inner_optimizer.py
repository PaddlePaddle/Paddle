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

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
from paddle.base import framework

from ..nn.clip import GradientClipBase
from .optimizer import Optimizer

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle import Tensor

__all__ = []

# Mirror muon.py: when this flag is on the whole dygraph optimizer update is
# skipped (used by sharding to bypass the local step on non-owner ranks).
g_shard_bypass_dygraph_optimizer = int(
    os.environ.get("FLAGS_shard_bypass_dygraph_optimizer", 0)
)


@dataclass
class ParamInfo:
    """Generic per-parameter metadata read by the sharding shell.

    This is the neutral data carrier of the shell <-> inner-optimizer contract.
    The sharding shell (e.g. PaddleFleet's ``MixedShardingOptimizer``)
    interprets exactly one field, ``keep_whole``, to decide V1 (whole-tensor)
    vs V2 (element-wise) partitioning. Every other concern (how to update the
    parameter) belongs to the inner optimizer, which may attach its own metadata
    either by subclassing this dataclass or by setting extra attributes on an
    instance -- neither of which requires changing the sharding shell or paddle
    core.

    Attributes:
        keep_whole: If True the shell keeps the whole tensor on one rank (V1);
            if False it shards element-wise (V2). ``None`` means unset; a shell
            that relies on this field (e.g. ``MixedShardingOptimizer``)
            requires it to be set explicitly.
    """

    keep_whole: bool | None = None


class InnerOptimizer(Optimizer):
    r"""Distribution-agnostic base class for extensible optimizers.

    ``InnerOptimizer`` factors out the machinery shared by optimizers that are
    meant to run *inside* a sharding shell (Sharding Stage1 V3 style, e.g.
    PaddleFleet's ``MixedShardingOptimizer``).  The shell owns all distributed
    logic (reduce /
    reduce-scatter, whole-tensor vs element-wise partition, all-gather /
    broadcast, comm buffers) and, after producing the *local* ``params_grads``
    (either a whole tensor on the owner rank, or a shard), delegates the actual
    per-parameter update math to the inner optimizer's ``_apply_optimize``.

    A subclass therefore only needs to implement the local update math and its
    accumulator schema; it never touches communication or partitioning.  This
    mirrors Megatron-LM's ``DistributedOptimizer`` (shell) + inner
    ``torch.optim`` split, and lets new optimizers be added at the framework
    layer without modifying the sharding shell.

    Contract expected by the sharding shell:
        * ``_apply_optimize(loss, startup_program, params_grads)`` -- update the
          local (whole-or-sharded) parameters in place.
        * ``_param_info_map`` -- generic per-parameter metadata the shell reads
          to decide V1 (whole tensor) vs V2 (element-wise) partitioning. The
          only field the generic shell needs is ``keep_whole``; any optimizer-
          specific fields (e.g. Muon's ``use_muon`` / ``use_hyperball`` /
          ``split_concat_func``) are interpreted only by that optimizer, not by
          this base class or the shell. The base class never defines Muon's
          legacy ``_muon_param_info_map`` name.
        * ``use_fusion_storage`` / ``_gen_master_weight_var_name`` /
          ``state_dict`` / ``set_state_dict`` -- inherited from ``Optimizer``.

    Subclasses MUST implement:
        * :meth:`_ensure_accumulators` -- allocate the accumulators a parameter
          needs (called once per parameter before the first update). What state
          a parameter needs is decided by the optimizer class itself.
        * :meth:`_update_parameters` -- the local update math for a batch of
          ``(param, grad)`` pairs whose accumulators already exist.

    Args:
        learning_rate (float | LRScheduler): Learning rate. Default: ``0.001``.
        parameters (list[Tensor]): Flat list of parameters (no param groups).
        weight_decay (float): Decoupled weight decay magnitude. Default: ``0.0``.
        grad_clip (GradientClipBase | None): Gradient clipping. Default: ``None``.
        lr_ratio (Callable[[Tensor], float] | None): Optional per-parameter
            multiplier applied to the learning rate. Default: ``None``.
        apply_decay_param_fun (Callable[[str], bool] | None): Selects which
            parameters receive weight decay. Default: ``None``.
        multi_precision (bool): Keep FP32 master weights under BF16/FP16
            training. Default: ``False``.
        param_info_map (dict | None): Maps ``param.name`` to per-parameter
            metadata (must expose ``keep_whole`` for the shell; optimizer-
            specific fields are optional). Default: ``None``.
        name (str | None): Optional optimizer name.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        parameters: list[Tensor] | None = None,
        weight_decay: float = 0.0,
        grad_clip: GradientClipBase | None = None,
        lr_ratio: Callable[[Tensor], float] | None = None,
        apply_decay_param_fun: Callable[[str], bool] | None = None,
        multi_precision: bool = False,
        param_info_map: dict | None = None,
        name: str | None = None,
        **kwargs,
    ) -> None:
        if parameters is None:
            raise ValueError(
                "parameters argument given to the Optimizer should not be None."
            )
        if not isinstance(parameters, list):
            raise TypeError("parameters must be a list.")
        if len(parameters) > 0 and isinstance(parameters[0], dict):
            raise TypeError(
                "InnerOptimizer only supports a flat list of parameters, "
                "not a list of parameter groups."
            )
        if grad_clip is not None and not isinstance(
            grad_clip, GradientClipBase
        ):
            raise TypeError(
                "'grad_clip' should be an instance of GradientClipBase's "
                "derived class"
            )

        super().__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name,
        )

        self._multi_precision = multi_precision
        self._master_weights = {}
        self._lr_ratio = lr_ratio
        self._apply_decay_param_fun = apply_decay_param_fun
        # Generic per-parameter metadata. The generic base and the shell only
        # rely on the neutral ``keep_whole`` field (V1/V2 sharding); any
        # optimizer-specific fields are interpreted only by that optimizer. It
        # deliberately does NOT define ``_muon_param_info_map`` -- that is
        # ``paddle.optimizer.Muon``'s private/legacy name and must not leak onto
        # every InnerOptimizer subclass. The sharding shell reads
        # ``_param_info_map`` generically.
        self._param_info_map = param_info_map or {}

    # ------------------------------------------------------------------
    # Hooks a subclass must implement
    # ------------------------------------------------------------------

    def _ensure_accumulators(self, param):
        """Allocate the accumulators *param* needs, if not created yet.

        Called once per parameter (via :meth:`_apply_optimize` /
        :meth:`_create_accumulators`) before its first update. Implementations
        typically call ``self._add_accumulator`` for each state tensor (e.g.
        moment1 / moment2 / beta_pow) and, under mixed precision,
        ``self._ensure_master_weight(param)``. What state a parameter needs is
        decided by the optimizer class itself (its own hyperparameters), not by
        any Muon-specific flag.
        """
        raise NotImplementedError(
            "Subclasses of InnerOptimizer must implement _ensure_accumulators()."
        )

    def _update_parameters(self, params_grads, lr):
        """Apply the local update math to a batch of ``(param, grad)`` pairs.

        Accumulators for every parameter here already exist. ``lr`` is the
        scalar learning rate for the current step (LRScheduler already
        resolved). Hyperparameters live on the instance (frozen at construction);
        implementations should use :meth:`_effective_lr` to honour ``lr_ratio``.
        """
        raise NotImplementedError(
            "Subclasses of InnerOptimizer must implement _update_parameters()."
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _effective_lr(self, param, lr):
        """Learning rate for *param* after applying the optional ``lr_ratio``."""
        if self._lr_ratio is not None:
            return lr * self._lr_ratio(param)
        return lr

    def _ensure_master_weight(self, param):
        """Create an FP32 master weight for *param* under mixed precision."""
        if self._multi_precision and self._is_dtype_fp16_or_bf16(param.dtype):
            if param.name not in self._master_weights:
                self._create_master_weight(param)

    def _create_accumulators(self, block, parameters):
        """Create checkpoint load targets without executing an optimizer step.

        Flex checkpoint loads into the existing accumulator tensors, so the
        same per-parameter allocation hook used by the update path must run
        before the distributed state loader constructs its target mapping.
        """
        if not framework.in_dygraph_mode():
            raise NotImplementedError(
                "InnerOptimizer only supports dygraph mode."
            )
        for param in parameters:
            self._ensure_accumulators(param)

    # ------------------------------------------------------------------
    # Optimizer-state layout (universal checkpoint contract)
    # ------------------------------------------------------------------

    def optimizer_state_layout(self) -> dict:
        """Describe this optimizer's checkpoint state, generically.

        Returns a dict with three entries used by every checkpoint/reshard/ZCC
        code path INSTEAD of hard-coding AdamW/Muon suffixes:

            {
                "vector": [<suffix>, ...],   # per-element tensors, sharded like
                                             # the parameter (e.g. moment1_0)
                "scalar": [<suffix>, ...],   # replicated scalars (e.g. beta1_pow_acc_0)
                "master": <suffix>,          # fp32 master-weight suffix
            }

        Suffixes are the trailing token of the accumulator var name
        (``<param_static_name>_<suffix>``). The framework-layer checkpoint code
        (ernie5 ``sharded_state_dict``, PaddleFleet resume gating, ZCC name
        generation) reads this so an optimizer with a different state layout only
        has to override this one method. The default matches the AdamW /
        Muon(+Hyperball) layout.
        """
        return {
            "vector": ["moment1_0", "moment2_0"],
            "scalar": ["beta1_pow_acc_0", "beta2_pow_acc_0"],
            "master": "fp32_master_0",
        }

    # ------------------------------------------------------------------
    # Update entry points
    # ------------------------------------------------------------------

    def _apply_optimize(self, loss, startup_program, params_grads):
        """Template method: ensure accumulators, then delegate the update math.

        Called by the sharding shell with the *local* (whole-or-sharded)
        ``params_grads`` after it has done all collective communication.
        """
        if not framework.in_dygraph_mode():
            raise NotImplementedError(
                "InnerOptimizer only supports dygraph mode."
            )

        # Same bypass as Optimizer._apply_optimize: skip all parameter updates
        # (e.g. on non-owner ranks under sharding).
        if g_shard_bypass_dygraph_optimizer:
            return

        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)

        # apply for zcc
        self._maybe_refuse()

        lr = self._learning_rate
        if isinstance(lr, paddle.optimizer.lr.LRScheduler):
            lr = lr()

        active_params_grads = []
        for param, grad in params_grads:
            if grad is None:
                continue
            self._ensure_accumulators(param)
            active_params_grads.append((param, grad))

        self._update_parameters(active_params_grads, lr)

    @framework.dygraph_only
    def step(self) -> None:
        params_grads = [
            (param, param._grad_ivar())
            for param in self._parameter_list
            if not param.stop_gradient and param._grad_ivar() is not None
        ]
        self._apply_optimize(
            loss=None, startup_program=None, params_grads=params_grads
        )
