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
from abc import ABCMeta, abstractmethod
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

# Sentinel for ``InnerOptimizer.keep_whole``: the subclass has not declared its
# sharding support yet. It is deliberately NOT ``None`` -- ``None`` is a
# meaningful declaration ("both V1 and V2 work, let the config pick"), so a
# subclass that simply forgot to declare must be distinguishable from one that
# declared "either".
KEEP_WHOLE_UNDECLARED = "undeclared"


@dataclass
class ParamInfo:
    """Generic per-parameter metadata of the shell <-> inner-optimizer contract.

    It deliberately carries no field of its own: V1 (whole-tensor) vs V2
    (element-wise) partitioning is a property of the *optimizer class*
    (:attr:`InnerOptimizer.keep_whole`, possibly narrowed per group by the
    config) rather than of an individual parameter, so the sharding shell asks
    the inner optimizer that owns a parameter instead of reading a
    per-parameter flag. Everything else (how to update the
    parameter) likewise belongs to the inner optimizer, which may attach its own
    metadata either by subclassing this dataclass (e.g.
    :class:`~paddle.optimizer.muon.MuonParamInfo`) or by setting extra
    attributes on an instance -- neither of which requires changing the sharding
    shell or paddle core.
    """


class InnerOptimizer(Optimizer, metaclass=ABCMeta):
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
        * :attr:`keep_whole` -- class-level declaration of which sharding modes
          this optimizer's update math supports: whole tensor (V1), element-wise
          (V2), or either; the shell reads the resolved per-instance answer
          through :meth:`keep_whole_of`.
        * ``_param_info_map`` -- optional per-parameter metadata. The generic
          shell does not interpret it at all; optimizer-specific fields (e.g.
          Muon's ``use_muon`` / ``use_hyperball`` / ``split_concat_func``) are
          read only by that optimizer. The base class never defines Muon's
          legacy ``_muon_param_info_map`` name.
        * ``use_fusion_storage`` / ``_gen_master_weight_var_name`` /
          ``state_dict`` / ``set_state_dict`` -- inherited from ``Optimizer``.

    Subclasses MUST implement (the three methods are ``@abstractmethod``, so a
    subclass that misses one cannot be instantiated; ``keep_whole`` is a class
    attribute and is checked the same way, at construction):
        * :attr:`keep_whole` -- ``True`` (only V1), ``False`` (only V2) or
          ``None`` (either; the config picks, default V2).
        * :meth:`_ensure_accumulators` -- allocate the accumulators a parameter
          needs (called once per parameter before the first update). What state
          a parameter needs is decided by the optimizer class itself.
        * :meth:`_update_parameters` -- the local update math for a batch of
          ``(param, grad)`` pairs whose accumulators already exist.
        * :meth:`optimizer_state_layout` -- name the accumulators created above
          and say which of them are sharded with the parameter. There is no
          default: it must match :meth:`_ensure_accumulators`, which only the
          subclass knows, and every checkpoint path parses state names through
          it.

    Args:
        learning_rate (float | LRScheduler): Learning rate. Default: ``0.001``.
        parameters (list[Tensor]): Flat list of parameters (no param groups).
        weight_decay (float): Decoupled weight decay magnitude. REQUIRED --
            there is no safe default (``0.0`` silently disables regularisation
            and a nonzero value silently changes the loss being minimised), so
            omitting it raises.
        grad_clip (GradientClipBase | None): Gradient clipping. Default: ``None``.
        lr_ratio (Callable[[Tensor], float] | None): Optional per-parameter
            multiplier applied to the learning rate. Default: ``None``.
        apply_decay_param_fun (Callable[[str], bool] | None): Selects which
            parameters receive weight decay. Default: ``None``.
        multi_precision (bool): Keep FP32 master weights under BF16/FP16
            training. Default: ``False``.
        param_info_map (dict | None): Maps ``param.name`` to optional
            optimizer-specific per-parameter metadata. Default: ``None``.
        name (str | None): Optional optimizer name.
    """

    # V1 (whole tensor on one owner rank) vs V2 (element-wise shard) sharding.
    # Every concrete subclass MUST declare this; the base-class value is a
    # sentinel that makes :meth:`declared_keep_whole` raise.
    #   True  -- only V1 works: the update math needs the whole matrix
    #            (Newton-Schulz, a Frobenius projection) and cannot run on a
    #            shard, so a config asking for V2 is rejected.
    #   None  -- BOTH work: a purely element-wise update is correct on a shard
    #            and on a whole tensor alike, so the config may choose
    #            (default V2, which balances the load across ranks).
    #   False -- only V2 works. Rarely needed, since element-wise math also
    #            runs on a whole tensor: prefer ``None``.
    # Aggregates holding a mix of both (``MixedInnerOptimizer``) declare ``None``
    # and override :meth:`keep_whole_of` to answer per parameter.
    keep_whole: bool | None | str = KEEP_WHOLE_UNDECLARED

    def __init__(
        self,
        learning_rate: float = 0.001,
        parameters: list[Tensor] | None = None,
        weight_decay: float | None = None,
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
        if weight_decay is None:
            raise ValueError(
                f"{type(self).__name__}: 'weight_decay' must be set "
                "explicitly. It has no safe default -- 0.0 silently disables "
                "regularisation and a nonzero value silently changes the "
                "objective -- so pass it (0.0 to turn it off)."
            )
        if grad_clip is not None and not isinstance(
            grad_clip, GradientClipBase
        ):
            raise TypeError(
                "'grad_clip' should be an instance of GradientClipBase's "
                "derived class"
            )
        # NOTE: the three required hooks (_ensure_accumulators,
        # _update_parameters, optimizer_state_layout) are @abstractmethod, so a
        # subclass that forgets one cannot be instantiated at all -- ABCMeta
        # raises here, at construction, instead of at the first checkpoint save
        # (hundreds of steps in) or, worse, silently parsing the state with
        # someone else's suffixes.

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
        # Optional per-parameter metadata. Neither this base class nor the
        # sharding shell interprets any field of it; optimizer-specific fields
        # are read only by that optimizer. It deliberately does NOT define
        # ``_muon_param_info_map`` -- that is ``paddle.optimizer.Muon``'s
        # private/legacy name and must not leak onto every InnerOptimizer
        # subclass.
        self._param_info_map = param_info_map or {}
        # Effective V1/V2 for THIS instance. Defaults to the class declaration
        # (resolving "either" to V2); a config layer may narrow it later with
        # :meth:`set_keep_whole`. Resolved here so a subclass that forgot to
        # declare ``keep_whole`` fails at construction, not much later when the
        # sharding shell starts classifying parameters.
        self._keep_whole = type(self).resolve_keep_whole(None)

    # ------------------------------------------------------------------
    # Sharding declaration (read by the sharding shell)
    # ------------------------------------------------------------------

    @classmethod
    def declared_keep_whole(cls) -> bool | None:
        """Validate and return the class-level :attr:`keep_whole`.

        ``True`` = only V1 (whole tensor on one rank), ``False`` = only V2
        (element-wise slices), ``None`` = either is correct for this update
        math, so the configuration may pick.
        """
        if cls.keep_whole is KEEP_WHOLE_UNDECLARED or not (
            cls.keep_whole is None or isinstance(cls.keep_whole, bool)
        ):
            raise TypeError(
                f"{cls.__name__} must declare a class-level 'keep_whole': "
                f"True = only whole tensor on one rank (V1, required by "
                f"whole-matrix math such as Newton-Schulz or a Frobenius "
                f"projection), None = either V1 or V2 is correct (pure "
                f"element-wise math) and the config picks, False = only "
                f"element-wise slices (V2, rarely needed -- prefer None). "
                f"Got {cls.keep_whole!r}."
            )
        return cls.keep_whole

    @classmethod
    def resolve_keep_whole(cls, requested: bool | None) -> bool:
        """Combine the class declaration with a *requested* V1/V2 choice.

        *requested* is what the configuration asked for (``None`` = no
        preference). Asking for a mode the class does not support raises;
        when the class supports both and nothing was requested the answer is
        V2, which spreads the parameter across ranks.
        """
        declared = cls.declared_keep_whole()
        if requested is None:
            return False if declared is None else declared
        if not isinstance(requested, bool):
            raise TypeError(
                f"keep_whole requested for {cls.__name__} must be a bool or "
                f"None, got {requested!r}."
            )
        if declared is not None and requested != declared:
            raise ValueError(
                f"{cls.__name__} only supports keep_whole={declared!r} "
                f"({'whole tensor on one rank / V1' if declared else 'element-wise slices / V2'}"
                f"), but keep_whole={requested!r} was requested. This follows "
                f"from the update math and is not configurable; remove the "
                f"keep_whole setting (or set it to {declared!r})."
            )
        return requested

    def set_keep_whole(self, requested: bool | None) -> bool:
        """Narrow this instance's V1/V2 to *requested*; returns the result.

        Used by the configuration layer for an optimizer class that supports
        both (``keep_whole = None``). A conflicting request raises.
        """
        self._keep_whole = type(self).resolve_keep_whole(requested)
        return self._keep_whole

    def keep_whole_of(self, param_name: str) -> bool:
        """V1/V2 sharding for *param_name* on this optimizer instance.

        The sharding shell calls this once per parameter while classifying.
        The default answer is this instance's effective ``keep_whole`` for
        every parameter (the class declaration, possibly narrowed by
        :meth:`set_keep_whole`); aggregates (``MixedInnerOptimizer``) override
        it to ask the sub-optimizer that owns *param_name*.
        """
        return self._keep_whole

    # ------------------------------------------------------------------
    # Hooks a subclass must implement (@abstractmethod: a subclass that
    # misses one cannot be instantiated)
    # ------------------------------------------------------------------

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def optimizer_state_layout(self) -> dict:
        """Describe this optimizer's checkpoint state, generically.

        REQUIRED -- there is no default. The suffixes follow from this class's
        own :meth:`_ensure_accumulators`, so only the subclass can state them;
        inheriting someone else's layout would either mis-shard a state tensor
        or fail to parse its name. A subclass that does not override this
        cannot be instantiated (ABCMeta).

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
        has to override this one method. An AdamW-like optimizer declares::

            return {
                "vector": ["moment1_0", "moment2_0"],
                "scalar": ["beta1_pow_acc_0", "beta2_pow_acc_0"],
                "master": "fp32_master_0",
            }

        Over-declaring is harmless (parsing matches by name, so a suffix that is
        never allocated never matches); under-declaring is not, hence no
        inherited default.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement optimizer_state_layout()."
        )

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
