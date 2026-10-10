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
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

from paddle.base import framework

from .inner_optimizer import InnerOptimizer
from .lr import LRScheduler

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle import Tensor

    from ..nn.clip import GradientClipBase

__all__ = []

# Mirror muon.py / inner_optimizer.py: when this flag is on the whole dygraph
# optimizer update is skipped (used by sharding to bypass the local step on
# non-owner ranks).
g_shard_bypass_dygraph_optimizer = int(
    os.environ.get("FLAGS_shard_bypass_dygraph_optimizer", 0)
)


# ----------------------------------------------------------------------
# Inner-optimizer registry
# ----------------------------------------------------------------------
# Maps a short string name (used in YAML / config, e.g. "muon", "adamw") to a
# ``InnerOptimizer`` subclass. A framework layer (ernie5, PaddleFleet, ...) can
# register a custom optimizer WITHOUT touching Paddle source: e.g. PaddleFleet
# defines ``class RMSpropOptimizer(InnerOptimizer)``, calls
# ``register_inner_optimizer("rmsprop", RMSpropOptimizer)`` and the config then
# references ``rmsprop``. ``MixedInnerOptimizer`` looks classes up here to build
# one inner instance per parameter group.
INNER_OPTIMIZER_REGISTRY: dict[str, type] = {}


def register_inner_optimizer(name: str, cls: type) -> None:
    """Register *cls* (an ``InnerOptimizer`` subclass) under *name*.

    Re-registering the same name with the same class is a no-op; re-registering
    with a different class raises to catch accidental shadowing. ``cls`` must be
    an ``InnerOptimizer`` subclass -- validated here so a mis-registered plugin
    fails at registration (early, with a clear message) instead of much later at
    construction / step time.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("inner optimizer name must be a non-empty string.")
    if not (isinstance(cls, type) and issubclass(cls, InnerOptimizer)):
        raise TypeError(
            f"inner optimizer {name!r} must be registered with an "
            f"InnerOptimizer subclass, got {cls!r}."
        )
    existing = INNER_OPTIMIZER_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"inner optimizer name {name!r} is already registered to "
            f"{existing!r}; refusing to shadow it with {cls!r}."
        )
    INNER_OPTIMIZER_REGISTRY[name] = cls


def get_inner_optimizer(name: str) -> type:
    """Return the ``InnerOptimizer`` subclass registered under *name*."""
    try:
        return INNER_OPTIMIZER_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no inner optimizer registered under {name!r}; known names: "
            f"{sorted(INNER_OPTIMIZER_REGISTRY)}. Call "
            "register_inner_optimizer(name, cls) first."
        )


class _GatheredMasterWeights(MutableMapping):
    """Live read/write view over the sub-optimizers' master-weight dicts.

    The authoritative master weights live on each sub-optimizer. This view lets
    the shell / offload / callback code treat ``chain._master_weights`` like a
    single dict: reads gather across subs, writes and deletes route to the sub
    that owns the parameter. Because it delegates (rather than snapshotting),
    ``view[name] = tensor`` actually lands on the owning sub -- a plain merged
    dict would drop the write.
    """

    def __init__(self, chain: MixedInnerOptimizer) -> None:
        self._chain = chain

    def _owner_dict(self, key):
        """Return the sub ``_master_weights`` dict that contains *key*, or None."""
        for sub in self._chain._sub_opts:
            m = getattr(sub, "_master_weights", None)
            if m is not None and key in m:
                return m
        return None

    def __getitem__(self, key):
        m = self._owner_dict(key)
        if m is None:
            raise KeyError(key)
        return m[key]

    def __setitem__(self, key, value):
        # Route to the sub that owns the parameter (raises if unrouted).
        self._chain._set_master_weight(key, value)

    def __delitem__(self, key):
        m = self._owner_dict(key)
        if m is None:
            raise KeyError(key)
        del m[key]

    def __iter__(self):
        seen = set()
        for sub in self._chain._sub_opts:
            for key in getattr(sub, "_master_weights", {}) or {}:
                if key in seen:
                    raise ValueError(
                        "parameter has master weights in more than one "
                        f"sub-optimizer: {key!r}"
                    )
                seen.add(key)
                yield key

    def __len__(self):
        return sum(
            len(getattr(sub, "_master_weights", {}) or {})
            for sub in self._chain._sub_opts
        )

    def __contains__(self, key):
        return self._owner_dict(key) is not None


class MixedInnerOptimizer(InnerOptimizer):
    r"""Route one ``params_grads`` batch to N per-group inner optimizers.

    ``MixedInnerOptimizer`` is the *single* optimizer object handed to the
    sharding shell (PaddleFleet's ``MixedShardingOptimizer``). It owns a list
    of inner ``InnerOptimizer`` instances -- one per parameter group -- each
    constructed with that group's own hyperparameters (momentum / betas /
    epsilon / ns_steps / weight_decay / ...). This is how truly per-group
    hyperparameters are achieved: the values are frozen into each
    sub-optimizer at construction time, instead of being collapsed to a single
    global value.

    The shell calls :meth:`_apply_optimize` exactly once per step with the local
    (whole-or-sharded) mixed ``params_grads``. This class applies the global
    gradient clip once (so clipping stays bit-identical to a single optimizer),
    then splits the batch by ``param.name`` and forwards each subset to the
    owning sub-optimizer.

    Contract required by the sharding shell:
        * ``_apply_optimize`` -- routed update (this class).
        * ``_param_info_map`` -- generic union over all groups; the shell reads
          each entry's ``keep_whole`` to decide V1 (whole) vs V2 (element-wise)
          sharding.
        * ``state_dict`` / ``set_state_dict`` -- aggregate / route over subs.
        * ``_master_weights`` -- a complete view gathered from the owning sub
          optimizers for shell/checkpoint consumers.
        * ``_multi_precision`` / ``_lr_ratio`` -- read via the shell's
          ``__getattr__`` (grad-clip info, ZCC / reshard, per-param lr).

    CRITICAL: this class MUST NOT expose an ``_inner_opt`` attribute. The shell's
    ``_set_inner_opt_attr`` walks the ``_inner_opt`` chain doing ``setattr``; if
    ``MixedInnerOptimizer`` had ``_inner_opt`` the walk would clobber a
    sub-optimizer's ``_parameter_list``. Sub-optimizers are stored in
    ``_sub_opts`` instead.
    """

    def __init__(
        self,
        sub_optimizers: list,
        learning_rate=0.001,
        lr_ratio: Callable[[Tensor], float] | None = None,
        grad_clip: GradientClipBase | None = None,
        apply_decay_param_fun: Callable[[str], bool] | None = None,
        multi_precision: bool = True,
        param_info_map: dict | None = None,
        name: str | None = None,
    ) -> None:
        if not sub_optimizers:
            raise ValueError(
                "MixedInnerOptimizer requires a non-empty list of "
                "sub-optimizers."
            )
        for sub in sub_optimizers:
            if not (
                hasattr(sub, "_apply_optimize")
                and callable(sub._apply_optimize)
            ):
                raise TypeError(
                    "each sub-optimizer must implement _apply_optimize(), got "
                    f"{type(sub)!r}."
                )

        # Union parameter list (flat), preserving each sub's order.
        param_list: list = []
        route: dict = {}
        for sub in sub_optimizers:
            for p in sub._parameter_list:
                if p.name in route:
                    raise ValueError(
                        f"parameter {p.name!r} is owned by more than one "
                        "sub-optimizer; groups must be mutually exclusive."
                    )
                route[p.name] = sub
                param_list.append(p)

        # Merge per-parameter info maps (union) using the GENERIC name. A sub
        # exposes it as ``_param_info_map`` (every InnerOptimizer subclass, e.g.
        # the concrete Muon / AdamW inner optimizers) -- we do NOT read Muon's
        # private ``_muon_param_info_map`` here. An explicit full map, when
        # provided, wins (guarantees the shell sees every parameter).
        merged_info: dict = {}
        for sub in sub_optimizers:
            merged_info.update(getattr(sub, "_param_info_map", {}) or {})
        if param_info_map:
            merged_info.update(param_info_map)

        super().__init__(
            learning_rate=learning_rate,
            parameters=param_list,
            weight_decay=0.0,
            grad_clip=grad_clip,
            lr_ratio=lr_ratio,
            apply_decay_param_fun=apply_decay_param_fun,
            multi_precision=multi_precision,
            param_info_map=merged_info,
            name=name,
        )

        # NOTE: intentionally NOT ``self._inner_opt`` (see class docstring).
        self._sub_opts = list(sub_optimizers)
        self._route = route

        for sub in self._sub_opts:
            # The chain owns clipping; sub-optimizers must not clip again.
            _null = getattr(sub, "_set_grad_clip_none", None)
            if callable(_null):
                _null()
            else:
                sub._grad_clip = None
        self._multi_precision = multi_precision

    @property
    def _master_weights(self):
        """A live read/write view over the sub-optimizers' master weights.

        Reads gather from every sub (the single source of truth stays on the
        subs); writes / deletes route to the owning sub. Returning a WRITE-
        THROUGH view (not a throwaway merged dict) means offload / callback code
        that does ``opt._master_weights[name] = tensor`` actually lands on the
        owning sub instead of silently mutating a temporary copy.
        """
        view = self.__dict__.get("_mw_view")
        if view is None:
            view = _GatheredMasterWeights(self)
            self.__dict__["_mw_view"] = view
        return view

    @_master_weights.setter
    def _master_weights(self, value):
        # InnerOptimizer.__init__ assigns ``self._master_weights = {}`` before
        # _sub_opts exists. That value is not authoritative (the subs own the
        # real dicts), so ignore whole-dict assignment; use the write-through
        # view (or _set_master_weight) for per-key updates.
        return None

    def _set_master_weight(self, param_name, tensor):
        sub = self._route.get(param_name)
        if sub is None:
            raise KeyError(
                f"parameter {param_name!r} has no sub-optimizer route"
            )
        sub_master = getattr(sub, "_master_weights", None)
        if sub_master is None:
            raise AttributeError(
                f"sub-optimizer for parameter {param_name!r} has no "
                "_master_weights store"
            )
        sub_master[param_name] = tensor

    def _create_master_weight(self, param):
        """Route master-weight creation to the sub that owns ``param``.

        The sharding shell may call this method through its inner optimizer
        interface when it recreates parameter storage. MixedInnerOptimizer must
        not use ``InnerOptimizer._create_master_weight`` for the mixed optimizer,
        because the owning sub-optimizer is the component that knows whether and
        how this parameter needs a master weight.
        """
        sub = self._route.get(param.name)
        if sub is None:
            raise KeyError(
                f"parameter {param.name!r} has no sub-optimizer route"
            )
        create = getattr(sub, "_create_master_weight", None)
        if not callable(create):
            raise AttributeError(
                f"sub-optimizer for parameter {param.name!r} does not "
                "implement _create_master_weight()"
            )
        return create(param)

    # ------------------------------------------------------------------
    # Update entry point (called by the sharding shell once per step)
    # ------------------------------------------------------------------

    def _apply_optimize(self, loss, startup_program, params_grads):
        """Clip once over ALL params, then route each subset to its sub."""
        if not framework.in_dygraph_mode():
            raise NotImplementedError(
                "MixedInnerOptimizer only supports dygraph mode."
            )
        if g_shard_bypass_dygraph_optimizer:
            return

        # Global gradient clip, exactly once, over the full mixed batch -- keeps
        # the clipped-norm bit-identical to the single-optimizer path. Each
        # sub-optimizer is constructed with grad_clip=None (set in __init__), so
        # it will NOT clip again.
        if self._grad_clip is not None:
            params_grads = self._grad_clip(params_grads)

        # Split by owning sub-optimizer, preserving input order within a bucket.
        buckets: dict = {id(sub): (sub, []) for sub in self._sub_opts}
        for param, grad in params_grads:
            if grad is None:
                continue
            sub = self._route.get(param.name)
            if sub is None:
                raise KeyError(
                    f"parameter {param.name!r} has no sub-optimizer route; "
                    "every trainable parameter must belong to exactly one "
                    "group."
                )
            buckets[id(sub)][1].append((param, grad))

        for sub, sub_params_grads in buckets.values():
            if sub_params_grads:
                sub._apply_optimize(
                    loss=None,
                    startup_program=None,
                    params_grads=sub_params_grads,
                )

        # After the sub-optimizers have created/updated their accumulators,
        # (re)build the single union fused buffer if fused storage is enabled
        # (ZCC). No-op otherwise.
        self._maybe_refuse()

    # ------------------------------------------------------------------
    # State dict (checkpoint save / load) -- aggregate over sub-optimizers
    # ------------------------------------------------------------------

    @framework.dygraph_only
    def state_dict(self) -> dict:
        """Union of every sub's optimizer state + gathered master weights.

        Each sub's own ``state_dict()`` is reused so the standard
        ``_accumulators`` / ``_accumulators_holder`` fallback is honoured: after
        ``load -> (no step) -> save`` a sub's ``_accumulators`` is still empty and
        the real tensors live in ``_accumulators_holder``; delegating to the sub
        picks them up (a raw ``_accumulators`` scan would silently drop them).

        Keys mirror ``Optimizer.state_dict`` (``<var.name>`` -> tensor), so the
        shell's ``sharded_state_dict`` can parse them with the suffixes declared
        by :meth:`optimizer_state_layout` (union over the subs). Per-sub
        ``master_weights`` / ``LR_Scheduler`` are dropped here and replaced by the
        gathered master weights and the chain scheduler state.
        """
        sd: dict = {}
        merged_master_weights: dict = {}
        for sub in self._sub_opts:
            sub_sd = sub.state_dict()
            sub_master_weights = sub_sd.pop("master_weights", None)
            if sub_master_weights:
                owned_names = {p.name for p in sub._parameter_list}
                for param_name, master_weight in sub_master_weights.items():
                    if param_name not in owned_names:
                        continue
                    if param_name in merged_master_weights:
                        raise ValueError(
                            "parameter has master weights in more than one "
                            f"sub-optimizer: {param_name!r}"
                        )
                    merged_master_weights[param_name] = master_weight
            sub_sd.pop("LR_Scheduler", None)
            sd.update(sub_sd)
        if merged_master_weights:
            sd["master_weights"] = merged_master_weights
        if isinstance(self._learning_rate, LRScheduler):
            sd["LR_Scheduler"] = self._learning_rate.state_dict()
        return sd

    @framework.dygraph_only
    def set_state_dict(self, state_dict: dict) -> None:
        """Route accumulator tensors and master weights back to their sub.

        The scheduler is restored once; because the (single or per-group)
        scheduler objects are shared with the sub-optimizers, that also restores
        each sub's ``_learning_rate`` in place.
        """
        sd = dict(state_dict)
        lr_sd = sd.pop("LR_Scheduler", None)
        master = sd.pop("master_weights", None)

        # Restore the scheduler once. Whether it is a single shared scheduler or
        # a ChainedLRScheduler over per-group schedulers, those scheduler objects
        # are SHARED with the sub-optimizers (each sub's ``_learning_rate`` IS one
        # of them), so this single call also restores every sub's lr in place.
        if lr_sd is not None and isinstance(self._learning_rate, LRScheduler):
            self._learning_rate.set_state_dict(lr_sd)

        # Route each master weight back to the sub that owns the parameter.
        if master:
            for pname, tensor in master.items():
                if pname in self._route:
                    self._set_master_weight(pname, tensor)

        # Accumulator vars are keyed by ``<param.name>_<acc>...`` -> match the
        # longest owning param-name prefix (handles names that are prefixes of
        # each other, e.g. ``linear_0`` vs ``linear_0_extra``).
        names_by_len = sorted(self._route, key=len, reverse=True)
        per_sub_acc: dict = {id(sub): {} for sub in self._sub_opts}
        for key, tensor in sd.items():
            for pname in names_by_len:
                if key.startswith(pname):
                    per_sub_acc[id(self._route[pname])][key] = tensor
                    break

        for sub in self._sub_opts:
            sub_state = dict(per_sub_acc[id(sub)])
            # base Optimizer.set_state_dict asserts on a missing "LR_Scheduler"
            # when the sub's learning_rate is an LRScheduler. That scheduler was
            # already restored in place above (shared object), so hand back its
            # OWN current state. No cross-group routing is needed, which is
            # exactly why an empty group on this rank cannot mis-route another
            # group's scheduler state.
            if isinstance(sub._learning_rate, LRScheduler):
                sub_state["LR_Scheduler"] = sub._learning_rate.state_dict()
            sub.set_state_dict(sub_state)

    # ------------------------------------------------------------------
    # Fused optimizer-state storage (ZCC) -- ONE buffer over ALL groups
    # ------------------------------------------------------------------
    #
    # Zero-Cost Checkpoint (ZCC) snapshots ONE fused optimizer-state buffer via
    # ``fused_states_*_meta``. If each per-group sub-optimizer fused separately
    # there would be N buffers (which ZCC's single-buffer helper cannot consume).
    # So the chain owns ONE ``FusionStorage`` built over the UNION of every sub's
    # accumulators + the gathered master weights. ``FusionStorage.mapping_tensor``
    # rebinds those (live) tensor objects -- the very ones each sub-optimizer
    # holds in its ``_accumulators`` -- to slices of the single buffer, so the
    # sub-optimizers keep updating straight into it. The inherited
    # ``fused_states_*`` properties (which read ``self.fusion_storage``) then
    # expose the one buffer to ZCC.

    def use_fusion_storage(self):
        """Enable a SINGLE chain-owned fused buffer (do NOT fuse per sub-optimizer).

        Only sets the chain's own flags; sub-optimizers are intentionally left
        unfused so there is exactly one fused buffer (built lazily in
        :meth:`_maybe_refuse` once the sub accumulators exist).
        """
        self._use_fusion_storage = True
        self.need_refuse()

    def _maybe_refuse(self):
        """Build / refresh the single union ``FusionStorage`` (dygraph only).

        Mirrors ``Optimizer._maybe_refuse`` but over the merged view instead of
        a single optimizer's ``_accumulators``. Rebuilds when a tensor is no
        longer a view into the current buffer (e.g. a sub-optimizer re-allocated
        an accumulator), matching the version/refuse self-heal of the base class.
        """
        if not getattr(self, "_use_fusion_storage", False):
            return
        if not framework.in_dygraph_mode():
            return
        from .fusion_utils import FusionStorage

        merged = self.merged_accumulators()
        if self.fusion_storage is not None:
            buf = self.fusion_storage.buffer
            for _acc, pmap in merged.items():
                for _pn, vv in pmap.items():
                    if not vv._is_shared_buffer_with(buf):
                        self.need_refuse()
            for _pn, vv in self._master_weights.items():
                if not vv._is_shared_buffer_with(buf):
                    self.need_refuse()
        if not self._need_refuse:
            return
        # FusionStorage requires a plain ``dict`` for master weights; our
        # ``_master_weights`` is a live gathered view (MutableMapping), so
        # materialise a snapshot ``{name: tensor}`` (the tensors are the live
        # sub-optimizer objects, which FusionStorage rebinds into the buffer).
        self.fusion_storage = FusionStorage(
            merged, dict(self._master_weights), self.merged_model_params
        )
        self._fuse_buffer_version += 1
        self.reset_need_refuse()

    def _create_accumulators(self, block, parameters):
        """Route accumulator creation to the owning sub-optimizer.

        Called by ``init_optimizer`` during flex/ZCC resume, BEFORE
        ``dist.load_state_dict``, to allocate the target tensors that the loader
        writes into. ``parameters`` mixes 2D whole params and 1D slice params
        (their ``.name`` is the original param name, so routing works). Each sub
        creates its own group's moment / beta_pow / master targets; without this
        (a no-op) resume would have no targets and silently fail.
        """
        buckets: dict = {id(sub): (sub, []) for sub in self._sub_opts}
        for p in parameters:
            sub = self._route.get(p.name)
            if sub is not None:
                buckets[id(sub)][1].append(p)
        for sub, sub_params in buckets.values():
            if sub_params:
                sub._create_accumulators(block, sub_params)

    # ------------------------------------------------------------------
    def merged_accumulators(self) -> dict:
        """Return a merged ``{acc_name: {param_name: tensor}}`` snapshot.

        A fresh dict is built on each call (the authoritative tensors stay in the
        sub-optimizers), so it is suitable for READ-only consumers such as the
        fusion-storage builder. Consumers that need to mutate state in place
        should update the owning sub-optimizer's ``_accumulators`` or
        ``_master_weights`` directly.
        """
        from collections import defaultdict

        merged: dict = defaultdict(dict)
        for sub in self._sub_opts:
            for acc_name, param_map in sub._accumulators.items():
                merged[acc_name].update(param_map)
        return merged

    def optimizer_state_layout(self) -> dict:
        """Aggregate the per-group layouts into one (union suffixes).

        The framework-layer checkpoint parsers apply ONE suffix vocabulary
        globally, so we union each sub's ``vector`` / ``scalar`` suffixes and
        require a single ``master`` suffix. Groups may differ in WHICH
        suffixes they use (a Muon-only group has no moment2 for its params, a
        plain-AdamW group does) -- the union is safe because parsing tries each
        suffix and only the ones actually present in a var name match.
        """
        vector: list = []
        scalar: list = []
        master = None
        for sub in self._sub_opts:
            layout = sub.optimizer_state_layout()
            for s in layout.get("vector", []):
                if s not in vector:
                    vector.append(s)
            for s in layout.get("scalar", []):
                if s not in scalar:
                    scalar.append(s)
            m = layout.get("master")
            if master is None:
                master = m
            elif m is not None and m != master:
                raise ValueError(
                    "sub-optimizers disagree on the master-weight suffix: "
                    f"{master!r} vs {m!r}; a single checkpoint parser cannot "
                    "handle two master suffixes."
                )
        return {
            "vector": vector,
            "scalar": scalar,
            "master": master or "fp32_master_0",
        }
